"""Parse a skillpack bundle's skill manifest.

A bundle has no root manifest file — instead, every skill is auto-discovered
by scanning `<skill>/skill.yaml` at the bundle root. Each skill.yaml is the
source of truth for that skill's entrypoint, schemas, model, tools, etc.

`skill.yaml` shape:

    title: Human-friendly name    # optional, shown above description in UI
    description: short text
    entrypoint: SKILL.md          # agentic loop (system prompt = file)
                                  # OR "main.py:run" for a deterministic skill
    model: claude/sonnet-4-6      # optional, agentic only — see docs/models
    disable_bash: false           # optional, agentic only
    input_schema:  { Puras dialect } # required, enforced before run
    output_schema: { Puras dialect } # required, enforced after run
    examples:                        # optional, 0..N playground seed scenarios
      - title: short label           #   optional — falls back to "Example N"
        description: 1-line note     #   optional
        inputs: { fully-formed input matching input_schema }
    tools:                           # optional, agentic only
      - name: foo
        description: ...
        entrypoint: tools/foo.py:run # path relative to skill dir
        input_schema:  { Puras dialect }
        output_schema: { Puras dialect }

Top-level skill name = the directory name (must be slug-style).

## Subskills

A top-level skill can have *subskills* nested under `<X>/subskills/<Y>/`.
They use the same `skill.yaml` format, but they're treated differently:

  - Their qualified name is `<X>/<Y>` (parent / sub).
  - They are hidden from the API submit endpoint, the public explore listing,
    and the playground — i.e. they can't be invoked as top-level skills.
  - They are callable ONLY from their parent skill's runtime, via
    `puras.subagent.run("<Y>", ...)`. The `/v1/subagent/invoke` resolver tries
    `<parent>/<Y>` first when the caller is a parent skill, so the parent
    references its subskills by their bare name.

Use subskills for pipeline-internal helpers that don't make sense to
publish — research stages, render stages, etc. Use top-level skills for
anything you'd want callable from elsewhere (MCP, dashboard, marketplace).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .llm_models import PUBLIC_SLUG_RE, is_known_slug

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

_MD_EXT = ".md"
_PY_ENTRYPOINT = re.compile(r"^(?P<file>[\w./\-]+\.py):(?P<func>[A-Za-z_][A-Za-z0-9_]*)$")

# Recognized top-level skill.yaml keys — kept in sync with the canonical set in
# api/app/manifest.py (which the docs generator reads). `_build_decl` rejects
# anything else so typos fail loudly. `model` is handled separately below.
TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "title",
        "description",
        "entrypoint",
        "text_model",
        "image_model",
        "video_model",
        "audio_model",
        "disable_bash",
        "input_schema",
        "output_schema",
        "tools",
        "examples",
        "evals",
        "routing",
        "marketing",
        "allowed_tools",
        "tool_limits",
    }
)


class ManifestError(ValueError):
    pass


@dataclass
class ToolDecl:
    """A tool declared in an agentic skill's `tools:` list.

    Two shapes:
      - **Local Python tool**: `entrypoint` + `input_schema` + `output_schema`
        are all set; the tool runs as a function in this skill's deployment.
      - **Skill tool**: `skill_ref` is set (a bare skill name, must resolve in
        this same deployment). The tool dispatches via `/v1/subagent/invoke`;
        schemas are copied from the target skill at load time so the agent
        sees the right `input_schema`. Use this to give an agentic skill
        other skills (top-level OR its own subskills) as callable tools.
    """

    name: str
    description: str
    # Local Python tool: dotted "<file.py>:<func>". Empty string for skill tools.
    entrypoint: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    # Skill tool: bare skill name to invoke. Resolution is done at load time
    # against the same deployment's manifest (subskill-first, then top-level).
    # None for local Python tools.
    skill_ref: str | None = None
    # `confirm: true` — a side-effectful tool (send, publish, delete, pay) that
    # requires explicit HUMAN approval before each call. The worker pauses the
    # run and waits for an approve/deny decision; the gate is enforced at the
    # dispatcher off this deploy-time flag, so neither the model nor injected
    # content can bypass it. See agent_runner + api/app/routers/approvals.py.
    confirm: bool = False

    @property
    def is_skill_tool(self) -> bool:
        return self.skill_ref is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "entrypoint": self.entrypoint,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "skill_ref": self.skill_ref,
            "confirm": self.confirm,
        }


@dataclass
class SkillExample:
    inputs: dict[str, Any]
    title: str | None = None
    description: str | None = None
    # Optional pre-computed result shaped like `output_schema`. Display-only
    # (playground/marketplace preview); the worker never executes against it.
    outputs: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": self.inputs,
            "title": self.title,
            "description": self.description,
            "outputs": self.outputs,
        }


@dataclass
class EvalDecl:
    """One grader in an agentic skill's `evals:` list. Evals are to a skill what
    unit tests are to code: each grader scores a run's output in [0,1]; the
    weighted mean (×100) is the run's `eval_score`. Four kinds:

      - kind="check": a deterministic Python grader. `entrypoint`
        ("<file.py>:<func>", relative to the skill dir, same as a tool) is called
        with `(inputs, output)` and returns `{score, passed, detail}` — the
        objective, unit-test layer (limits, counts, schema-shape assertions).
      - kind="rubric": an LLM-as-judge grader. `criteria` (+ optional anchored
        `levels`, a {"0": "...", "1": "..."} map) is handed to the skill's text
        model, which returns a 0..1 score with reasoning — the qualitative layer
        (voice, fidelity, language).
      - kind="exact_match": deterministic, free. Compares the run's output to the
        case's `expected` (from the eval dataset). `field` (optional dotted path
        like "label" or "result.category") narrows the comparison to one value;
        omit it to compare the whole output. Only runs in an eval suite where the
        case carries an `expected`; skipped on a live run.
      - kind="schema": deterministic, free. Validates the output against a JSON
        Schema. `schema` (Puras-dialect mapping) gives an explicit shape; omit it
        to validate against the skill's own `output_schema`.
    """

    name: str
    kind: str                      # "check" | "rubric" | "exact_match" | "schema"
    weight: float = 1.0
    entrypoint: str = ""           # check only: "<file.py>:<func>"
    criteria: str = ""             # rubric only: what the judge scores
    levels: dict[str, str] = field(default_factory=dict)  # rubric only: anchored score → meaning
    # `schema` is declared before `field` ON PURPOSE: a dataclass attribute named
    # `field` shadows the imported `dataclasses.field` for every line after it in
    # the class body, so any `= field(...)` below it would crash at class-build.
    schema: dict[str, Any] = field(default_factory=dict)  # schema only: explicit shape (else skill output_schema)
    field: str = ""                # exact_match only: dotted path into output/expected

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "weight": self.weight,
            "entrypoint": self.entrypoint,
            "criteria": self.criteria,
            "levels": self.levels,
            "field": self.field,
            "schema": self.schema,
        }


@dataclass
class SkillDecl:
    name: str                      # = directory name (or "<parent>/<dir>" for subskills)
    path: str                      # bundle-relative dir (e.g. "foo")
    description: str
    entrypoint: str                # "SKILL.md" or "main.py:run"
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    title: str | None = None       # optional human-friendly display name
    model: str | None = None       # LLM slug like "claude/sonnet-4-6" (skill.yaml `text_model:`), agentic only
    # Default media model/family per kind — the generate_image / generate_video /
    # generate_audio verbs resolve to these when a call doesn't pin a model.
    image_model: str | None = None
    video_model: str | None = None
    audio_model: str | None = None
    # Which media kinds this skill generates (detected at deploy from the
    # entrypoint/source), so the playground knows which model pickers to show.
    media_kinds: list[str] = field(default_factory=list)
    disable_bash: bool = False
    tools: list[ToolDecl] = field(default_factory=list)
    examples: list[SkillExample] = field(default_factory=list)
    # Graders that score each finished run's output (agentic only). Optional —
    # a skill with no `evals:` simply produces no eval_score. See EvalDecl.
    evals: list[EvalDecl] = field(default_factory=list)
    # Optional offline eval dataset: a bundle-relative (to the skill dir) path to
    # a JSONL file of cases `{id, inputs, expected?, tags?}`. Powers the offline
    # eval suite (POST /v1/skillpacks/{id}/evals). None = no dataset; the graders
    # still score live runs. See `evals.dataset` in skill.yaml.
    eval_dataset: str | None = None
    # Optional eval-time tool mocks (skill.yaml `evals.mocks` — `{tool: response}`):
    # in SUITE/test mode (an eval suite, never a live run) the named tools return
    # the given canned response instead of really executing, so a test run can't
    # trigger real side effects (renders, sends, writes). Built-in side-effecting
    # verbs get a safe default stub even with no entry; an entry overrides it and
    # is the only way to mock a custom tool. {} = none. See `worker.eval_mocks`.
    eval_mocks: dict[str, Any] = field(default_factory=dict)
    # Optional model routing/escalation (skill.yaml `routing:`, agentic only): run
    # on the cheap `text_model` by default and switch to a premium model when the
    # cheap one can't deliver (e.g. repeated set_output schema failures). Shape:
    # `{escalate_to: <slug>, on: ["schema_fail"], after: <int>}`. None = no
    # escalation. See `_parse_routing`.
    routing: dict[str, Any] | None = None
    # Least-privilege tool scope (skill.yaml `allowed_tools:`, agentic only): a
    # whitelist of tool names this skill may use — built-ins AND its own declared
    # tools are filtered to this set, and a call to anything outside it is refused
    # at dispatch (defense in depth). None = no restriction (every tool offered).
    allowed_tools: list[str] | None = None
    # Per-run, per-tool call caps (skill.yaml `tool_limits:` — `{tool: max}`): a
    # tool that has already been called `max` times this run is refused (the model
    # gets a soft error it can react to). Guards runaway loops / cost. {} = none.
    tool_limits: dict[str, int] = field(default_factory=dict)
    # Set for subskills (nested under `<parent>/subskills/<X>/`). Subskills
    # are hidden from submit/list/MCP and only callable from their parent's
    # runtime via `puras.subagent.run("<X>", ...)`. None for top-level skills.
    parent_skill: str | None = None
    # Optional SEO/landing copy (skill.yaml `marketing:`) for the skill's public
    # page. Display-only; never read by the runtime — carried through here only
    # so a worker re-parse → re-store round-trip doesn't drop it. See
    # _parse_marketing.
    marketing: dict[str, Any] | None = None

    @property
    def is_agentic(self) -> bool:
        return self.entrypoint.lower().endswith(_MD_EXT)

    @property
    def is_subskill(self) -> bool:
        return self.parent_skill is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "title": self.title,
            "description": self.description,
            "entrypoint": self.entrypoint,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "model": self.model,
            "image_model": self.image_model,
            "video_model": self.video_model,
            "audio_model": self.audio_model,
            "media_kinds": self.media_kinds,
            "disable_bash": self.disable_bash,
            "tools": [t.to_dict() for t in self.tools],
            "examples": [e.to_dict() for e in self.examples],
            "evals": [e.to_dict() for e in self.evals],
            "eval_dataset": self.eval_dataset,
            "eval_mocks": self.eval_mocks,
            "routing": self.routing,
            "allowed_tools": self.allowed_tools,
            "tool_limits": self.tool_limits,
            "is_agentic": self.is_agentic,
            "parent_skill": self.parent_skill,
            "marketing": self.marketing,
        }


@dataclass
class Manifest:
    skills: list[SkillDecl] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"skills": [s.to_dict() for s in self.skills]}

    def skill(self, name: str) -> SkillDecl | None:
        for s in self.skills:
            if s.name == name:
                return s
        return None


def parse_bundle_dir(bundle_root: Path) -> Manifest:
    """Scan a skillpack bundle directory for `<skill>/skill.yaml` + subskill
    `<skill>/subskills/<sub>/skill.yaml` files.

    The skillpack itself is the bundle root; each top-level directory that
    contains a `skill.yaml` is a skill. No `skills/` wrapper.
    """
    skills: list[SkillDecl] = []
    for entry in sorted(bundle_root.iterdir()):
        if not entry.is_dir():
            continue
        yaml_path = entry / "skill.yaml"
        if not yaml_path.is_file():
            # Top-level dirs without a skill.yaml are just bundle scaffolding
            # (lib/, docs/, .git/, etc.) — skip silently.
            continue
        if not _SLUG.match(entry.name):
            raise ManifestError(f"{entry.name}: dir name must be slug-style")
        decl = _parse_skill_yaml(entry.name, entry, yaml_path)
        skills.append(decl)

        # Subskills: `<parent>/subskills/<sub>/skill.yaml`. Same format,
        # but the parsed SkillDecl is recorded under the qualified name
        # "<parent>/<sub>" with parent_skill set. These are runtime-private to
        # the parent — hidden from submit/list/MCP.
        subskills_dir = entry / "subskills"
        if subskills_dir.is_dir():
            for sub_entry in sorted(subskills_dir.iterdir()):
                if not sub_entry.is_dir():
                    continue
                sub_yaml = sub_entry / "skill.yaml"
                if not sub_yaml.is_file():
                    raise ManifestError(
                        f"{entry.name}/subskills/{sub_entry.name}: missing skill.yaml"
                    )
                if not _SLUG.match(sub_entry.name):
                    raise ManifestError(
                        f"{entry.name}/subskills/{sub_entry.name}: "
                        f"dir name must be slug-style"
                    )
                qual = f"{entry.name}/{sub_entry.name}"
                sub_decl = _parse_skill_yaml(qual, sub_entry, sub_yaml)
                sub_decl.path = f"{entry.name}/subskills/{sub_entry.name}"
                sub_decl.parent_skill = entry.name
                skills.append(sub_decl)

    if not skills:
        raise ManifestError("bundle has no skills (looking for `<skill>/skill.yaml`)")

    seen: set[str] = set()
    for s in skills:
        if s.name in seen:
            raise ManifestError(f"duplicate skill name: {s.name}")
        seen.add(s.name)

    return Manifest(skills=skills)


def parse_bundle_zip(zf) -> Manifest:
    """Read a zipfile.ZipFile and parse the same way as a dir.

    Used by the API on push() — we don't extract to disk, we just read the
    yaml files in-memory to validate the bundle and produce the manifest.

    Bundle layout (flat, no `skills/` wrapper):
      <skill>/skill.yaml                              (top-level)
      <skill>/subskills/<sub>/skill.yaml              (subskill)
    """
    top_blobs: dict[str, bytes] = {}
    sub_blobs: dict[str, bytes] = {}  # key = "parent/sub"
    for name in zf.namelist():
        parts = name.split("/")
        # Top-level: <slug>/skill.yaml  (exactly 2 segments)
        if len(parts) == 2 and parts[1] == "skill.yaml" and parts[0]:
            slug = parts[0]
            if not _SLUG.match(slug):
                raise ManifestError(f"{slug}: dir name must be slug-style")
            top_blobs[slug] = zf.read(name)
            continue
        # Subskill: <parent>/subskills/<sub>/skill.yaml  (exactly 4 segments)
        if (
            len(parts) == 4
            and parts[1] == "subskills"
            and parts[3] == "skill.yaml"
            and parts[0]
            and parts[2]
        ):
            parent_slug, sub_slug = parts[0], parts[2]
            if not _SLUG.match(parent_slug):
                raise ManifestError(f"{parent_slug}: dir name must be slug-style")
            if not _SLUG.match(sub_slug):
                raise ManifestError(
                    f"{parent_slug}/subskills/{sub_slug}: dir name must be slug-style"
                )
            sub_blobs[f"{parent_slug}/{sub_slug}"] = zf.read(name)

    if not top_blobs:
        raise ManifestError("bundle has no skills (looking for `<skill>/skill.yaml`)")

    skills: list[SkillDecl] = []
    for slug in sorted(top_blobs.keys()):
        decl = _parse_skill_yaml_bytes(slug, slug, top_blobs[slug])
        skills.append(decl)

    for qual in sorted(sub_blobs.keys()):
        parent_slug, sub_slug = qual.split("/", 1)
        if parent_slug not in top_blobs:
            raise ManifestError(
                f"{parent_slug}/subskills/{sub_slug}: parent skill "
                f"`{parent_slug}` not found"
            )
        sub_decl = _parse_skill_yaml_bytes(
            qual,
            f"{parent_slug}/subskills/{sub_slug}",
            sub_blobs[qual],
        )
        sub_decl.parent_skill = parent_slug
        skills.append(sub_decl)

    return Manifest(skills=skills)


# --- per-skill parser -------------------------------------------------------


def _parse_skill_yaml(slug: str, skill_dir: Path, yaml_path: Path) -> SkillDecl:
    data = yaml.safe_load(yaml_path.read_text("utf-8"))
    # Bundles are flat: the skill dir sits directly at <bundle_root>/<slug>,
    # so the bundle-relative path is just the slug (no `skills/` wrapper).
    # Must match parse_bundle_zip's `_parse_skill_yaml_bytes(slug, slug, ...)`
    # — otherwise the worker re-parses a deployment from disk and stamps a
    # stale `skills/<slug>` path that no longer resolves.
    decl = _build_decl(slug, slug, data)
    _verify_entrypoint_file_exists(decl, skill_dir)
    return decl


def _parse_skill_yaml_bytes(slug: str, path: str, blob: bytes) -> SkillDecl:
    data = yaml.safe_load(blob.decode("utf-8"))
    return _build_decl(slug, path, data)


def _build_decl(slug: str, path: str, data: Any) -> SkillDecl:
    if not isinstance(data, dict):
        raise ManifestError(f"skills/{slug}/skill.yaml: must be a mapping")

    # Reject unrecognized top-level keys (typos, stray fields). `model` is let
    # through here — it has its own legacy handling below.
    unknown = set(data.keys()) - TOP_LEVEL_KEYS - {"model"}
    if unknown:
        allowed = ", ".join(sorted(TOP_LEVEL_KEYS))
        bad = ", ".join(repr(k) for k in sorted(unknown))
        raise ManifestError(
            f"skills/{slug}: unknown top-level key(s) {bad} in skill.yaml — "
            f"allowed: {allowed} (see /docs/skill-yaml-reference)"
        )

    entrypoint = data.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        raise ManifestError(f"skills/{slug}: `entrypoint` required")
    entrypoint = entrypoint.strip()

    is_agentic = entrypoint.lower().endswith(_MD_EXT)
    if not is_agentic and not _PY_ENTRYPOINT.match(entrypoint):
        raise ManifestError(
            f"skills/{slug}: entrypoint `{entrypoint}` must be a .md file "
            f"or `<file.py>:<func>` form"
        )

    title_raw = data.get("title")
    if title_raw is not None and not isinstance(title_raw, str):
        raise ManifestError(f"skills/{slug}: `title` must be a string")
    title = title_raw.strip() if isinstance(title_raw, str) else None
    if title == "":
        title = None

    description = str(data.get("description") or "").strip()
    input_schema = data.get("input_schema")
    output_schema = data.get("output_schema")
    if not isinstance(input_schema, dict):
        raise ManifestError(f"skills/{slug}: `input_schema` (object) required")
    if not isinstance(output_schema, dict):
        raise ManifestError(f"skills/{slug}: `output_schema` (object) required")

    model = data.get("model")
    if model is not None:
        if not isinstance(model, str):
            raise ManifestError(f"skills/{slug}: `model` must be a string")
        if ":" in model:
            raise ManifestError(
                f"skills/{slug}: `model` no longer accepts a `provider:` prefix "
                f"(got {model!r}); use a public slug like `claude/sonnet-4-6` — "
                f"see docs/models"
            )
        if not PUBLIC_SLUG_RE.match(model):
            raise ManifestError(
                f"skills/{slug}: `model` must be a `family/variant` slug "
                f"(got {model!r}); see docs/models for the list"
            )
        if not is_known_slug(model):
            raise ManifestError(
                f"skills/{slug}: unknown model `{model}` — see docs/models for the list"
            )
    if model is not None and not is_agentic:
        raise ManifestError(
            f"skills/{slug}: `model` only applies to agentic (.md) skills"
        )

    disable_bash = bool(data.get("disable_bash", False))
    if disable_bash and not is_agentic:
        raise ManifestError(
            f"skills/{slug}: `disable_bash` only applies to agentic skills"
        )

    tools_raw = data.get("tools") or []
    if not isinstance(tools_raw, list):
        raise ManifestError(f"skills/{slug}: `tools` must be a list")
    if tools_raw and not is_agentic:
        raise ManifestError(
            f"skills/{slug}: `tools` only apply to agentic (.md) skills"
        )

    tools: list[ToolDecl] = []
    seen_tool: set[str] = set()
    for i, t in enumerate(tools_raw):
        if not isinstance(t, dict):
            raise ManifestError(f"skills/{slug}.tools[{i}]: must be a mapping")
        tn = t.get("name")
        if not isinstance(tn, str) or not _SLUG.match(tn.replace("_", "-")):
            raise ManifestError(f"skills/{slug}.tools[{i}]: invalid `name`")
        if tn in seen_tool:
            raise ManifestError(f"skills/{slug}: duplicate tool name `{tn}`")
        seen_tool.add(tn)

        # Skill-tool shape: `skill: <bare-name>`. Schemas + description are
        # resolved from the target skill at load time so the author doesn't
        # restate them. entrypoint / *_schema must be absent.
        skill_ref_raw = t.get("skill")
        if skill_ref_raw is not None:
            if not isinstance(skill_ref_raw, str) or not skill_ref_raw.strip():
                raise ManifestError(
                    f"skills/{slug}.tools[{tn}]: `skill` must be a non-empty string"
                )
            ref = skill_ref_raw.strip()
            if "/" in ref:
                raise ManifestError(
                    f"skills/{slug}.tools[{tn}]: `skill` must be a bare skill "
                    f"name resolvable in this deployment; cross-project "
                    f"`project/skill` is not yet supported in tool refs — use "
                    f"`puras.subagent.run` from a bash tool instead"
                )
            for forbidden in ("entrypoint", "input_schema", "output_schema"):
                if forbidden in t:
                    raise ManifestError(
                        f"skills/{slug}.tools[{tn}]: a `skill:` tool may not "
                        f"declare `{forbidden}` — it's copied from the target"
                    )
            tools.append(
                ToolDecl(
                    name=tn,
                    description=str(t.get("description") or "").strip(),
                    skill_ref=ref,
                    confirm=bool(t.get("confirm", False)),
                )
            )
            continue

        te = t.get("entrypoint")
        if not isinstance(te, str) or not _PY_ENTRYPOINT.match(te):
            raise ManifestError(
                f"skills/{slug}.tools[{tn}]: entrypoint must be `<file.py>:<func>`"
            )
        tis = t.get("input_schema")
        tos = t.get("output_schema")
        if not isinstance(tis, dict) or not isinstance(tos, dict):
            raise ManifestError(
                f"skills/{slug}.tools[{tn}]: input_schema + output_schema required"
            )
        tools.append(
            ToolDecl(
                name=tn,
                description=str(t.get("description") or "").strip(),
                entrypoint=te,
                input_schema=tis,
                output_schema=tos,
                confirm=bool(t.get("confirm", False)),
            )
        )

    examples_raw = data.get("examples") or []
    if not isinstance(examples_raw, list):
        raise ManifestError(f"skills/{slug}: `examples` must be a list")
    examples: list[SkillExample] = []
    for i, ex in enumerate(examples_raw):
        if not isinstance(ex, dict):
            raise ManifestError(f"skills/{slug}.examples[{i}]: must be a mapping")
        ex_inputs = ex.get("inputs")
        if not isinstance(ex_inputs, dict):
            raise ManifestError(
                f"skills/{slug}.examples[{i}]: `inputs` (object) required"
            )
        ex_title = ex.get("title")
        if ex_title is not None and not isinstance(ex_title, str):
            raise ManifestError(f"skills/{slug}.examples[{i}]: `title` must be string")
        ex_desc = ex.get("description")
        if ex_desc is not None and not isinstance(ex_desc, str):
            raise ManifestError(
                f"skills/{slug}.examples[{i}]: `description` must be string"
            )
        ex_outputs = ex.get("outputs")
        if ex_outputs is not None and not isinstance(ex_outputs, dict):
            raise ManifestError(
                f"skills/{slug}.examples[{i}]: `outputs` must be an object"
            )
        examples.append(
            SkillExample(
                inputs=ex_inputs,
                title=ex_title.strip() if isinstance(ex_title, str) else None,
                description=ex_desc.strip() if isinstance(ex_desc, str) else None,
                outputs=ex_outputs,
            )
        )

    evals = _parse_evals(slug, data, is_agentic)
    eval_dataset = _parse_eval_dataset(slug, data, is_agentic)
    eval_mocks = _parse_eval_mocks(slug, data, is_agentic)
    routing = _parse_routing(slug, data, is_agentic, model)
    marketing = _parse_marketing(slug, data)
    allowed_tools = _parse_allowed_tools(slug, data)
    tool_limits = _parse_tool_limits(slug, data)

    return SkillDecl(
        name=slug,
        path=path,
        description=description,
        entrypoint=entrypoint,
        input_schema=input_schema,
        output_schema=output_schema,
        title=title,
        model=model,
        disable_bash=disable_bash,
        tools=tools,
        examples=examples,
        evals=evals,
        eval_dataset=eval_dataset,
        eval_mocks=eval_mocks,
        routing=routing,
        marketing=marketing,
        allowed_tools=allowed_tools,
        tool_limits=tool_limits,
    )


def _parse_allowed_tools(slug: str, data: Any) -> list[str] | None:
    """`allowed_tools:` — a whitelist of tool names (strings). Omitted/empty = no
    restriction (None). Non-list, or non-string entries, are a manifest error."""
    raw = data.get("allowed_tools")
    if raw is None:
        return None
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise ManifestError(f"skills/{slug}.allowed_tools must be a list of tool names")
    names = [x.strip() for x in raw if x.strip()]
    return names or None


def _parse_tool_limits(slug: str, data: Any) -> dict[str, int]:
    """`tool_limits:` — `{tool_name: max_calls_per_run}`. Values must be positive
    ints. Omitted = {} (no caps)."""
    raw = data.get("tool_limits")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ManifestError(f"skills/{slug}.tool_limits must be a map of tool -> max calls")
    out: dict[str, int] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ManifestError(
                f"skills/{slug}.tool_limits['{k}'] must be a positive integer"
            )
        out[k.strip()] = v
    return out


# Escalation triggers a `routing:` block may list under `on:`. Only `schema_fail`
# (the cheap model can't produce schema-valid set_output) is implemented today;
# the list shape leaves room for `low_confidence` / `stuck` later.
_ROUTING_TRIGGERS = ("schema_fail",)
_DEFAULT_ESCALATE_AFTER = 2


def _parse_routing(
    slug: str, data: Any, is_agentic: bool, model: str | None
) -> dict[str, Any] | None:
    """Parse the optional `routing:` block — model tiering/escalation. The skill
    runs on its cheap `text_model` and escalates to `escalate_to` once a trigger
    fires (today: `schema_fail`, after N invalid set_output attempts).

    ```yaml
    text_model: claude/haiku-4-5     # the cheap default the run starts on
    routing:
      escalate_to: claude/sonnet-4-6 # premium model to switch to
      on: [schema_fail]              # optional — defaults to [schema_fail]
      after: 2                       # optional — escalate after N schema failures
    ```
    Agentic-only; `escalate_to` must be a known slug and differ from the default.
    """
    raw = data.get("routing")
    if raw is None:
        return None
    if not is_agentic:
        raise ManifestError(f"skills/{slug}: `routing` only applies to agentic (.md) skills")
    if not isinstance(raw, dict):
        raise ManifestError(f"skills/{slug}: `routing` must be a mapping")

    esc = raw.get("escalate_to")
    if not isinstance(esc, str) or not esc.strip():
        raise ManifestError(
            f"skills/{slug}: `routing.escalate_to` (a model slug) is required"
        )
    esc = esc.strip()
    if not PUBLIC_SLUG_RE.match(esc):
        raise ManifestError(
            f"skills/{slug}: `routing.escalate_to` must be a `family/variant` slug "
            f"(got {esc!r}); see docs/models"
        )
    if not is_known_slug(esc):
        raise ManifestError(
            f"skills/{slug}: `routing.escalate_to` unknown model `{esc}` — see docs/models"
        )
    if model and esc == model:
        raise ManifestError(
            f"skills/{slug}: `routing.escalate_to` ({esc}) is the same as the "
            f"skill's `text_model` — nothing to escalate to"
        )

    on_raw = raw.get("on")
    if on_raw is None:
        on = ["schema_fail"]
    else:
        if not isinstance(on_raw, list) or not all(isinstance(t, str) for t in on_raw):
            raise ManifestError(f"skills/{slug}: `routing.on` must be a list of strings")
        on = [t.strip() for t in on_raw if t.strip()]
        bad = [t for t in on if t not in _ROUTING_TRIGGERS]
        if bad:
            raise ManifestError(
                f"skills/{slug}: `routing.on` has unknown trigger(s) "
                f"{', '.join(repr(b) for b in bad)} — supported: "
                f"{', '.join(_ROUTING_TRIGGERS)}"
            )
        if not on:
            on = ["schema_fail"]

    after = raw.get("after", _DEFAULT_ESCALATE_AFTER)
    if isinstance(after, bool) or not isinstance(after, int) or after < 1:
        raise ManifestError(
            f"skills/{slug}: `routing.after` must be a positive integer "
            f"(escalate after N failures)"
        )

    return {"escalate_to": esc, "on": on, "after": after}


_EVAL_KINDS = ("check", "rubric", "exact_match", "schema")


def _parse_evals(slug: str, data: Any, is_agentic: bool) -> list[EvalDecl]:
    """Parse the optional `evals:` block of a skill.yaml. Accepts either a bare
    list of graders or a mapping with a `graders:` list (and an optional
    `dataset:` sibling — see `_parse_eval_dataset`). Each grader is a `check`
    (deterministic entrypoint), a `rubric` (LLM judge), an `exact_match` (output
    vs the case's `expected`), or a `schema` (output vs a JSON Schema). Evals are
    agentic-only."""
    evals_raw = data.get("evals")
    if evals_raw is None:
        return []
    if isinstance(evals_raw, dict):
        evals_raw = evals_raw.get("graders") or []
    if not isinstance(evals_raw, list):
        raise ManifestError(
            f"skills/{slug}: `evals` must be a list (or a mapping with `graders`)"
        )
    if evals_raw and not is_agentic:
        raise ManifestError(
            f"skills/{slug}: `evals` only apply to agentic (.md) skills"
        )

    evals: list[EvalDecl] = []
    seen: set[str] = set()
    for i, ev in enumerate(evals_raw):
        if not isinstance(ev, dict):
            raise ManifestError(f"skills/{slug}.evals[{i}]: must be a mapping")
        name = ev.get("name")
        if not isinstance(name, str) or not _SLUG.match(name.replace("_", "-")):
            raise ManifestError(f"skills/{slug}.evals[{i}]: invalid `name`")
        if name in seen:
            raise ManifestError(f"skills/{slug}: duplicate eval name `{name}`")
        seen.add(name)

        kind = ev.get("kind")
        if kind not in _EVAL_KINDS:
            raise ManifestError(
                f"skills/{slug}.evals[{name}]: `kind` must be one of "
                f"{', '.join(repr(k) for k in _EVAL_KINDS)}"
            )

        weight = ev.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
            raise ManifestError(
                f"skills/{slug}.evals[{name}]: `weight` must be a positive number"
            )

        if kind == "check":
            ep = ev.get("entrypoint")
            if not isinstance(ep, str) or not _PY_ENTRYPOINT.match(ep):
                raise ManifestError(
                    f"skills/{slug}.evals[{name}]: check `entrypoint` must be "
                    f"`<file.py>:<func>`"
                )
            evals.append(
                EvalDecl(name=name, kind="check", weight=float(weight), entrypoint=ep)
            )
        elif kind == "rubric":
            criteria = ev.get("criteria")
            if not isinstance(criteria, str) or not criteria.strip():
                raise ManifestError(
                    f"skills/{slug}.evals[{name}]: rubric `criteria` "
                    f"(non-empty string) required"
                )
            levels_raw = ev.get("levels") or {}
            if not isinstance(levels_raw, dict):
                raise ManifestError(
                    f"skills/{slug}.evals[{name}]: `levels` must be a mapping"
                )
            levels = {str(k): str(v) for k, v in levels_raw.items()}
            evals.append(
                EvalDecl(
                    name=name,
                    kind="rubric",
                    weight=float(weight),
                    criteria=criteria.strip(),
                    levels=levels,
                )
            )
        elif kind == "exact_match":
            fld = ev.get("field")
            if fld is not None and (not isinstance(fld, str) or not fld.strip()):
                raise ManifestError(
                    f"skills/{slug}.evals[{name}]: exact_match `field` must be a "
                    f"non-empty dotted path string (or omit it to compare the "
                    f"whole output)"
                )
            evals.append(
                EvalDecl(
                    name=name,
                    kind="exact_match",
                    weight=float(weight),
                    field=fld.strip() if isinstance(fld, str) else "",
                )
            )
        else:  # schema
            sch = ev.get("schema")
            if sch is not None and not isinstance(sch, dict):
                raise ManifestError(
                    f"skills/{slug}.evals[{name}]: schema `schema` must be a "
                    f"mapping (or omit it to validate against the skill's "
                    f"output_schema)"
                )
            evals.append(
                EvalDecl(
                    name=name,
                    kind="schema",
                    weight=float(weight),
                    schema=sch or {},
                )
            )
    return evals


def _parse_eval_dataset(slug: str, data: Any, is_agentic: bool) -> str | None:
    """Parse the optional `evals.dataset` path — a bundle-relative (to the skill
    dir) JSONL file of cases for the offline eval suite. Only the mapping form
    `evals: { dataset: ..., graders: [...] }` carries it; a bare list has none."""
    evals_raw = data.get("evals")
    if not isinstance(evals_raw, dict):
        return None
    ds = evals_raw.get("dataset")
    if ds is None:
        return None
    if not isinstance(ds, str) or not ds.strip():
        raise ManifestError(
            f"skills/{slug}: `evals.dataset` must be a non-empty path string"
        )
    if not is_agentic:
        raise ManifestError(
            f"skills/{slug}: `evals.dataset` only applies to agentic (.md) skills"
        )
    ds = ds.strip().lstrip("/")
    if ".." in ds.split("/"):
        raise ManifestError(
            f"skills/{slug}: `evals.dataset` must not contain '..' segments"
        )
    if not ds.endswith(".jsonl"):
        raise ManifestError(
            f"skills/{slug}: `evals.dataset` must be a `.jsonl` file (got {ds!r})"
        )
    return ds


def _parse_eval_mocks(slug: str, data: Any, is_agentic: bool) -> dict[str, Any]:
    """Parse the optional `evals.mocks` block — a mapping of tool-name → canned
    response used to short-circuit that tool in SUITE/test mode (so an eval run
    never triggers a real side effect). Only the mapping form
    `evals: { mocks: {...}, graders: [...] }` carries it; a bare graders list has
    none. Each value is the tool-result payload the agent sees in place of really
    running the tool (e.g. `generate_video: { drive_path: "drive/fixtures/x.mp4" }`).
    Built-in side-effecting verbs get a safe default stub even without an entry;
    an entry overrides that default and is the only way to mock a custom tool.
    Eval-time only — never consulted on a live run."""
    evals_raw = data.get("evals")
    if not isinstance(evals_raw, dict):
        return {}
    mocks = evals_raw.get("mocks")
    if mocks is None:
        return {}
    if not isinstance(mocks, dict):
        raise ManifestError(
            f"skills/{slug}: `evals.mocks` must be a mapping of tool-name → response"
        )
    if not is_agentic:
        raise ManifestError(
            f"skills/{slug}: `evals.mocks` only applies to agentic (.md) skills"
        )
    out: dict[str, Any] = {}
    for tool_name, spec in mocks.items():
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ManifestError(
                f"skills/{slug}: `evals.mocks` keys must be non-empty tool names"
            )
        out[tool_name.strip()] = spec
    return out


def _parse_marketing(slug: str, data: Any) -> dict[str, Any] | None:
    """Parse the optional `marketing:` block of a skill.yaml — SEO/landing copy
    for the skill's public page (headline, feature cards, personas, comparison,
    FAQ). Every field is optional; display-only, never read by the runtime. Kept
    in sync with api/app/manifest.py._parse_marketing. See that twin for the
    full shape.
    """
    raw = data.get("marketing")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ManifestError(f"skills/{slug}: `marketing` must be a mapping")

    out: dict[str, Any] = {}

    kw = raw.get("keywords")
    if kw is not None:
        if not isinstance(kw, list) or not all(isinstance(k, str) for k in kw):
            raise ManifestError(
                f"skills/{slug}: `marketing.keywords` must be a list of strings"
            )
        cleaned_kw = [k.strip() for k in kw if k.strip()]
        if cleaned_kw:
            out["keywords"] = cleaned_kw

    for key in ("features", "personas"):
        items = raw.get(key)
        if items is None:
            continue
        if not isinstance(items, list):
            raise ManifestError(f"skills/{slug}: `marketing.{key}` must be a list")
        cards: list[dict[str, str]] = []
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                raise ManifestError(
                    f"skills/{slug}: `marketing.{key}[{i}]` must be a mapping"
                )
            t = it.get("title")
            b = it.get("body")
            if not isinstance(t, str) or not t.strip():
                raise ManifestError(
                    f"skills/{slug}: `marketing.{key}[{i}].title` "
                    f"(non-empty string) required"
                )
            if b is not None and not isinstance(b, str):
                raise ManifestError(
                    f"skills/{slug}: `marketing.{key}[{i}].body` must be a string"
                )
            cards.append({"title": t.strip(), "body": (b or "").strip()})
        if cards:
            out[key] = cards

    comp = raw.get("comparison")
    if comp is not None:
        if not isinstance(comp, dict):
            raise ManifestError(
                f"skills/{slug}: `marketing.comparison` must be a mapping"
            )
        rows_raw = comp.get("rows")
        if not isinstance(rows_raw, list) or not rows_raw:
            raise ManifestError(
                f"skills/{slug}: `marketing.comparison.rows` must be a non-empty list"
            )
        rows: list[dict[str, str]] = []
        for i, r in enumerate(rows_raw):
            if not isinstance(r, dict):
                raise ManifestError(
                    f"skills/{slug}: `marketing.comparison.rows[{i}]` must be a mapping"
                )
            dim = r.get("dimension")
            if not isinstance(dim, str) or not dim.strip():
                raise ManifestError(
                    f"skills/{slug}: `marketing.comparison.rows[{i}].dimension` "
                    f"(non-empty string) required"
                )
            cmp_cell = r.get("competitor")
            pur_cell = r.get("puras")
            rows.append(
                {
                    "dimension": dim.strip(),
                    "competitor": cmp_cell.strip()
                    if isinstance(cmp_cell, str)
                    else "",
                    "puras": pur_cell.strip() if isinstance(pur_cell, str) else "",
                }
            )
        label = comp.get("competitor_label")
        out["comparison"] = {
            "competitor_label": label.strip()
            if isinstance(label, str) and label.strip()
            else "Other tools",
            "rows": rows,
        }

    faq_raw = raw.get("faq")
    if faq_raw is not None:
        if not isinstance(faq_raw, list):
            raise ManifestError(f"skills/{slug}: `marketing.faq` must be a list")
        faq: list[dict[str, str]] = []
        for i, item in enumerate(faq_raw):
            if not isinstance(item, dict):
                raise ManifestError(
                    f"skills/{slug}: `marketing.faq[{i}]` must be a mapping"
                )
            q = item.get("q")
            a = item.get("a")
            if (
                not isinstance(q, str)
                or not q.strip()
                or not isinstance(a, str)
                or not a.strip()
            ):
                raise ManifestError(
                    f"skills/{slug}: `marketing.faq[{i}]` needs non-empty `q` and `a`"
                )
            faq.append({"q": q.strip(), "a": a.strip()})
        if faq:
            out["faq"] = faq

    return out or None


def _verify_entrypoint_file_exists(decl: SkillDecl, skill_dir: Path) -> None:
    if decl.is_agentic:
        ep_file = skill_dir / decl.entrypoint
    else:
        m = _PY_ENTRYPOINT.match(decl.entrypoint)
        assert m  # already validated
        ep_file = skill_dir / m.group("file")
    if not ep_file.exists():
        raise ManifestError(
            f"skills/{decl.name}: entrypoint file `{decl.entrypoint}` not found"
        )


def from_stored_dict(d: dict[str, Any]) -> Manifest:
    """Rebuild a Manifest from the JSON shape we store in DeploymentRow.manifest.

    Used by the worker when it pulls the deployment row from the DB.
    """
    skills_raw = d.get("skills") or []
    if not isinstance(skills_raw, list):
        raise ManifestError("stored manifest: `skills` must be a list")
    skills: list[SkillDecl] = []
    for s in skills_raw:
        tools = [
            ToolDecl(
                name=t["name"],
                description=t.get("description", ""),
                entrypoint=t.get("entrypoint", "") or "",
                input_schema=t.get("input_schema", {}) or {},
                output_schema=t.get("output_schema", {}) or {},
                skill_ref=t.get("skill_ref"),
                confirm=bool(t.get("confirm", False)),
            )
            for t in (s.get("tools") or [])
        ]
        examples = [
            SkillExample(
                inputs=e["inputs"],
                title=e.get("title"),
                description=e.get("description"),
                outputs=e.get("outputs") if isinstance(e.get("outputs"), dict) else None,
            )
            for e in (s.get("examples") or [])
            if isinstance(e, dict) and isinstance(e.get("inputs"), dict)
        ]
        evals = [
            EvalDecl(
                name=e["name"],
                kind=e.get("kind", "check"),
                weight=float(e.get("weight", 1.0) or 1.0),
                entrypoint=e.get("entrypoint", "") or "",
                criteria=e.get("criteria", "") or "",
                levels=e.get("levels") or {},
                field=e.get("field", "") or "",
                schema=e.get("schema") or {},
            )
            for e in (s.get("evals") or [])
            if isinstance(e, dict) and e.get("name") and e.get("kind") in _EVAL_KINDS
        ]
        skills.append(
            SkillDecl(
                name=s["name"],
                path=s["path"],
                description=s.get("description", ""),
                entrypoint=s["entrypoint"],
                input_schema=s["input_schema"],
                output_schema=s["output_schema"],
                title=s.get("title"),
                model=s.get("model"),
                image_model=s.get("image_model"),
                video_model=s.get("video_model"),
                audio_model=s.get("audio_model"),
                media_kinds=s.get("media_kinds") or [],
                disable_bash=bool(s.get("disable_bash", False)),
                tools=tools,
                examples=examples,
                evals=evals,
                eval_dataset=s.get("eval_dataset") if isinstance(s.get("eval_dataset"), str) else None,
                eval_mocks=s.get("eval_mocks") if isinstance(s.get("eval_mocks"), dict) else {},
                routing=s.get("routing") if isinstance(s.get("routing"), dict) else None,
                allowed_tools=s.get("allowed_tools") if isinstance(s.get("allowed_tools"), list) else None,
                tool_limits=s.get("tool_limits") if isinstance(s.get("tool_limits"), dict) else {},
                parent_skill=s.get("parent_skill"),
                marketing=s.get("marketing") if isinstance(s.get("marketing"), dict) else None,
            )
        )
    return Manifest(skills=skills)
