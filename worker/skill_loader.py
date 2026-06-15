"""Resolve a runnable skill from a deployment.

After the worker pulls the deployment row from the DB, it has the manifest
JSON shape. `load(manifest, deployment_root, name)` returns a `LoadedSkill`
with everything the dispatchers (agent runner, function runner) need:

  - For agentic skills (.md entrypoint): the system_prompt text + tools list
    + output_schema dict.
  - For deterministic skills (.py entrypoint): the resolved module path:func
    string (ready for function_runner) + output_schema dict.

Tool entrypoints are *relative to the skill dir*. We convert them to
Python dotted module paths anchored at the deployment root so the
subprocess runner can import them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .manifest import (
    EvalDecl,
    Manifest,
    SkillDecl,
    ToolDecl,
    _PY_ENTRYPOINT,  # re-use the validator
)  # noqa: F401


@dataclass
class LoadedTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    # Local Python tool fields: dotted module + callable. Empty for skill tools.
    module: str = ""
    func: str = ""
    # Skill-ref tool: qualified name (e.g. `ugc-video/creative-research` or a
    # top-level `analyze`) resolved against the same deployment's manifest at
    # load time. Dispatched via `/v1/subagent/invoke` at runtime.
    skill_ref: str | None = None
    # True when the tool needs explicit human approval before each call
    # (`confirm: true` in skill.yaml). Enforced by the agent_runner dispatcher.
    confirm: bool = False

    @property
    def is_skill_tool(self) -> bool:
        return self.skill_ref is not None


@dataclass
class LoadedEval:
    """An eval grader resolved for a run. For `check` graders the relative
    entrypoint is pre-resolved to a dotted module + func (like a tool) so the
    eval_runner can hand it straight to the subprocess function runner; for
    `rubric` graders it carries the judge `criteria` + anchored `levels`;
    `exact_match` carries the optional `field` path (compared to the case's
    expected); `schema` carries the optional explicit `schema` (else the skill's
    output_schema is used)."""

    name: str
    kind: str                                 # "check" | "rubric" | "exact_match" | "schema"
    weight: float = 1.0
    module: str = ""                          # check: dotted module from deployment root
    func: str = ""                            # check: callable name
    criteria: str = ""                        # rubric: what the judge scores
    levels: dict[str, str] = field(default_factory=dict)  # rubric: anchored score → meaning
    # `schema` before `field`: a field named `field` shadows `dataclasses.field`
    # for the rest of the class body, so `= field(...)` below it would crash.
    schema: dict[str, Any] = field(default_factory=dict)  # schema: explicit shape (else output_schema)
    field: str = ""                           # exact_match: dotted path into output/expected


@dataclass
class LoadedSkill:
    name: str
    root: Path                                # deployment_root / skill.path
    is_agentic: bool
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    # True for an ad-hoc subagent: a bundle `*.md` run via run_subagent /
    # subagent.run that has no manifest entry, no declared schemas, and a
    # free-form `set_output` (see agent_runner._build_tools).
    is_adhoc: bool = False
    # Agentic-only fields
    system_prompt: str | None = None
    model: str | None = None
    disable_bash: bool = False
    tools: list[LoadedTool] = field(default_factory=list)
    # Output graders (agentic only) run after a successful run to score it.
    evals: list[LoadedEval] = field(default_factory=list)
    # Bundle-relative (to the skill dir) path to the offline eval dataset, or None.
    eval_dataset: str | None = None
    # Model routing/escalation policy `{escalate_to, on, after}`, or None. The run
    # starts on `model` and switches to `escalate_to` when a trigger fires.
    routing: dict[str, Any] | None = None
    # Least-privilege tool scope: a whitelist of tool names (None = no limit) and
    # per-run per-tool call caps `{tool: max}` ({} = none). Enforced by
    # agent_runner (_build_tools filters the offered set; dispatch refuses an
    # out-of-scope or over-limit call). See manifest.SkillDecl.
    allowed_tools: list[str] | None = None
    tool_limits: dict[str, int] = field(default_factory=dict)
    # Deterministic-only fields
    py_module: str | None = None              # dotted module from deployment root
    py_func: str | None = None


def _file_to_module(skill_path: str, rel_file: str) -> str:
    """skill_path='foo', rel_file='tools/bar.py' → 'foo.tools.bar'."""
    p = (Path(skill_path) / rel_file).with_suffix("")
    return ".".join(p.parts)


def load(manifest: Manifest, deployment_root: Path, name: str) -> LoadedSkill:
    decl: SkillDecl | None = manifest.skill(name)
    if decl is None:
        raise ValueError(f"skill `{name}` not declared in manifest")
    root = deployment_root / decl.path
    if not root.is_dir():
        raise ValueError(f"skill `{name}`: dir not found at {root}")

    if decl.is_agentic:
        md_file = root / decl.entrypoint
        if not md_file.exists():
            raise ValueError(f"skill `{name}`: entrypoint file `{decl.entrypoint}` missing")
        system_prompt = md_file.read_text("utf-8")
        tools = [_load_tool(t, decl, manifest) for t in decl.tools]
        evals = [_load_eval(e, decl) for e in decl.evals]
        return LoadedSkill(
            name=name,
            root=root,
            is_agentic=True,
            input_schema=decl.input_schema,
            output_schema=decl.output_schema,
            system_prompt=system_prompt,
            model=decl.model,
            disable_bash=decl.disable_bash,
            tools=tools,
            evals=evals,
            eval_dataset=decl.eval_dataset,
            routing=decl.routing,
            allowed_tools=decl.allowed_tools,
            tool_limits=decl.tool_limits,
        )

    # Deterministic .py skill
    m = _PY_ENTRYPOINT.match(decl.entrypoint)
    assert m, f"entrypoint already validated upstream: {decl.entrypoint!r}"
    py_file = root / m.group("file")
    if not py_file.exists():
        raise ValueError(f"skill `{name}`: entrypoint file `{m.group('file')}` missing")
    return LoadedSkill(
        name=name,
        root=root,
        is_agentic=False,
        input_schema=decl.input_schema,
        output_schema=decl.output_schema,
        py_module=_file_to_module(decl.path, m.group("file")),
        py_func=m.group("func"),
    )


def apply_prompt_override(loaded: LoadedSkill, override: dict[str, Any] | None) -> LoadedSkill:
    """Return a copy of `loaded` with the optimizer's candidate prompt/model/routing
    swapped in (the candidate-injection seam).

    Override shape: `{system_prompt?, model?, routing?}`. A key that is PRESENT is
    applied (None clears it — e.g. `routing: None` disables escalation); a key that
    is ABSENT keeps the skill's deployed value. Only agentic skills are overridable;
    a non-agentic or empty/None override returns the skill unchanged.
    """
    if not override or not loaded.is_agentic:
        return loaded
    changes: dict[str, Any] = {}
    if "system_prompt" in override:
        sp = override["system_prompt"]
        if sp is not None:  # never blank out the prompt — ignore a null body
            changes["system_prompt"] = str(sp)
    if "model" in override:
        changes["model"] = override["model"]
    if "routing" in override:
        changes["routing"] = override["routing"]
    return replace(loaded, **changes) if changes else loaded


def load_adhoc(deployment_root: Path, prompt_rel_path: str) -> LoadedSkill:
    """Load a bundle `*.md` file as an ad-hoc subagent.

    Used when a job's `skill_name` is a markdown path (not a manifest skill) —
    i.e. it was spawned by `run_subagent` / `subagent.run("references/x.md", ...)`.
    The file's text is the system prompt; the agent gets the same builtin tools
    plus a free-form `set_output` (no declared schema either way).

    `prompt_rel_path` is relative to the deployment root, e.g.
    `product-reveal-video/references/storyboard-director.md`. Its first segment
    is the owning skill dir, which becomes `root` so the subagent's
    `references/`, `scripts/`, and venv (`requirements.txt`) all resolve like
    the parent skill's.
    """
    rel = prompt_rel_path.strip().lstrip("/")
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts or not rel.endswith(".md"):
        raise ValueError(f"invalid ad-hoc prompt path `{prompt_rel_path}`")
    md_file = (deployment_root / p).resolve()
    try:
        md_file.relative_to(deployment_root.resolve())
    except ValueError as e:
        raise ValueError(
            f"ad-hoc prompt path `{prompt_rel_path}` escapes the bundle"
        ) from e
    if not md_file.is_file():
        raise ValueError(f"ad-hoc prompt file not found: {prompt_rel_path}")
    skill_dir = p.parts[0]
    root = deployment_root / skill_dir
    return LoadedSkill(
        name=rel,
        root=root,
        is_agentic=True,
        is_adhoc=True,
        input_schema={},
        output_schema=None,  # free-form; set_output records any object
        system_prompt=md_file.read_text("utf-8"),
        model=None,          # worker default model
        disable_bash=False,
        tools=[],
    )


def load_inline(deployment_root: Path, prompt: str) -> LoadedSkill:
    """Load an inline prompt string as a subagent.

    Used when a job's `skill_name` is the `@inline` sentinel and it carries an
    `inline_prompt` (spawned by `run_subagent` / `subagent.run(prompt=...)`).
    The prompt text is the system prompt; the subagent gets the same builtin
    tools plus a free-form `set_output` (no declared schema either way), exactly
    like an ad-hoc `*.md` subagent.

    Runs in the parent skillpack's bundle context: `root` is the deployment
    root, so bash sees the whole bundle and any `requirements.txt` at the root
    builds the venv. There is no specific skill dir, so reference bundle files
    by their full path from the bundle root.
    """
    text = (prompt or "").strip()
    if not text:
        raise ValueError("inline subagent prompt is empty")
    return LoadedSkill(
        name="@inline",
        root=deployment_root,
        is_agentic=True,
        is_adhoc=True,
        input_schema={},
        output_schema=None,  # free-form; set_output records any object
        system_prompt=text,
        model=None,          # worker default model
        disable_bash=False,
        tools=[],
    )


def _load_eval(decl: EvalDecl, skill: SkillDecl) -> LoadedEval:
    """Resolve one EvalDecl into a runnable LoadedEval. A `check` grader's
    relative `<file.py>:<func>` entrypoint is anchored to the skill dir and
    converted to a dotted module — same scheme as a Python tool — so the
    subprocess runner can import it. A `rubric` grader carries its judge text;
    `exact_match` its compare `field`; `schema` its explicit `schema`."""
    if decl.kind == "check":
        m = _PY_ENTRYPOINT.match(decl.entrypoint)
        # The manifest parser already validated the shape; guard anyway so a
        # malformed stored manifest degrades to a skipped grader, not a crash.
        module = _file_to_module(skill.path, m.group("file")) if m else ""
        func = m.group("func") if m else ""
        return LoadedEval(
            name=decl.name, kind="check", weight=decl.weight, module=module, func=func
        )
    if decl.kind == "rubric":
        return LoadedEval(
            name=decl.name,
            kind="rubric",
            weight=decl.weight,
            criteria=decl.criteria,
            levels=dict(decl.levels or {}),
        )
    if decl.kind == "exact_match":
        return LoadedEval(
            name=decl.name,
            kind="exact_match",
            weight=decl.weight,
            field=decl.field or "",
        )
    # schema
    return LoadedEval(
        name=decl.name,
        kind="schema",
        weight=decl.weight,
        schema=dict(decl.schema or {}),
    )


def _load_tool(decl: ToolDecl, skill: SkillDecl, manifest: Manifest) -> LoadedTool:
    if decl.skill_ref is not None:
        # Resolve target skill in the same deployment. Try
        # `<parent_top>/<ref>` first (subskill of the caller), then bare
        # top-level. This matches the runtime resolver in /v1/subagent/invoke,
        # so the LLM sees the same tool input_schema either way.
        caller_top = (skill.parent_skill or skill.name).split("/", 1)[0]
        candidates: list[str] = []
        if caller_top:
            candidates.append(f"{caller_top}/{decl.skill_ref}")
        candidates.append(decl.skill_ref)

        target: SkillDecl | None = None
        resolved_name: str | None = None
        for cand in candidates:
            t = manifest.skill(cand)
            if t is None:
                continue
            # Subskills are reachable only via their parent.
            if t.is_subskill and t.parent_skill != caller_top:
                continue
            target = t
            resolved_name = cand
            break
        if target is None or resolved_name is None:
            raise ValueError(
                f"skill `{skill.name}`: tool `{decl.name}` references skill "
                f"`{decl.skill_ref}` which is not in this deployment "
                f"(tried: {', '.join(candidates)})"
            )
        return LoadedTool(
            name=decl.name,
            description=decl.description or target.description,
            input_schema=target.input_schema,
            output_schema=target.output_schema,
            skill_ref=resolved_name,
            confirm=decl.confirm,
        )

    m = _PY_ENTRYPOINT.match(decl.entrypoint)
    assert m, f"tool entrypoint already validated upstream: {decl.entrypoint!r}"
    return LoadedTool(
        name=decl.name,
        description=decl.description,
        module=_file_to_module(skill.path, m.group("file")),
        func=m.group("func"),
        input_schema=decl.input_schema,
        output_schema=decl.output_schema,
        confirm=decl.confirm,
    )
