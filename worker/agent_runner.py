"""Agentic loop driving an LLM via tool_use.

Tools available to the agent:
- `bash` (auto-added unless skill.disable_bash=true): runs shell in workdir
- `set_output` (auto-added if skill has output_schema): records the job's
  structured result; calling it cleanly ends the run
- platform built-ins: todo_write, media, web_search, image_search, web_fetch,
  web_screenshot, download_url, file_read
- user tools declared in skill.yaml under `tools:` (each with input/output schemas);
  executed via the deployment's venv python, with output_schema validated before
  feeding the result back to the agent
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import re
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import structlog

# Offline-runner import hygiene (open-core PR 3b): the platform-only modules
# below — `db`, `queue`, `approvals`, `checkpoint`, `memory_store` — pull the
# DB stack (sqlalchemy / asyncpg / pgvector) at import time, and `db`/`queue`
# can't even be made light (they build engines / SQL at module scope). None of
# them are reached on a LOCAL run (every call site is gated on
# `ctx.platform_enabled`), so they are imported LAZILY at their hosted call
# sites instead of here. That lets `pip install puras[local]` ship a dependency
# -light runner: `import agent_runner` (and a full local run) never touches the
# DB stack. `prompt_cache` / `storage` / `providers` ARE on the local path, so
# they were made light internally (lazy heavy imports) and stay imported here.
if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

from .attachments import load_attachment, model_supports_vision, sanitize_url
from .config import get_settings, service_token
from .deployment import ResolvedDeployment, build_skill_python
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .agent_tool_specs import BUILTIN_AGENT_TOOLS, set_output_tool_spec
from .function_runner import run_function
from .llm_models import is_known_slug, resolve as resolve_model
from .run_context import DbRunContext, RunContext
from .pricing import with_margin
from .proc_limits import child_preexec
from .prompt_cache import cached_messages_create
from .providers import make_provider
from .memory import derive_identity, format_memory_digest
from .event_ctx import event_ctx
from .schema_dialect import prune_extras, to_jsonschema, to_output_jsonschema
from .skill_loader import (
    LoadedSkill,
    LoadedTool,
    load as load_skill,
    load_adhoc,
    load_inline,
)
from .drive import resolve_output_dir, workspace_drive
from .storage import (
    ensure_local_drive_file,
    push_input_files,
    signed_url,
    upload_drive_file,
)
from .workdir import attach_skill, cleanup_workdir, create_workdir

# ── Tracing (P0-3) ───────────────────────────────────────────────────────────
# OTel-style spans for a run: the whole run (root), each agent step, each model
# call, each tool call, and each nested subagent run, timed and stitched by
# parent_span_id. The current enclosing span id is held in a ContextVar so spans
# nest automatically across `await`s AND across the concurrent-tool gather (each
# task inherits a copy of the context, so a tool span created in a child task
# still points at the step span that spawned it). Recording is best-effort and
# routed through RunContext, so it works hosted (Postgres) and local (in-memory).
_current_span: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "puras_current_span", default=None
)


class _Span:
    """A live span. `attrs` is mutable so the body can attach results learned
    during the work (token counts, stop_reason, ok) before it's recorded."""

    __slots__ = ("id", "attrs")

    def __init__(self, span_id: str, attrs: dict):
        self.id = span_id
        self.attrs = attrs


@asynccontextmanager
async def _span(ctx: RunContext, kind: str, name: str, attributes: dict | None = None):
    """Open a trace span around a block: time it, nest it under the current span,
    and record it on exit. Never raises out of the recording itself."""
    span_id = uuid4().hex[:16]
    parent = _current_span.get()
    sp = _Span(span_id, dict(attributes or {}))
    token = _current_span.set(span_id)
    start = time.monotonic()
    ok = True
    try:
        yield sp
    except BaseException:
        ok = False
        raise
    finally:
        _current_span.reset(token)
        sp.attrs.setdefault("ok", ok)
        try:
            await ctx.record_span(
                span_id=span_id,
                parent_span_id=parent,
                kind=kind,
                name=name,
                duration_ms=int((time.monotonic() - start) * 1000),
                attributes=sp.attrs,
            )
        except Exception:
            log.warning("span_record_failed", kind=kind, name=name, exc_info=True)


# Max subagent call-graph depth (in-process recursion). Mirrors the API's
# MAX_INVOKE_DEPTH for the cross-skillpack fallback path. depth=1 is a top-level
# job; a subagent it spawns is depth=2, etc. We refuse to spawn at >= this.
MAX_SUBAGENT_DEPTH = 5

# Media verbs whose remote generation can run concurrently when the model asks
# for several in one turn (a fan-out wave). They go through _call_media → the API
# (billing + fal) and never touch the worker's DB session, so a batch is safe to
# gather. `_PARALLEL_MEDIA_LIMIT` caps how many fire at once to stay within the
# upstream provider's rate limits and the worker's thread pool.
_MEDIA_VERBS = {"generate_image", "generate_video", "generate_audio", "transcribe"}
_PARALLEL_MEDIA_LIMIT = max(1, int(os.getenv("PARALLEL_MEDIA_LIMIT", "4")))

# Built-in tools that need the platform — Postgres (workspace memory), the
# bucket (drive_url/drive_pull), or the platform API (media generation + web).
# The open-core line: on a local run (`puras run --local`, ctx.platform_enabled
# False) these are NOT offered to the model, so it never reaches for a capability
# that can't exist offline. What stays is the free local surface: text, bash,
# the file tools, deterministic skill tools, and in-process subagents. Mirrors
# the LocalRunContext docstring and the run_context open-core switch.
PLATFORM_ONLY_TOOLS = frozenset(
    {
        # media generation (→ /v1/media, billing + fal)
        "generate_image", "generate_video", "generate_audio", "transcribe",
        # web (→ /v1/web search/fetch, and the headless-browser screenshot)
        "web_search", "image_search", "web_fetch", "web_screenshot", "download_url",
        # workspace shared memory (Postgres)
        "memory_search", "memory_get", "memory_put", "memory_forget",
        # drive↔bucket helpers (no bucket locally)
        "drive_url", "drive_pull",
    }
)

# File extensions that mark a drive_path as renderable media, so the job's
# pipeline view can show it as a card. Mirrors frontend/lib/media.ts.
_MEDIA_EXTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("image", (".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif", ".bmp", ".svg")),
    ("video", (".mp4", ".webm", ".mov", ".m4v", ".mkv")),
    ("audio", (".mp3", ".wav", ".m4a", ".ogg", ".opus", ".flac", ".aac")),
)


def _media_kind_for_path(path: str) -> str | None:
    p = path.split("?", 1)[0].split("#", 1)[0].lower()
    for kind, exts in _MEDIA_EXTS:
        if p.endswith(exts):
            return kind
    return None


def _collect_tool_media(value: Any) -> list[dict]:
    """Walk a custom tool's result for `drive_path` strings that point at a media
    file, returning deduped `[{drive_path, kind}]`. Unlike the generate_* verbs,
    a skill-declared tool (stitch, frame_grab, the auto-caption burn, …) returns
    its file as a plain dict field, so its media never reaches the tool_result
    event. Lifting it here lets the pipeline render those outputs as cards too."""
    out: list[dict] = []
    seen: set[str] = set()

    def walk(v: Any) -> None:
        if isinstance(v, dict):
            dp = v.get("drive_path")
            if isinstance(dp, str) and dp and dp not in seen:
                kind = _media_kind_for_path(dp)
                if kind:
                    seen.add(dp)
                    out.append({"drive_path": dp, "kind": kind})
            for child in v.values():
                walk(child)
        elif isinstance(v, list):
            for child in v:
                walk(child)

    walk(value)
    return out

# Same-turn tool fan-out. When a model turn emits >1 tool call and NONE of them
# are parallel-unsafe, the calls run concurrently — each in its own asyncio task
# with its OWN db session (the worker session isn't concurrency-safe) and its own
# event ctx. So 2 web_searches, 2 audio converts, 2 subagents, etc. all overlap.
# Only set_output is excluded: it ends the run (a control-flow terminator, nothing
# to parallelize), so a turn containing it falls back to serial. bash IS included
# — concurrent subprocesses lean on the kernel-OOM-victim guard (a runaway child
# is killed, not the worker) and a failed one surfaces as a clean tool error; just
# don't have two bash calls write the SAME drive file in one turn. The limit caps
# concurrent tasks → bounds DB connections (see db.get_engine pool floor) and
# upstream/memory load. Subagents count here, so keep it modest.
# Tools that force the whole turn to run serially. `set_output` ends the run and
# uses the non-concurrency-safe main session. `file_write`/`file_edit` mutate
# drive files: two edits to the SAME file in one turn would race (read-modify-
# write on stale bytes), so serialize any turn that contains a file mutation —
# the writes are fast, so the lost parallelism is negligible next to the safety.
_PARALLEL_UNSAFE = {"set_output", "file_write", "file_edit"}
_PARALLEL_TOOL_LIMIT = max(1, int(os.getenv("PARALLEL_TOOL_LIMIT", "3")))
# Only fan out at shallow depths (root run_agent is depth 1) so the per-task
# sessions a turn opens can't exceed the DB pool floor — which is sized for ONE
# level of fan-out (db.get_engine: conc*(1+PARALLEL_TOOL_LIMIT)+2). Nested
# subagents therefore run their own tools serially. Raising this also needs the
# pool floor raised, or deep+wide nesting hits (gracefully-handled) pool waits.
_PARALLEL_TOOL_MAX_DEPTH = max(1, int(os.getenv("PARALLEL_TOOL_MAX_DEPTH", "1")))

log = structlog.get_logger()

# Staged-input URLs end up in jobs.inputs (and in the agent's first user
# message). The History tab reads jobs.inputs back days/weeks later to show
# input thumbnails — a 1h TTL meant most cards turned into broken images.
# 30d matches the persisted-output TTL on the API side.
_STAGED_URL_TTL_SECONDS = 30 * 24 * 3600


def _resolve_drive_url(path: Any, ttl: Any, workspace_id: str) -> dict:
    """Mint a signed URL for a drive-relative path. Backs the `drive_url` tool."""
    if not isinstance(path, str) or not path.strip():
        return {
            "ok": False,
            "error": "drive_url requires 'path' (a drive-relative file path)",
        }
    clean = path.strip().lstrip("/")
    if clean.startswith("drive/"):
        clean = clean[len("drive/") :]
    if ".." in clean.split("/"):
        return {"ok": False, "error": "'..' segments not allowed in drive paths"}
    try:
        ttl_i = int(ttl) if ttl is not None else 3600
    except (TypeError, ValueError):
        ttl_i = 3600
    ttl_i = max(60, min(86400, ttl_i))
    s = get_settings()
    try:
        url = signed_url(s.drive_bucket, f"{workspace_id}/{clean}", ttl_i)
    except Exception as e:
        return {"ok": False, "error": f"could not sign '{clean}': {e}"}
    return {"ok": True, "url": url, "path": clean, "expires_in": ttl_i}


def _proc_output_text(stdout: Any, stderr: Any) -> str:
    """Join a subprocess's stdout+stderr into capped text, bytes-safe.

    On a POSIX timeout, ``subprocess.TimeoutExpired`` carries the partial
    stdout/stderr as RAW BYTES even when the call used ``text=True`` — the
    decode step only runs after a clean ``communicate()``, which never happens
    when it times out. The old assembly ``(stdout or "") + (stderr or "")``
    then mixed bytes with a str fallback and raised
    ``TypeError: can't concat str to bytes``, which propagated out of
    ``asyncio.to_thread`` and killed the whole agent job (a single slow
    ``find`` over the s3fs drive mount was enough). Coerce every piece to str
    before joining.
    """

    def _astext(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, bytes):
            return x.decode("utf-8", "replace")
        return x

    return (_astext(stdout) + _astext(stderr))[-8192:]


def _run_bash(command: str, timeout: int, cwd: Path, env_extra: dict[str, str]) -> dict:
    import os

    s = get_settings()
    t = max(1, min(s.bash_max_timeout, timeout or s.bash_default_timeout))
    env = {**os.environ, **env_extra}
    try:
        # Mark the shell (and any ffmpeg it forks) the kernel's OOM victim so a
        # runaway encode is killed before the worker process — see proc_limits.
        proc = subprocess.run(
            ["bash", "-c", command],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=t,
            env=env,
            preexec_fn=child_preexec(),
        )
        return {"exit": proc.returncode, "output": _proc_output_text(proc.stdout, proc.stderr)}
    except subprocess.TimeoutExpired as e:
        out = _proc_output_text(e.stdout, e.stderr)
        return {"exit": -1, "output": f"[timed out after {t}s]\n{out}"}


def _run_file_read(paths: Any, workspace_id: str, model_slug: str) -> list[dict] | str:
    """Load drive files into a list of content blocks for a tool_result.

    Returns:
        list[dict] — block list on success (mixed text/image/document)
        str        — error string if input was malformed (returned as-is to LLM)
    """
    if not isinstance(paths, list) or not paths:
        return "ERROR: file_read requires 'paths' (non-empty list of drive paths)"
    if len(paths) > 10:
        return "ERROR: file_read accepts at most 10 paths per call"

    vision_ok = model_supports_vision(model_slug)
    blocks: list[dict] = []
    errors: list[str] = []
    for p in paths:
        if not isinstance(p, str) or not p.strip():
            errors.append(f"{p!r}: path must be a non-empty string")
            continue
        try:
            loaded = load_attachment({"drive_path": p}, workspace_id)
        except (ValueError, FileNotFoundError) as e:
            errors.append(f"{p}: {e}")
            continue
        block = loaded["block"]
        if block is not None and not vision_ok:
            errors.append(
                f"{p}: file is {loaded['mime']} but model `{model_slug}` "
                f"can't process image/document blocks"
            )
            continue
        blocks.append({"type": "text", "text": f"=== {loaded['label']} ==="})
        if block is not None:
            blocks.append(block)
        elif loaded["text"] is not None:
            blocks.append({"type": "text", "text": loaded["text"]})
    if errors:
        blocks.append({"type": "text", "text": "Errors: " + "; ".join(errors)})
    if not blocks:
        return "ERROR: nothing readable in paths"
    return blocks


def _run_file_write(path: Any, content: Any, workspace_id: str) -> dict:
    """Back the `file_write` tool: create/overwrite a text file in the drive.

    Resolves a drive-relative path under the workspace drive root (rejecting
    `..` traversal), writes UTF-8 text, re-stats to confirm the bytes landed
    (a short write means a drive/storage fault, not a content problem), and
    pushes to the bucket so the API and later steps can read it. Mirrors the
    persistence model of `build.py` and the media/screenshot write paths."""
    clean, err = _clean_drive_path(path)
    if err:
        return {"ok": False, "error": err}
    if not isinstance(content, str):
        return {"ok": False, "error": "`content` must be a string"}
    dest = workspace_drive(workspace_id) / clean
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        dest.write_bytes(data)
    except OSError as e:
        return {"ok": False, "error": f"write failed: {e}"}
    persisted = dest.stat().st_size
    if persisted != len(data):
        return {
            "ok": False,
            "error": (
                f"wrote {len(data)} bytes to {clean} but the drive persisted "
                f"{persisted} — the write did not land (drive/storage fault, "
                f"not a content problem); do not retry blindly"
            ),
        }
    upload_drive_file(workspace_id, clean)
    return {
        "ok": True,
        "drive_path": clean,
        "size_bytes": persisted,
        # Exact character count (≠ bytes for non-ASCII) — lets the agent check
        # a budget-capped media prompt against its limit without re-emitting it.
        "chars": len(content),
        "lines": content.count("\n") + (0 if content.endswith("\n") or not content else 1),
        "created": True,
    }


def _offload_tool_result(
    tu_name: str, tu_id: str, content: Any, job_id: Any, workspace_id: Any
) -> Any:
    """Keep large tool results out of the running context (token economy / P1).

    A multi-step agent re-reads its whole conversation each turn, so a big tool
    result (a fetched page, long bash stdout, a verbose subagent return) is paid
    for on every later step it stays inline. When the result's text exceeds the
    configured threshold we OFFLOAD it: persist the full payload to a drive file
    and return a head + a `file_read` pointer. Restorable (the full text is one
    read away, à la Manus' "drop the page, keep the URL") and cache-safe — the
    stub is what enters the append-only history, so the prompt cache is never
    invalidated (unlike retroactive clearing).

    Returns `content` unchanged when it's small, non-text (image/multimodal
    blocks), offloading is disabled, or the drive write fails — offloading must
    never grow a result or lose data it can't restore."""
    s = get_settings()
    limit = s.tool_result_offload_chars
    if limit <= 0 or not isinstance(content, str) or len(content) <= limit:
        return content
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(tu_id))[:80] or "tool"
    rel = f"_jobs/{job_id}/_toolout/{safe_id}.txt"
    out = _run_file_write(rel, content, str(workspace_id))
    if not out.get("ok"):
        return content  # write failed → leave the result intact (never lose data)
    head_chars = max(0, s.tool_result_offload_head_chars)
    head = content[:head_chars]
    return (
        f"{head}\n\n"
        f"…[{tu_name} result truncated for context economy: {len(content):,} chars "
        f"total, first {len(head):,} shown. Full result saved to drive `{rel}` — "
        f"`file_read` it (or `drive_pull` then read it from bash) for the rest.]"
    )


def _match_line_numbers(text: str, sub: str, limit: int = 6) -> list[int]:
    """1-based line numbers where `sub` begins, up to `limit` (for disambiguating
    a non-unique edit anchor)."""
    out: list[int] = []
    pos = text.find(sub)
    while pos != -1 and len(out) < limit:
        out.append(text.count("\n", 0, pos) + 1)
        pos = text.find(sub, pos + 1)
    return out


def _edit_diff(before: str, after: str, path: str, max_lines: int = 80) -> str:
    """A unified diff (with @@ line-number hunks) of an edit, capped so a huge
    rewrite doesn't flood the tool result."""
    import difflib

    lines = list(
        difflib.unified_diff(
            before.splitlines(), after.splitlines(),
            fromfile=path, tofile=path, lineterm="", n=2,
        )
    )
    if len(lines) > max_lines:
        extra = len(lines) - max_lines
        lines = lines[:max_lines] + [f"… (+{extra} more diff lines)"]
    return "\n".join(lines)


def _run_file_edit(args: dict, workspace_id: str) -> dict:
    """Back the `file_edit` tool: an atomic, surgical edit of a drive text file.

    STRING mode (`old_string`/`new_string`, optional `replace_all`) is the
    robust default — an exact, unique match or it's refused, so the agent never
    silently patches the wrong place. LINE mode (`start_line`/`end_line`) covers
    the case where a clean anchor is awkward. Returns a line-numbered unified
    diff of the change. Reads through the bucket on a local miss (like file_read)
    and pushes the result back."""
    if not isinstance(args, dict):
        return {"ok": False, "error": "bad arguments"}
    clean, err = _clean_drive_path(args.get("path"))
    if err:
        return {"ok": False, "error": err}
    local = workspace_drive(workspace_id) / clean
    if not local.is_file() and not ensure_local_drive_file(workspace_id, clean):
        return {
            "ok": False,
            "error": f"file not found: {clean} — use `file_write` to create it first",
        }
    if not local.is_file():
        return {"ok": False, "error": f"file not found: {clean}"}

    new_string = args.get("new_string")
    if not isinstance(new_string, str):
        return {"ok": False, "error": "`new_string` is required (a string)"}

    original = local.read_text("utf-8", errors="replace")
    old_string = args.get("old_string")
    start_line = args.get("start_line")
    end_line = args.get("end_line")
    replace_all = bool(args.get("replace_all"))

    if isinstance(old_string, str) and old_string != "":
        count = original.count(old_string)
        if count == 0:
            return {
                "ok": False,
                "error": (
                    "`old_string` not found. It must match the file EXACTLY, "
                    "including whitespace/indentation. Read the file (file_read) "
                    "and copy an exact, unique snippet."
                ),
            }
        if count > 1 and not replace_all:
            lns = _match_line_numbers(original, old_string)
            return {
                "ok": False,
                "error": (
                    f"`old_string` is not unique — {count} matches (lines "
                    f"{', '.join(map(str, lns))}…). Add surrounding context to "
                    f"make the anchor unique, or pass replace_all=true."
                ),
            }
        updated = (
            original.replace(old_string, new_string)
            if replace_all
            else original.replace(old_string, new_string, 1)
        )
        replacements = count if replace_all else 1
    elif start_line is not None or end_line is not None:
        try:
            s = int(start_line)
            e = int(end_line)
        except (TypeError, ValueError):
            return {
                "ok": False,
                "error": "LINE mode needs integer `start_line` and `end_line` (1-based, inclusive)",
            }
        kept = original.splitlines(keepends=True)
        if s < 1 or e < s or e > len(kept):
            return {
                "ok": False,
                "error": f"line range {s}..{e} out of bounds (file has {len(kept)} lines)",
            }
        block = new_string
        if block and not block.endswith("\n"):
            block += "\n"
        updated = "".join(kept[: s - 1]) + block + "".join(kept[e:])
        replacements = 1
    else:
        return {
            "ok": False,
            "error": (
                "provide either `old_string` (STRING mode) or "
                "`start_line`+`end_line` (LINE mode)"
            ),
        }

    if updated == original:
        return {"ok": False, "error": "edit is a no-op (new content identical to old)"}

    try:
        local.write_text(updated, "utf-8")
    except OSError as ex:
        return {"ok": False, "error": f"write failed: {ex}"}
    persisted = local.stat().st_size
    upload_drive_file(workspace_id, clean)
    return {
        "ok": True,
        "drive_path": clean,
        "replacements": replacements,
        "size_bytes": persisted,
        "chars": len(updated),
        "lines": len(updated.splitlines()),
        "diff": _edit_diff(original, updated, clean),
    }


def _resolve_prompt_path(prompt_path: Any, workspace_id: str) -> tuple[str | None, str | None]:
    """Resolve a generate_* `prompt_path` (a drive text file) to the prompt string.

    Lets a skill author a long media prompt ONCE with `file_write` (whose
    result reports the exact `chars`, so the budget check is free) and pass the
    path here, instead of re-emitting the same ~2k chars inside the tool call.
    Reads through the bucket on a local miss, like file_edit. Returns
    (prompt, error)."""
    clean, err = _clean_drive_path(prompt_path)
    if err:
        return None, f"prompt_path: {err}"
    local = workspace_drive(workspace_id) / clean
    if not local.is_file() and not ensure_local_drive_file(workspace_id, clean):
        return None, (
            f"prompt_path file not found: {clean} — `file_write` the prompt "
            f"there first (or pass `prompt` inline)"
        )
    if not local.is_file():
        return None, f"prompt_path file not found: {clean}"
    try:
        text = local.read_text("utf-8", errors="replace").strip()
    except OSError as e:
        return None, f"prompt_path read failed: {e}"
    if not text:
        return None, f"prompt_path file is empty: {clean}"
    return text, None


_LABEL_FIELD_DESCRIPTION = (
    "Optional short progress label shown to the end user in the playground "
    "UI while this call runs. Present-continuous English, max ~40 chars, "
    "written from the user's perspective describing the immediate action — "
    "not the tool name. Examples: 'Downloading reference image', "
    "'Upscaling photo', 'Compressing because too big', 'Searching the web', "
    "'Writing final answer'. OMIT for internal/plumbing calls that should "
    "not be surfaced to the end user (e.g. resolving a drive URL between "
    "two user-facing steps) — leave `_label` out entirely in that case."
)


def _inject_label_field(spec: dict) -> dict:
    """Return a copy of `spec` with an optional `_label` string field added
    to its input_schema, so the model can emit a per-call progress label
    that the playground UI surfaces while the call runs.

    `_label` is placed first in `properties` but intentionally NOT marked
    required: the agent omits it for internal calls it doesn't want
    surfaced to the end user (e.g. resolving a drive URL between two
    user-facing steps). It is stripped from the input before dispatch
    (see run_agent) so handlers and user-tool schema validators never
    see it.
    """
    out = dict(spec)
    schema = dict(spec.get("input_schema") or {"type": "object", "properties": {}})
    props = dict(schema.get("properties") or {})
    if "_label" not in props:
        props = {
            "_label": {"type": "string", "description": _LABEL_FIELD_DESCRIPTION},
            **props,
        }
    schema["properties"] = props
    out["input_schema"] = schema
    return out


def _coerce_json_arg(v):
    """Models frequently serialize a nested object/array tool-arg as a JSON
    *string* (e.g. inputs="{\\"video\\": ...}", refs="[\\"u1\\"]"). Parse it back
    so isinstance/schema checks see the real value; leave ordinary text and
    non-strings untouched. Only strings that begin with `{` or `[` are parsed,
    so a plain prompt/URL is never mangled, and a parse failure returns the
    original value unchanged."""
    if isinstance(v, str) and v.strip() and v.lstrip()[:1] in "{[":
        try:
            return json.loads(v)
        except (ValueError, TypeError):
            return v
    return v


def _coerce_output_args(output_schema, value):
    """Parse back top-level set_output fields the schema declares as
    object/array but the model sent as a JSON *string* (same failure mode
    `_coerce_json_arg` handles for other tools). Only declared object/array
    properties are touched, so a legitimately string-typed field that happens
    to start with `{`/`[` is never mangled."""
    if not isinstance(output_schema, dict) or not isinstance(value, dict):
        return value
    props = output_schema.get("properties")
    if not isinstance(props, dict):
        return value
    out = dict(value)
    for k, p in props.items():
        if (
            isinstance(p, dict)
            and p.get("type") in ("object", "array")
            and isinstance(out.get(k), str)
        ):
            out[k] = _coerce_json_arg(out[k])
    return out


def _check_output_payload(output_schema, value) -> str | None:
    """Validate a set_output payload exactly the way main.py will after the
    run (prune undeclared keys, then all-properties-required). Returns the
    error message instead of raising, so the agent loop can hand it back as a
    tool error and let the model fix its arguments — instead of recording a
    bad output that hard-fails the job after the run has already ended."""
    try:
        pruned = prune_extras(output_schema, value)
        Draft202012Validator(to_output_jsonschema(output_schema)).validate(pruned)
    except ValidationError as e:
        path = ".".join(str(p) for p in e.absolute_path) or "<root>"
        # e.message embeds the offending value — cap it so a huge payload
        # doesn't flood the transcript (the model only needs path + reason).
        return f"output validation failed at `{path}`: {e.message[:1500]}"
    return None


def _build_tools(
    skill: LoadedSkill, *, platform_enabled: bool = True
) -> tuple[list[dict], bool]:
    """Returns (tools, has_set_output).

    User tools come from `skill.tools` (LoadedTool list) with their declared
    input schemas. Output schemas are enforced at dispatch time in run_agent.

    `platform_enabled` is the open-core switch: when False (a local run) the
    platform-only built-ins (memory / media / web / drive-bucket helpers — see
    PLATFORM_ONLY_TOOLS) are dropped from the offered tool list, so the model is
    never handed a capability that can't exist offline.
    Every tool gets an auto-injected optional `_label` field so the model
    can attach a short progress label the playground UI surfaces — omitted
    when the call should stay internal.

    Tool names MUST be globally unique in the array we hand the Anthropic
    client — it 400s ("tools: Tool names must be unique") on any duplicate,
    which hard-fails the whole agent run. So we assemble by name and let the
    platform built-ins (and `set_output`) WIN over a skill-declared tool of
    the same name: the run_agent dispatcher matches built-in names before the
    `tools_by_name` fallback, so a shadowing skill tool could never be
    dispatched to its own entrypoint anyway. A skill that declares the same
    name twice is also collapsed. This makes the 400 impossible by
    construction regardless of what a (possibly older, immutable) deployment
    bundle declares.
    """
    by_name: dict[str, dict] = {}
    dropped: list[str] = []

    # 1) skill-declared tools first (de-dup within the skill's own list).
    for t in skill.tools:
        if t.name in by_name:
            dropped.append(t.name)
            continue
        by_name[t.name] = _inject_label_field(
            {
                "name": t.name,
                "description": t.description or f"User tool `{t.name}`",
                "input_schema": to_jsonschema(t.input_schema),
            }
        )

    # 2) Platform-provided tools, auto-discovered from agent_tool_specs.
    # Only `bash` is conditional. A built-in OVERRIDES any skill tool of the
    # same name (built-in handler is what the dispatcher would run anyway).
    for spec in BUILTIN_AGENT_TOOLS:
        if spec["name"] == "bash" and skill.disable_bash:
            continue
        # Local run: hosted-only built-ins are switched off (open-core line).
        if not platform_enabled and spec["name"] in PLATFORM_ONLY_TOOLS:
            continue
        if spec["name"] in by_name:
            dropped.append(spec["name"])  # a skill tried to shadow this built-in
        by_name[spec["name"]] = _inject_label_field(spec)

    # 3) set_output last (reserved; also overrides any skill tool named it).
    if skill.is_adhoc:
        # Ad-hoc / inline subagents (a bundle `*.md` or an inline prompt run via
        # run_subagent / subagent.run) have no declared output_schema. Give them
        # a free-form set_output that records any JSON object verbatim, so a
        # stage can return structured data to its caller without a manifest
        # contract.
        has_set_output = True
        if "set_output" in by_name:
            dropped.append("set_output")
        by_name["set_output"] = _inject_label_field(
            {
                "name": "set_output",
                "description": (
                    "Record the final output object for this subagent run. "
                    "Calling this ends the run. Pass any JSON object — it is "
                    "returned to whoever invoked this prompt."
                ),
                "input_schema": {"type": "object", "additionalProperties": True},
            }
        )
    else:
        has_set_output = skill.output_schema is not None
        if has_set_output:
            if "set_output" in by_name:
                dropped.append("set_output")
            by_name["set_output"] = _inject_label_field(
                set_output_tool_spec(skill.output_schema)  # type: ignore[arg-type]
            )

    # Least-privilege tool scope (P1-5 / P2-9): if the skill declares an
    # `allowed_tools` whitelist, drop everything outside it from the offered set
    # — built-ins AND its own declared tools. `set_output` is reserved run
    # infrastructure (it ends the run), never gated, so a tight allowlist can't
    # strand a schema skill that can't finish.
    allow = getattr(skill, "allowed_tools", None)
    if allow:
        allow_set = set(allow) | {"set_output"}
        by_name = {n: spec for n, spec in by_name.items() if n in allow_set}

    if dropped:
        log.warning(
            "tool_name_collision_dropped",
            skill=getattr(skill, "name", None),
            dropped=sorted(set(dropped)),
        )
    return list(by_name.values()), has_set_output


def _call_media(
    model: str,
    inputs: dict,
    workspace_id: str,
    job_id: str,
    *,
    verb: str | None = None,
    output_path: str | None = None,
    output_dir: str | None = None,
    output_url_path: str | None = None,
    kind: str = "auto",
) -> dict:
    """Call our own /v1/media/generate endpoint (sync). Backs the `media`
    built-in agent tool and the generate_image/video/audio verbs (verb set).

    `persist=False`: the API runs fal + billing and returns the (temporary) fal
    `output_url`; WE stream it onto local disk (so a follow-up bash/file_read/
    stitch reads it with no extra fetch) and then push it to the bucket (so the
    pipeline preview, served by the API from the bucket, shows it)."""
    import httpx

    s = get_settings()
    # The API resolves any drive-path input (refs, an image to edit) to a signed
    # bucket URL for Fal, so push those local files to the bucket first.
    push_input_files(workspace_id, inputs or {})
    body = {
        "workspace_id": workspace_id,
        "job_id": job_id,
        "verb": verb,
        "model": model,
        "inputs": inputs or {},
        "output_path": output_path,
        "output_dir": output_dir,
        "output_url_path": output_url_path,
        "kind": kind,
        "persist": False,
    }
    # A transient network error must come back as a soft {ok: False} (like
    # _call_web / _call_subagent_invoke), NOT a raise — the dispatch turns it into
    # a tool error the agent can react to, and a concurrent media batch (gather)
    # mustn't have one HTTP hiccup abort its siblings.
    try:
        r = httpx.post(
            f"{s.api_base.rstrip('/')}/v1/media/generate",
            headers={
                "X-Puras-Service-Token": service_token(),
                "Content-Type": "application/json",
            },
            json=body,
            timeout=600,
        )
    except httpx.HTTPError as e:
        return {"ok": False, "error": f"media/generate request failed: {e}"}
    if not r.is_success:
        return {"ok": False, "error": f"media/generate {r.status_code}: {r.text[:500]}"}
    try:
        data = r.json()
    except ValueError as e:
        return {"ok": False, "error": f"media/generate bad response: {e}"}
    out_url = data.pop("output_url", None)
    drive_path = data.get("drive_path")
    if out_url and drive_path:
        try:
            _persist_url_to_drive(out_url, workspace_id, drive_path)
        except Exception as e:
            return {"ok": False, "error": f"failed to save media to drive: {e}"}
        # The file is now local (bash/file_read/stitch reach it with no extra
        # fetch). Push it to the bucket too so the pipeline preview — served by
        # the API, which only reads the bucket — shows it immediately.
        try:
            upload_drive_file(workspace_id, drive_path)
        except Exception:
            log.warning("media_bucket_push_failed", drive_path=drive_path, exc_info=True)
    return {"ok": True, **data}


def _persist_url_to_drive(url: str, workspace_id: str, drive_path: str) -> None:
    """Stream a URL into the local workspace drive (chunked → bounded memory for
    large video outputs). Used to land a media verb's fal output on local disk so
    a follow-up bash/file_read/stitch reads it with no extra fetch; the caller
    then pushes it to the bucket."""
    import httpx

    dest = workspace_drive(workspace_id) / drive_path.lstrip("/")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Stream to a unique temp, then atomically rename into place. Two concurrent
    # writers to the SAME drive_path (e.g. a same-turn media batch that reused an
    # output_path) each produce a whole file and the last rename wins — instead of
    # interleaving their chunks into one half-written, corrupt file.
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{threading.get_ident()}.part")
    with httpx.Client(timeout=httpx.Timeout(300.0), follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
    os.replace(tmp, dest)


# /v1/subagent/invoke caps body.timeout at 1800s (api schemas — the long-poll
# can't hold an API connection open longer). Anything above 422s the WHOLE
# dispatch, and the model usually omits `timeout` — so an over-cap default here
# made every first cross-skillpack run_subagent bounce on a 422 and burn a
# retry round-trip (see jobs d9edd295/1b919e94). Clamp at this boundary; the
# in-process dispatch path never reaches it and has no cap.
INVOKE_TIMEOUT_CAP_S = 1800


def _clamp_invoke_timeout(timeout) -> int:
    try:
        t = int(timeout)
    except (TypeError, ValueError):
        t = INVOKE_TIMEOUT_CAP_S
    return max(1, min(t, INVOKE_TIMEOUT_CAP_S))


def _call_subagent_invoke(
    parent_job_id: str,
    target: str | None,
    inputs: dict,
    *,
    prompt: str | None = None,
    version: int | None = None,
    timeout: int = INVOKE_TIMEOUT_CAP_S,
) -> dict:
    """Call /v1/subagent/invoke synchronously for a subagent dispatch.

    The agent runs in-process inside the worker, so we re-use the same
    service-token path the SDK uses from subprocesses. Pass exactly one of
    `target` (skill ref / bundle `*.md` path) or `prompt` (inline prompt).
    `version` pins a skill `target` to a specific deployment version of its
    skillpack. Returns `{ok: True, **SubagentInvokeOut}` on a non-error HTTP,
    or `{ok: False, ...}`.
    """
    import httpx

    s = get_settings()
    timeout = _clamp_invoke_timeout(timeout)
    body = {
        "parent_job_id": parent_job_id,
        "target": target,
        "prompt": prompt,
        "inputs": inputs or {},
        "version": version,
        "timeout": timeout,
    }
    try:
        r = httpx.post(
            f"{s.api_base.rstrip('/')}/v1/subagent/invoke",
            headers={
                "X-Puras-Service-Token": service_token(),
                "Content-Type": "application/json",
            },
            json=body,
            timeout=timeout + 30,
        )
    except httpx.HTTPError as e:
        return {"ok": False, "error": f"subagent/invoke request failed: {e}"}
    if not r.is_success:
        return {"ok": False, "error": f"subagent/invoke {r.status_code}: {r.text[:500]}"}
    return {"ok": True, **r.json()}


def _validate_subagent_inputs(input_schema: dict | None, inputs: dict) -> None:
    if not input_schema:
        return
    try:
        Draft202012Validator(to_jsonschema(input_schema)).validate(inputs)
    except ValidationError as e:
        path = ".".join(str(p) for p in e.absolute_path) or "<root>"
        msg = f"subagent input invalid at `{path}`: {e.message}"
        # The model calling run_subagent never sees the target's real input
        # schema (run_subagent advertises `inputs` as a free-form object), so a
        # bare "X is a required property" sends it guessing field names. Append
        # the actual expected shape so it self-corrects in one step instead of
        # looping. Pass `inputs` as a JSON object matching these fields.
        hint = _input_schema_summary(input_schema)
        if hint:
            msg += "\n\nThe subagent expects `inputs` to be an object with:\n\n" + hint
        raise ValueError(msg) from e


def _validate_subagent_output(output_schema: dict | None, value):
    if not output_schema:
        return value
    pruned = prune_extras(output_schema, value)
    try:
        Draft202012Validator(to_output_jsonschema(output_schema)).validate(pruned)
    except ValidationError as e:
        path = ".".join(str(p) for p in e.absolute_path) or "<root>"
        raise ValueError(f"subagent output failed schema at `{path}`: {e.message}") from e
    return pruned


def _resolve_local_subagent(
    deployment: ResolvedDeployment,
    parent_skill: LoadedSkill,
    *,
    target: str | None,
    prompt: str | None,
) -> LoadedSkill | None:
    """Resolve a subagent dispatch to a skill IN THE PARENT'S OWN deployment, so
    it can run in-process (no queued child job, no second worker slot → no
    deadlock). Returns the LoadedSkill, or None if the ref points at another
    skillpack/workspace (the caller then falls back to the job/HTTP path).
    Raises ValueError on a malformed local ref.

    Mirrors the resolver in /v1/subagent/invoke for the local cases:
      - prompt              → inline subagent in the parent's bundle
      - "references/x.md"    → ad-hoc subagent (.md) in the parent's skill dir
      - "skill" / "parent/sub" → a skill in the SAME deployment manifest
    """
    if prompt is not None:
        return load_inline(deployment.root, prompt)

    ref = (target or "").strip()
    parent_top = (parent_skill.name or "").split("/", 1)[0]

    if ref.endswith(".md"):
        rel = ref.lstrip("/")
        # Bundle path is relative to the parent's skill dir; prefix it unless the
        # caller already wrote the full path (matches the API's normalization).
        if parent_top and rel.split("/", 1)[0] != parent_top:
            rel = f"{parent_top}/{rel}"
        return load_adhoc(deployment.root, rel)

    # Bare/qualified skill ref: try the local manifest. For a bare ref, prefer
    # the caller's subskill namespace (`<parent_top>/<ref>`) before a top-level
    # name, exactly like _load_tool / the API resolver.
    candidates: list[str] = []
    if parent_top and "/" not in ref:
        candidates.append(f"{parent_top}/{ref}")
    candidates.append(ref)
    for cand in candidates:
        try:
            return load_skill(deployment.manifest, deployment.root, cand)
        except ValueError:
            continue
    return None  # not in this deployment → cross-skillpack (fallback to job path)


async def _run_inproc_subagent(
    *,
    ctx: RunContext,
    job_id: UUID,
    workspace_id: UUID,
    deployment: ResolvedDeployment,
    child_skill: LoadedSkill,
    child_inputs: dict,
    secrets: dict[str, str] | None,
    unique_key: str,
    depth: int,
    out_dir: str | None = None,
    use_cache: bool = True,
) -> dict:
    """Run a same-deployment child skill as a nested IN-PROCESS agent (or
    function) under the parent's job_id — events + cost fold into the parent job
    (so the per-run cost cap bounds the whole pipeline), and the only isolation is
    a fresh sub-workdir + the child's own system prompt / message history. This is
    the Claude-Code subagent model (nested loop), not a queued job. Returns the
    same {ok, status, result/error} shape as _call_subagent_invoke."""
    try:
        _validate_subagent_inputs(child_skill.input_schema, child_inputs)
    except ValueError as e:
        return {"ok": True, "status": "failed", "error": str(e)}

    # Sibling sub-workdir, keyed off the parent job + the tool_use id so nested /
    # concurrent subagents never collide. Cleaned up in the finally.
    sub_id = f"{job_id}.sub.{unique_key}"
    sub_wd = create_workdir(sub_id, str(workspace_id), child_inputs)
    try:
        attach_skill(sub_wd, child_skill.root)
        child_python, child_venv = await asyncio.to_thread(
            build_skill_python, child_skill.root
        )
        if child_skill.is_agentic:
            # Tag every event this child emits with its own ctx (== the subagent's
            # id) so the UI nests them under this subagent's node — essential when
            # sibling subagents run concurrently and their events interleave. The
            # token-reset restores the parent's ctx for the serial case (the child
            # runs in the parent's task), and is harmless in the concurrent case
            # (the child runs in its own task with a copied context).
            _ctx_token = event_ctx.set(unique_key)
            try:
                res = await run_agent(
                    ctx.session, job_id, workspace_id, deployment, child_skill,
                    child_inputs, sub_wd, secrets,
                    python_exe=child_python, venv_dir=child_venv, depth=depth + 1,
                    out_dir=out_dir, use_cache=use_cache,
                    # Hand the child the SAME ctx so a local run's subagents also
                    # run offline (and hosted folds their events/cost into the
                    # parent exactly as before — DbRunContext is per (session,job)).
                    ctx=ctx,
                )
            finally:
                event_ctx.reset(_ctx_token)
            out = _validate_subagent_output(child_skill.output_schema, res.get("output"))
            return {"ok": True, "status": "succeeded", "result": out}

        out = await asyncio.to_thread(
            run_function,
            f"{child_skill.py_module}:{child_skill.py_func}",
            child_inputs, sub_wd, deployment.root, child_python, secrets,
            str(workspace_id), str(job_id),
        )
        if not out.get("ok"):
            return {"ok": True, "status": "failed", "error": out.get("error") or "function failed"}
        result_value = _validate_subagent_output(child_skill.output_schema, out.get("result"))
        return {"ok": True, "status": "succeeded", "result": result_value}
    except Exception as e:
        # A child failure is reported back to the PARENT agent as a soft tool
        # error (matching the old queued-job behavior) so the parent LLM can
        # react — it does NOT crash the parent job. Pipeline-global stops
        # (cancellation, the per-run cost cap) are NOT lost: the child commits
        # its cost/events per step, and the parent loop re-checks is_cancelled +
        # the cost cap on its very next step and aborts there. asyncio
        # CancelledError is a BaseException, so a real task cancellation still
        # propagates past this `except Exception`.
        log.info("subagent_failed", job_id=str(job_id), error=f"{type(e).__name__}: {e}")
        return {"ok": True, "status": "failed", "error": f"{type(e).__name__}: {e}"}
    finally:
        cleanup_workdir(sub_id)


async def _dispatch_subagent(
    *,
    ctx: RunContext,
    job_id: UUID,
    workspace_id: UUID,
    deployment: ResolvedDeployment,
    parent_skill: LoadedSkill,
    secrets: dict[str, str] | None,
    target: str | None,
    prompt: str | None,
    child_inputs: dict,
    timeout: int,
    depth: int,
    unique_key: str,
    label: str,
    version: int | None = None,
    out_dir: str | None = None,
    use_cache: bool = True,
) -> dict:
    """Route a subagent dispatch: run it IN-PROCESS when it resolves inside the
    parent's own deployment (the common case — inline / .md / same-skillpack,
    deadlock-free), else fall back to the queued-job /v1/subagent/invoke path
    (cross-skillpack / cross-workspace public calls).

    A pinned `version` may name a deployment other than the one this job is
    running, which the in-process path (it reuses the parent's already-loaded
    bundle) can't serve — so any pinned call goes straight to the queued
    /v1/subagent/invoke path, which resolves the exact (skillpack, version)."""
    if version is None:
        try:
            child_skill = _resolve_local_subagent(
                deployment, parent_skill, target=target, prompt=prompt
            )
        except ValueError as e:
            return {"ok": True, "status": "failed", "error": str(e)}

        if child_skill is not None:
            # `sid` ties start↔end↔the child's own events (which carry ctx=sid)
            # together by id, so the UI nests them correctly even when sibling
            # subagents run concurrently and their events interleave. It matches
            # the run_subagent tool_use id (unique_key).
            await ctx.emit_event("subagent_start",
                {"label": label, "depth": depth + 1, "inproc": True,
                 "sid": unique_key},
            )
            out = await _run_inproc_subagent(
                ctx=ctx, job_id=job_id, workspace_id=workspace_id,
                deployment=deployment, child_skill=child_skill,
                child_inputs=child_inputs, secrets=secrets,
                unique_key=unique_key, depth=depth, out_dir=out_dir,
                use_cache=use_cache,
            )
            await ctx.emit_event("subagent_end",
                {"label": label, "status": out.get("status"), "sid": unique_key},
            )
            return out

    # Not in this deployment (another skillpack/workspace) or a pinned version:
    # keep the queued-job path (the API resolves cross-workspace policy,
    # publisher secrets, and the exact pinned deployment). This needs the
    # platform — a local run can only dispatch subagents within its own bundle.
    if not ctx.platform_enabled:
        return {
            "ok": True,
            "status": "failed",
            "error": (
                "cross-skillpack subagents need the Puras platform and aren't "
                "available on a local run; only same-bundle subagents (inline "
                "prompts, `references/*.md`, and skills in this skillpack) run "
                "offline"
            ),
        }
    return await asyncio.to_thread(
        _call_subagent_invoke, str(job_id), target, child_inputs,
        prompt=prompt, version=version, timeout=timeout,
    )


def _call_web(path: str, body: dict) -> dict:
    """Call our own /v1/web/<path> endpoint (sync). Backs the web_search /
    image_search / web_fetch / download_url built-in agent tools."""
    import httpx

    s = get_settings()
    try:
        r = httpx.post(
            f"{s.api_base.rstrip('/')}/v1/web/{path}",
            headers={
                "X-Puras-Service-Token": service_token(),
                "Content-Type": "application/json",
            },
            json=body,
            timeout=90,
        )
    except httpx.HTTPError as e:
        return {"ok": False, "error": f"web/{path} request failed: {e}"}
    if not r.is_success:
        return {"ok": False, "error": f"web/{path} {r.status_code}: {r.text[:500]}"}
    return {"ok": True, **r.json()}


def _clean_drive_path(path: Any) -> tuple[str | None, str | None]:
    """Normalize an agent-supplied drive path → (clean_rel, error).

    Strips a leading `drive/`, rejects `..` traversal. Shared by the drive-file
    rendering tools."""
    if not isinstance(path, str) or not path.strip():
        return None, "path must be a non-empty string"
    clean = path.strip().lstrip("/")
    if clean.startswith("drive/"):
        clean = clean[len("drive/") :]
    if ".." in clean.split("/"):
        return None, "'..' segments not allowed in drive paths"
    return clean, None


def _run_web_fetch_js(url: str, max_chars: int) -> dict:
    """Fetch a URL with JavaScript executed (headless Chromium) and return the
    rendered visible text. Backs `web_fetch(render_js=true)`. Runs entirely in
    the worker — no API roundtrip, no upstream billing."""
    from . import browser

    res = browser.render(url=url, screenshot=False)
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error", "render failed")}
    text = res.get("text") or ""
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…[truncated]"
    return {
        "ok": True,
        "url": res.get("final_url") or url,
        "title": res.get("title", ""),
        "content": text,
        "length": len(text),
        "rendered": True,
        "console_errors": res.get("console_errors") or [],
        "billed_micros": 0,
    }


def _run_web_screenshot(
    args: dict, workspace_id: str, model_slug: str, out_dir: str | None = None
) -> dict:
    """Render a URL or a drive HTML file and capture a PNG into the drive.

    Returns a dict with `ok`, a `content` payload for the tool_result (a block
    list with a text summary + the screenshot image on vision models, else a
    JSON string), plus `drive_path` / `output_url` for the lifecycle event."""
    import base64 as _b64
    import uuid as _uuid

    from . import browser

    url = sanitize_url(args.get("url"))
    raw_path = args.get("path")
    if bool(url) == bool(raw_path):
        return {"ok": False, "content": "ERROR: provide exactly one of 'url' or 'path'"}

    file_path: str | None = None
    if raw_path:
        clean, err = _clean_drive_path(raw_path)
        if err:
            return {"ok": False, "content": f"ERROR: {err}"}
        local = workspace_drive(workspace_id) / clean
        if not local.is_file() and not ensure_local_drive_file(workspace_id, clean):
            return {"ok": False, "content": f"ERROR: file not found in drive: {clean}"}
        file_path = str(local)
    elif not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return {"ok": False, "content": "ERROR: url must start with http:// or https://"}

    # Resolve output path (where the PNG lands in the drive). With no explicit
    # output_path, default into this run's folder so screenshots land alongside
    # the run's other outputs instead of a shared root `screenshots/` pile.
    default_dir = out_dir or "screenshots"
    out_clean, err = _clean_drive_path(
        args.get("output_path") or f"{default_dir}/shot-{_uuid.uuid4().hex[:8]}.png"
    )
    if err:
        return {"ok": False, "content": f"ERROR: bad output_path: {err}"}
    if not out_clean.lower().endswith(".png"):
        out_clean += ".png"

    res = browser.render(
        url=url if url else None,
        file_path=file_path,
        viewport_width=int(args.get("viewport_width") or 1280),
        viewport_height=int(args.get("viewport_height") or 800),
        full_page=bool(args.get("full_page")),
        wait_ms=int(args.get("wait_ms") if args.get("wait_ms") is not None else 1200),
        screenshot=True,
        scroll_y=int(args.get("scroll_y") or 0),
    )
    if not res.get("ok"):
        return {"ok": False, "content": f"ERROR: {res.get('error', 'render failed')}"}

    png: bytes | None = res.get("screenshot_png")
    if not png:
        return {"ok": False, "content": "ERROR: screenshot produced no bytes"}

    # Write locally (immediately readable by bash/file_read), then push to the
    # bucket so the API can serve the screenshot.
    out_local = workspace_drive(workspace_id) / out_clean
    out_local.parent.mkdir(parents=True, exist_ok=True)
    out_local.write_bytes(png)
    try:
        upload_drive_file(workspace_id, out_clean)
    except Exception:
        log.warning("screenshot_bucket_push_failed", drive_path=out_clean, exc_info=True)

    console_errors = res.get("console_errors") or []
    summary = {
        "drive_path": out_clean,
        "title": res.get("title", ""),
        "console_errors": console_errors,
        "rendered_text_preview": (res.get("text") or "")[:500],
    }

    # A very tall capture (e.g. a full_page screenshot of a long report) can exceed
    # the model's per-side pixel limit and 400 the whole job. The full-resolution
    # PNG is already saved to the drive above; if it's oversized we keep the text
    # summary but DON'T attach the image — and tell the agent how to get a usable
    # one (a single viewport, optionally with scroll_y) instead of crashing.
    from .attachments import oversize_image_reason

    oversize = oversize_image_reason(png, "image/png")

    # Attach the screenshot itself so a vision model can look at it directly —
    # mirrors how `file_read` returns image blocks.
    content: list[dict] | str
    if oversize:
        summary["image_omitted"] = (
            f"{oversize}; the full-res PNG is saved at {out_clean}. To see it, "
            f"capture a single viewport (omit full_page) and use scroll_y to grab "
            f"lower sections."
        )
        content = json.dumps(summary, default=str)
    elif model_supports_vision(model_slug):
        content = [
            {"type": "text", "text": json.dumps(summary, default=str)},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": _b64.b64encode(png).decode("ascii"),
                },
            },
        ]
    else:
        content = json.dumps(summary, default=str)

    return {
        "ok": True,
        "content": content,
        "drive_path": out_clean,
        "console_errors": console_errors,
    }


# download_url runs IN the worker (not the API) so the bytes land on local disk
# first — a `file_read` / `bash` on the returned drive_path reads it immediately
# with no extra fetch — and are then pushed to the bucket. Mirrors
# _run_web_screenshot.
_DOWNLOAD_TIMEOUT_S = 60
_DOWNLOAD_MAX_BYTES = 50 * 1024 * 1024  # 50MB hard cap per download
_DOWNLOAD_USER_AGENT = "Mozilla/5.0 (compatible; PurasBot/1.0)"


def _run_download_url(url: Any, raw_path: Any, workspace_id: str) -> dict:
    """Fetch a URL over plain HTTP(S) and save it into the workspace drive.

    Returns {ok, drive_path, bytes, content_type, billed_micros} on success,
    else {ok: False, error}. The body is streamed so an oversized download is
    aborted before it's fully buffered. A `path` that ends in '/' or whose last
    segment has no extension is treated as a directory and the filename is
    inferred from the (post-redirect) URL — matching the prior API behavior."""
    import uuid
    from pathlib import PurePosixPath
    from urllib.parse import urlparse

    import httpx

    url = sanitize_url(url)
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return {"ok": False, "error": "url must start with http:// or https://"}
    clean, err = _clean_drive_path(raw_path)
    if err:
        return {"ok": False, "error": f"bad path: {err}"}

    try:
        with httpx.Client(
            timeout=httpx.Timeout(_DOWNLOAD_TIMEOUT_S),
            follow_redirects=True,
            headers={"User-Agent": _DOWNLOAD_USER_AGENT},
        ) as client:
            with client.stream("GET", url) as resp:
                if not resp.is_success:
                    return {
                        "ok": False,
                        "error": f"download returned HTTP {resp.status_code}",
                    }
                parts: list[bytes] = []
                total = 0
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if total > _DOWNLOAD_MAX_BYTES:
                        return {
                            "ok": False,
                            "error": f"download exceeds {_DOWNLOAD_MAX_BYTES} bytes",
                        }
                    parts.append(chunk)
                blob = b"".join(parts)
                ctype = (
                    resp.headers.get("content-type", "application/octet-stream")
                    .split(";")[0]
                    .strip()
                    or "application/octet-stream"
                )
                final_url = str(resp.url)
    except (httpx.HTTPError, httpx.InvalidURL) as e:
        return {"ok": False, "error": f"download failed: {e}"}

    if clean.endswith("/") or "." not in PurePosixPath(clean).name:
        fname = (
            PurePosixPath(urlparse(final_url).path).name
            or f"download-{uuid.uuid4().hex[:8]}"
        )
        base = clean.rstrip("/")
        clean = f"{base}/{fname}" if base else fname

    out_local = workspace_drive(workspace_id) / clean
    out_local.parent.mkdir(parents=True, exist_ok=True)
    out_local.write_bytes(blob)
    # Local first (bash/file_read see it now), then push to the bucket so it's
    # servable and signable for an upstream.
    try:
        upload_drive_file(workspace_id, clean)
    except Exception:
        log.warning("download_url_bucket_push_failed", drive_path=clean, exc_info=True)

    return {
        "ok": True,
        "drive_path": clean,
        "bytes": len(blob),
        "content_type": ctype,
        "billed_micros": 0,
        "url": final_url,
    }


_FILE_SHAPE_KEYS = ("drive_path", "url", "base64", "data")
_PURAS_FILE_TYPES = frozenset({"image", "video", "audio", "file"})


def _looks_like_file_shape(v: Any) -> bool:
    return isinstance(v, dict) and any(k in v for k in _FILE_SHAPE_KEYS)


# File-ish URL path extensions for the ad-hoc heuristic below. Deliberately
# media/document only: a page URL (.html, no extension) is NOT a file — it's
# something the subagent reads as text and fetches itself if it wants to.
_ADHOC_FILE_EXTS = (
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff",
    ".mp4", ".mov", ".webm", ".mkv", ".avi",
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac",
    ".pdf",
)


def _adhoc_file_like(v: Any) -> bool:
    """Conservative file detector for schema-less (ad-hoc) inputs.

    Only flags *unambiguous* file values — a file-shape dict (`{drive_path}` /
    `{url}` / `{base64}`), a `data:` URI, or a single-token http(s) URL whose
    path ends in a media/document extension. Unlike `_normalize_file_value`,
    a bare relative string is NOT treated as a drive path here: with no schema
    to lean on, that would shadow ordinary text inputs (a brief, a name) and
    try to stage them as files.

    A bare URL string must clear two bars a real text input fails:
      - no internal whitespace once trimmed — a brief like
        "https://apps.apple.com/...\\ntake screenshots from here" is TEXT whose
        first line happens to be a URL; staging it downloads garbage (or 404s
        once the URL gets whitespace-folded) and kills the subagent;
      - a file-ish extension on the URL path — a store/product PAGE url is
        research material the subagent web_fetches itself, not a file to pull.
    """
    if isinstance(v, dict):
        return _looks_like_file_shape(v)
    if not isinstance(v, str):
        return False
    if v.startswith("data:"):
        return True
    s = v.strip()
    if not s.startswith(("http://", "https://")) or any(c.isspace() for c in s):
        return False
    from urllib.parse import urlparse

    return urlparse(s).path.lower().endswith(_ADHOC_FILE_EXTS)


def _normalize_file_value(v: Any) -> dict | None:
    """Normalize any of the three file input shapes to the dict form
    `load_attachment` expects. Returns None if `v` doesn't look like a file.
    """
    if isinstance(v, dict):
        if any(k in v for k in _FILE_SHAPE_KEYS):
            if isinstance(v.get("url"), str):
                return {**v, "url": sanitize_url(v["url"])}
            return v
        return None
    if isinstance(v, str) and v:
        v = sanitize_url(v) if v.startswith(("http://", "https://")) else v
        if v.startswith("data:"):
            # data URI → split media_type + base64 body
            try:
                head, body = v.split(",", 1)
                meta = head[len("data:") :]
                if ";base64" in meta:
                    media_type = meta.replace(";base64", "")
                    return {
                        "base64": body,
                        "media_type": media_type or "application/octet-stream",
                    }
            except ValueError:
                return None
            return None
        if v.startswith(("http://", "https://")):
            return {"url": v}
        # Bare relative path: assume drive_path.
        return {"drive_path": v}
    return None


def _required_inputs_hint(input_schema: dict | None) -> str:
    """One-line `name (type, required/optional)` list for a validation error,
    so the end user sees exactly which fields the skill expects."""
    if not isinstance(input_schema, dict):
        return ""
    props = input_schema.get("properties")
    if not isinstance(props, dict) or not props:
        return ""
    required = set(input_schema.get("required") or [])
    parts: list[str] = []
    for name, p in props.items():
        if not isinstance(p, dict):
            continue
        t = p.get("type") or "any"
        if t == "array" and isinstance(p.get("items"), dict) and p["items"].get("type"):
            t = f"{p['items']['type']}[]"
        parts.append(f"`{name}` ({t}, {'required' if name in required else 'optional'})")
    return ", ".join(parts)


def _resolve_input_bytes(value: Any, workspace_id: str) -> bytes | None:
    """Return the bytes behind a file-input value when they're locally knowable
    (a drive file or an inline base64/data URI), else None for a remote `url`
    (which the run fetches) or a non-file value. Raises FileNotFoundError when a
    declared drive file genuinely isn't there."""
    normalized = _normalize_file_value(value)
    if normalized is None:
        return None
    if "drive_path" in normalized:
        clean, err = _clean_drive_path(normalized["drive_path"])
        if err or clean is None:
            raise ValueError(err or "bad drive path")
        local = workspace_drive(workspace_id) / clean
        if not local.is_file() and not ensure_local_drive_file(workspace_id, clean):
            raise FileNotFoundError(clean)
        return local.read_bytes()
    # NB: an empty-string base64 is the bug we're hunting — keep `""` (falsy),
    # don't let `or` collapse it to the `data` key or to None.
    raw_b64 = normalized.get("base64")
    if raw_b64 is None:
        raw_b64 = normalized.get("data")
    if isinstance(raw_b64, str):
        import base64 as _b64
        try:
            return _b64.b64decode(raw_b64, validate=True)
        except Exception as e:
            raise ValueError(f"invalid base64: {e}") from e
    return None  # url-only — leave it to the run


def validate_input_files(
    input_schema: dict | None, inputs: dict, workspace_id: str
) -> None:
    """Pre-flight the declared file inputs BEFORE the agent (and any model spend).

    The JSON-schema pass (`main._validate`) only checks SHAPE: a well-formed
    `{drive_path}` clears it even when the file behind it is empty (0 bytes) or
    not a real image. The failure then surfaces deep in the run — an empty image
    block 400s the model call (`image cannot be empty`), an oversized one 400s on
    the pixel limit — as an opaque error the user can't act on. Resolve each
    declared file input here and fail fast with a message that names the field
    and lists the skill's inputs, so a bad upload is rejected up front.

    Only locally-resolvable shapes are byte-checked (drive_path / base64 / data);
    a remote `url` is left to the run. A transient drive miss is NOT treated as a
    bad input — it's left for the run to surface rather than blocking the job.
    """
    from .attachments import IMAGE_MIMES, _guess_mime, _sniff_image_mime, oversize_image_reason

    if not isinstance(input_schema, dict) or not isinstance(inputs, dict):
        return
    props = input_schema.get("properties")
    if not isinstance(props, dict):
        return

    def _check(field: str, idx: int | None, value: Any, ftype: str) -> None:
        slot = field if idx is None else f"{field}[{idx}]"
        try:
            data = _resolve_input_bytes(value, workspace_id)
        except FileNotFoundError:
            return  # transient/bucket miss — let the run materialize or surface it
        except ValueError as e:
            raise ValueError(
                f"input `{slot}` is not a usable {ftype}: {e}. Re-upload a valid "
                f"{ftype}. Skill inputs — {_required_inputs_hint(input_schema)}."
            ) from e
        if data is None:
            return  # url-only or not a file value (shape pass already covered it)
        if not data:
            raise ValueError(
                f"input `{slot}` is an empty {ftype} (0 bytes) — nothing was "
                f"uploaded, or the upload was truncated. Re-upload a valid "
                f"{ftype}. Skill inputs — {_required_inputs_hint(input_schema)}."
            )
        if ftype == "image":
            mime = _sniff_image_mime(data) or _guess_mime(field)
            if mime not in IMAGE_MIMES:
                raise ValueError(
                    f"input `{slot}` is not a valid image (its bytes aren't a "
                    f"PNG/JPEG/GIF/WebP). Re-upload a real image. Skill inputs — "
                    f"{_required_inputs_hint(input_schema)}."
                )
            reason = oversize_image_reason(data, mime)
            if reason:
                raise ValueError(
                    f"input `{slot}`: {reason}. Resize it and re-upload. Skill "
                    f"inputs — {_required_inputs_hint(input_schema)}."
                )

    for name, p in props.items():
        if not isinstance(p, dict) or name not in inputs:
            continue
        t = p.get("type")
        if t in _PURAS_FILE_TYPES:
            _check(name, None, inputs[name], t)
        elif (
            t == "array"
            and isinstance(p.get("items"), dict)
            and p["items"].get("type") in _PURAS_FILE_TYPES
        ):
            seq = inputs[name]
            if isinstance(seq, list):
                it = p["items"]["type"]
                for i, v in enumerate(seq):
                    _check(name, i, v, it)


_PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent / "worker_prompt.md"
_PROMPT_TEMPLATE = _PROMPT_TEMPLATE_PATH.read_text("utf-8")


def _render_template(template: str, vars: dict[str, Any]) -> str:
    """Minimal Mustache-like substitution used to build the agent system prompt.

    Syntax (see `worker_prompt.md`):
      - `{{var}}` — substitute with `str(vars[var])`. Missing → empty string.
      - `{{?var}}...{{/var}}` — include block only when `vars[var]` is truthy.
    """
    import re

    def repl_cond(m: re.Match[str]) -> str:
        return m.group(2) if vars.get(m.group(1)) else ""

    out = re.sub(
        r"\{\{\?(\w+)\}\}(.*?)\{\{/\1\}\}", repl_cond, template, flags=re.DOTALL
    )

    def repl_var(m: re.Match[str]) -> str:
        v = vars.get(m.group(1))
        return "" if v is None or isinstance(v, bool) else str(v)

    out = re.sub(r"\{\{(\w+)\}\}", repl_var, out)
    return out


def _build_system_prompt(
    skill_body: str, inputs_summary: str, has_output_schema: bool
) -> str:
    """Render the agent system prompt from `worker_prompt.md`.

    All injection points live in that template; runtime supplies the three
    dynamic slots:
      - `skill_body` — verbatim SKILL.md (or any `.md` entrypoint)
      - `inputs_summary` — markdown from `_input_schema_summary` (may be "")
      - `has_output_schema` — toggles the `set_output` reminder block
    """
    return _render_template(
        _PROMPT_TEMPLATE,
        {
            "skill_body": skill_body.rstrip(),
            "inputs_summary": inputs_summary,
            "has_output_schema": has_output_schema,
        },
    ).rstrip() + "\n"


def _input_schema_summary(schema: dict | None) -> str:
    """Compact markdown summary of input_schema for the system prompt.

    Auto-appended so a skill author doesn't have to redocument their inputs
    in the SKILL.md body — the schema's `description`, type, and constraints
    become a structured "Inputs" section the agent reads alongside the JSON
    values that arrive in the first user message.

    When any input is a file type (top-level or as array items), a
    "File inputs" paragraph is appended explaining that those files are
    already attached and ready to use — no fetch/upload step required.
    """
    if not isinstance(schema, dict):
        return ""
    props = schema.get("properties")
    if not isinstance(props, dict) or not props:
        return ""
    required = set(schema.get("required") or [])
    lines: list[str] = []
    file_fields: list[str] = []
    for name, p in props.items():
        if not isinstance(p, dict):
            continue
        t = p.get("type") or "any"
        req = "required" if name in required else "optional"
        constraints: list[str] = []
        if isinstance(p.get("enum"), list):
            constraints.append("one of " + ", ".join(repr(x) for x in p["enum"]))
        if "minimum" in p or "maximum" in p:
            mn = p.get("minimum")
            mx = p.get("maximum")
            constraints.append(
                f"{mn if mn is not None else '−∞'}..{mx if mx is not None else '∞'}"
            )
        if "minLength" in p or "maxLength" in p:
            mn = p.get("minLength")
            mx = p.get("maxLength")
            constraints.append(f"length {mn or 0}..{mx or '∞'}")
        if isinstance(p.get("items"), dict):
            it = p["items"].get("type")
            if it:
                constraints.append(f"items: {it}")
        head = f"- `{name}` ({t}, {req}"
        if constraints:
            head += "; " + "; ".join(constraints)
        head += ")"
        desc = (p.get("description") or "").strip()
        if desc:
            head += f" — {desc}"
        lines.append(head)
        if t in _PURAS_FILE_TYPES:
            file_fields.append(name)
        elif _items_file_type(p):
            file_fields.append(f"{name}[]")
    if not lines:
        return ""
    out = "## Inputs\n\nThe first user message carries these fields:\n\n" + "\n".join(
        lines
    )
    if file_fields:
        listed = ", ".join(f"`{n}`" for n in file_fields)
        out += (
            "\n\n**File inputs are staged in the workspace drive, not inlined.** "
            f"{listed} appear in the first user message as `drive_path:` + "
            "`url:` pairs — not as embedded image/document blocks. Each gives "
            "you two equivalent handles to the same file:\n"
            "- `url:` — a fresh signed URL (TTL ~1h). Pass to tools that need "
            "a URL (e.g. the `media` tool's `image_url` / `elements`, an HTTP "
            "request).\n"
            "- `drive_path:` — the file's path inside the workspace drive. Pass "
            "to drive-aware tools (`file_read`, `drive_url`).\n\n"
            "If your skill just forwards a file to a tool, hand over the path "
            "or URL — there is no need to load its contents. Call `file_read` "
            "only when you actually need to *look* at the file (e.g. to "
            "describe it in a prompt). Either way, **do NOT** `web_fetch`, "
            "`download_url`, `image_search`, or otherwise re-fetch/re-upload "
            "these files — they are already provisioned for this job."
        )
    return out


def _items_file_type(prop_schema: dict | None) -> str | None:
    """If `prop_schema` is `array` with file-typed `items`, return the item type."""
    if not isinstance(prop_schema, dict):
        return None
    if prop_schema.get("type") != "array":
        return None
    items = prop_schema.get("items")
    if not isinstance(items, dict):
        return None
    t = items.get("type")
    if isinstance(t, str) and t in _PURAS_FILE_TYPES:
        return t
    return None


def _stage_file_inputs(
    inputs: dict,
    workspace_id: str,
    job_id: str,
    input_schema: dict | None,
    heuristic: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Materialize every file-typed input into the workspace drive.

    Two motivations:
      1. The agent doesn't need to *see* every file at job start — auto-
         inlining as base64 image blocks burns input tokens for files the
         skill may only forward to a tool. Let the skill decide via
         `file_read` if it needs to look.
      2. Tools like `media` and `drive_url` want a path/URL inside the
         workspace drive. By staging URL/base64 inputs to a known job-local
         spot, the agent can hand over a `drive_path` (or fresh signed
         `url`) for any file without an extra fetch.

    For each file-typed property (scalar or array items):
      - `drive_path` shape  → kept in place; we only mint a fresh signed URL.
      - `url` shape         → downloaded, written under
                              `_jobs/<job_id>/<field>[/<i>].<ext>`.
      - `base64` / data-URI → decoded, written to the same staging spot.

    Returns:
      - `staged`: shallow copy of `inputs` with file values replaced by
        `{drive_path, url}` (or list thereof). Non-file fields pass through.
      - `labels`: one short label per staged file, for telemetry.
    """
    import base64 as b64lib
    import mimetypes

    import httpx

    from .drive import workspace_drive

    s = get_settings()
    drive_root = workspace_drive(workspace_id)

    props: dict[str, Any] = {}
    if isinstance(input_schema, dict):
        raw_props = input_schema.get("properties")
        if isinstance(raw_props, dict):
            props = raw_props

    labels: list[str] = []

    def _slot_label(field: str, idx: int | None) -> str:
        return field if idx is None else f"{field}[{idx}]"

    def _stage_one(value: Any, field: str, idx: int | None) -> dict[str, str]:
        slot = _slot_label(field, idx)
        normalized = _normalize_file_value(value)
        if normalized is None:
            raise ValueError(f"input `{slot}` is not a file value: {value!r}")

        if "drive_path" in normalized:
            dp = normalized["drive_path"].strip().lstrip("/")
            if dp.startswith("drive/"):
                dp = dp[len("drive/") :]
            full = (drive_root / dp).resolve()
            try:
                full.relative_to(drive_root.resolve())
            except ValueError as e:
                raise ValueError(
                    f"input `{slot}`: drive path escapes workspace root"
                ) from e
            # Make sure the input is on local disk before handing it to the
            # subagent — already-local files are a no-op; one produced by an
            # earlier job (bucket-only) is pulled on-miss. Same materialization
            # `ensure_input_files` does for declared inputs, at the subagent
            # boundary. Local presence means the subagent's own reads hit too.
            if not ensure_local_drive_file(workspace_id, dp):
                raise FileNotFoundError(f"input `{slot}`: drive file not found: {dp}")
            url = signed_url(s.drive_bucket, f"{workspace_id}/{dp}", _STAGED_URL_TTL_SECONDS)
            labels.append(f"{slot} → drive:{dp}")
            return {"drive_path": dp, "url": url}

        if "url" in normalized:
            url_src = normalized["url"]
            # Fetching a remote input is the most common transient failure — a
            # brief upstream disconnect ("Server disconnected without sending a
            # response") would otherwise kill the whole job. The GET is
            # idempotent, so retry transport-level errors a few times before
            # giving up. HTTP 4xx/5xx (raise_for_status) are NOT retried.
            data = b""
            ct = ""
            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    with httpx.Client(timeout=30, follow_redirects=True) as c:
                        r = c.get(url_src)
                        r.raise_for_status()
                        data = r.content
                        ct = (r.headers.get("content-type") or "").split(";")[0].strip()
                    break
                except (httpx.InvalidURL, httpx.UnsupportedProtocol) as e:
                    # A malformed URL can never succeed on retry. InvalidURL is
                    # NOT an httpx.HTTPError subclass — uncaught it would escape
                    # as a raw crash that fails the whole (sub)job with an
                    # unactionable message; name the input instead so the
                    # caller can fix it.
                    raise ValueError(f"input `{slot}`: invalid URL: {e}") from e
                except httpx.HTTPStatusError as e:
                    raise ValueError(
                        f"input `{slot}`: URL returned HTTP "
                        f"{e.response.status_code}"
                    ) from e
                except httpx.TransportError as e:
                    last_exc = e
                    if attempt < 2:
                        time.sleep(0.5 * (attempt + 1))
            else:
                raise ValueError(
                    f"input `{slot}`: could not fetch URL after 3 attempts "
                    f"({type(last_exc).__name__})"
                ) from last_exc
            mime = (
                normalized.get("media_type")
                or ct
                or (mimetypes.guess_type(url_src)[0])
                or "application/octet-stream"
            )
        elif "base64" in normalized:
            try:
                data = b64lib.b64decode(normalized["base64"], validate=True)
            except Exception as e:
                raise ValueError(f"input `{slot}`: invalid base64: {e}") from e
            mime = normalized.get("media_type") or "application/octet-stream"
        else:
            raise ValueError(
                f"input `{slot}`: missing drive_path / url / base64"
            )

        ext = mimetypes.guess_extension(mime) or ".bin"
        if idx is None:
            rel = f"_jobs/{job_id}/{field}{ext}"
        else:
            rel = f"_jobs/{job_id}/{field}/{idx}{ext}"
        target = drive_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        url = signed_url(s.drive_bucket, f"{workspace_id}/{rel}", _STAGED_URL_TTL_SECONDS)
        labels.append(f"{slot} → staged:{rel}")
        return {"drive_path": rel, "url": url}

    def _stage_guessed(value: Any, field: str, idx: int | None) -> Any:
        # Heuristic staging is a GUESS — if the guessed file can't be staged
        # (a dead URL, a missing drive path), pass the original value through
        # as text instead of failing the whole subagent over our own guess.
        # Declared-schema file inputs keep the hard error: the caller
        # explicitly said "this is a file".
        try:
            return _stage_one(value, field, idx)
        except (ValueError, FileNotFoundError) as e:
            labels.append(f"{_slot_label(field, idx)} → passed through unstaged ({e})")
            return value

    staged: dict[str, Any] = {}
    for k, v in inputs.items():
        prop_schema = props.get(k) if isinstance(props.get(k), dict) else None
        declared_type = (prop_schema or {}).get("type")
        if isinstance(declared_type, str) and declared_type in _PURAS_FILE_TYPES:
            staged[k] = _stage_one(v, k, None)
        elif _items_file_type(prop_schema) and isinstance(v, list):
            staged[k] = [_stage_one(item, k, i) for i, item in enumerate(v)]
        elif heuristic and prop_schema is None and _adhoc_file_like(v):
            # Schema-less (ad-hoc) input: stage clear file values by shape so
            # the subagent can look at them; pass everything else through.
            staged[k] = _stage_guessed(v, k, None)
        elif (
            heuristic
            and prop_schema is None
            and isinstance(v, list)
            and v
            and all(_adhoc_file_like(item) for item in v)
        ):
            staged[k] = [_stage_guessed(item, k, i) for i, item in enumerate(v)]
        else:
            staged[k] = v

    return staged, labels


def _build_initial_user_text(staged_inputs: dict, adhoc: bool = False) -> str:
    """Plain-text first user message describing the staged inputs.

    File values are rendered as `drive_path:` + `url:` lines (no inline
    bytes). Non-file values are JSON-encoded so the agent can parse them
    unambiguously.

    For ad-hoc / inline subagents (`run_subagent` / `subagent.run` with a `.md`
    ref or inline prompt) the
    whole input dict is handed over as one JSON block — the caller passed a
    dictionary, so the subagent reads it straight from this message; there's no
    `_inputs.json` round-trip to do (file handles keep their `{drive_path,url}`
    so the subagent can `file_read` or sign them).
    """
    def _is_file_handle(v: Any) -> bool:
        return isinstance(v, dict) and "drive_path" in v and "url" in v

    if adhoc:
        body = json.dumps(staged_inputs, ensure_ascii=False, indent=2)
        return (
            "Your inputs, passed by the caller as a dictionary — use them "
            "directly. File values are `{drive_path, url}` handles: `file_read` "
            "the `drive_path` to look at the file, or pass the `url` to a tool "
            "that needs a URL.\n\n"
            f"```json\n{body}\n```"
        )

    lines: list[str] = ["Inputs for this job:", ""]
    for k, v in staged_inputs.items():
        if _is_file_handle(v):
            lines.append(f"{k}:")
            lines.append(f"  drive_path: {v['drive_path']}")
            lines.append(f"  url: {v['url']}")
        elif isinstance(v, list) and v and all(_is_file_handle(x) for x in v):
            lines.append(f"{k}: ({len(v)} file{'s' if len(v) != 1 else ''})")
            for i, x in enumerate(v):
                lines.append(f"  [{i}] drive_path: {x['drive_path']}")
                lines.append(f"      url: {x['url']}")
        else:
            lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    return "\n".join(lines)


def _resolve_escalation(routing: dict | None, current_slug: str) -> dict | None:
    """Build the escalation plan from a skill's `routing` policy, or None when
    there's nothing to escalate to. The premium provider is built lazily (on the
    first escalation) so a run that never escalates pays no setup cost.

    Returns `{slug, on, after, _provider, _family, _variant}` where `_provider`
    starts None and is filled on escalation. Defensive: an unknown/degenerate
    escalate_to (e.g. == the current model) disables escalation rather than
    raising — the manifest parser already validated the well-formed case."""
    if not isinstance(routing, dict):
        return None
    esc_slug = routing.get("escalate_to")
    if (
        not isinstance(esc_slug, str)
        or not esc_slug
        or esc_slug == current_slug
        or not is_known_slug(esc_slug)
    ):
        return None
    on = routing.get("on") or ["schema_fail"]
    after = routing.get("after")
    after = after if isinstance(after, int) and after >= 1 else 2
    return {
        "slug": esc_slug,
        "on": list(on),
        "after": after,
        "_provider": None,
        "_family": esc_slug.partition("/")[0],
        "_variant": esc_slug.partition("/")[2],
    }


async def _local_confirm(
    ctx: RunContext, name: str, inputs: dict
) -> tuple[str, str | None]:
    """Console approve/deny for a `confirm:` tool on a local run (no dashboard).

    The operator launched the run themselves, so they ARE the reviewer — we
    prompt on the console and read the answer. Fail-closed: a non-interactive
    stdin (no TTY), EOF, or anything other than y/yes denies, since the gate
    guards a real side effect. Returns the same (decision, reason) shape the
    hosted approval flow does."""
    await ctx.emit_event("confirm_requested", {"tool": name, "input": inputs})
    prompt = (
        f"\n⚠  This run wants to use `{name}` (a confirm-gated action).\n"
        f"   inputs: {json.dumps(inputs, default=str)[:500]}\n"
        f"   approve? [y/N] "
    )
    try:
        answer = (await asyncio.to_thread(input, prompt)).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "denied", "no interactive console to approve on"
    if answer in ("y", "yes"):
        return "approved", None
    return "denied", "the operator declined"


async def run_agent(
    session: AsyncSession,
    job_id: UUID,
    workspace_id: UUID,
    deployment: ResolvedDeployment,
    skill: LoadedSkill,
    inputs: dict,
    workdir: Path,
    secrets: dict[str, str] | None = None,
    *,
    python_exe: str,
    venv_dir: Path | None,
    depth: int = 1,
    model_override: str | None = None,
    out_dir: str | None = None,
    use_cache: bool = True,
    ctx: RunContext | None = None,
) -> dict[str, Any]:
    s = get_settings()

    # Per-run deliverables folder (`<skill-slug>/<jobshort>`). Normally computed in
    # main.py and passed in (threaded into nested subagents via this same arg); the
    # None fallback covers direct callers/tests. `generate_*` and `web_screenshot`
    # default their output here, and the platform relocates the final deliverable
    # here at end-of-job no matter where the skill wrote it — so every run's outputs
    # land in one browsable, skill-grouped folder, not the old flat
    # `media/`+`screenshots/`+`_jobs/` scatter. The skill itself stays oblivious.
    if out_dir is None:
        out_dir = resolve_output_dir(str(workspace_id), skill.name, job_id)

    # Resolve the public model slug (`family/variant`) to its upstream
    # routing info. The slug is what we surface in events, the dashboard,
    # and pricing; the upstream provider/id stays internal.
    #
    # A per-run override (playground model picker → jobs.media_overrides["text"])
    # wins over the skill's declared model, but only when it's a known slug —
    # otherwise a stale picker option would break the run, so we fall back to the
    # skill's own model.
    override = model_override if model_override and is_known_slug(model_override) else None
    model_slug = override or skill.model or s.default_model_slug
    info = resolve_model(model_slug)
    family, _, variant = model_slug.partition("/")
    provider = make_provider(info.upstream_provider, info.upstream_id)

    # Model routing/escalation (P0-2b): run on the cheap `model_slug` and switch
    # to the skill's `routing.escalate_to` once a trigger fires (today:
    # `schema_fail` — the cheap model can't produce schema-valid set_output after
    # `after` attempts). Resolved once; the premium provider is built lazily on
    # the first escalation. A per-run override that already equals escalate_to
    # disables it (nothing to escalate to).
    escalation = _resolve_escalation(skill.routing, model_slug)
    escalated = False
    schema_fail_count = 0

    # RunContext is the seam between this loop and its environment (P1-6): the
    # hosted DbRunContext routes platform I/O to Postgres/the API, while a
    # LocalRunContext runs the SAME loop offline (`puras run --local`). The loop's
    # I/O — events, usage, cancellation, the cost cap, and checkpoints — all flow
    # through `ctx` (behavior-identical for hosted: DbRunContext just delegates to
    # the queue helpers). The concurrent dispatch rebinds a per-task ctx onto its
    # own session via `with_session`.
    #
    # A caller may inject a `ctx` (the local runner passes a LocalRunContext, and
    # a nested in-proc subagent passes its parent's ctx); hosted top-level runs
    # leave it None and we build the DbRunContext from the session. When a ctx is
    # supplied we rebind `session` to `ctx.session` so the not-yet-abstracted
    # platform DB paths (memory injection, the prompt cache) read the right one
    # (None offline — those paths are gated on `ctx.platform_enabled`).
    if ctx is None:
        ctx = DbRunContext(session, job_id, workspace_id)
    else:
        session = ctx.session

    tools_by_name: dict[str, LoadedTool] = {t.name: t for t in skill.tools}
    # Local runs (ctx.platform_enabled False) drop the hosted-only built-ins.
    tools, has_set_output = _build_tools(skill, platform_enabled=ctx.platform_enabled)

    # System prompt is rendered from `worker_prompt.md` — see that file for
    # the full shape and the named-ref injection points.
    system_prompt = _build_system_prompt(
        skill_body=skill.system_prompt or "",
        inputs_summary=_input_schema_summary(skill.input_schema),
        has_output_schema=has_set_output,
    )

    bash_env = {
        **(secrets or {}),  # skillpack secrets first; we override the few keys below
        # Make user's venv binaries available in bash (e.g. installed pip CLIs)
        "PATH": f"{venv_dir / 'bin' if venv_dir else ''}:"
        + str(__import__("os").environ.get("PATH", "")),
        # The skill bundle is mounted flat at workdir root via per-entry
        # symlinks (see workdir.attach_skill), so imports under the skill
        # dir resolve from the workdir itself; no separate deployment-root
        # entry is needed.
        "PYTHONPATH": f"{workdir}",
        "PURAS_WORKSPACE_ID": str(workspace_id),
    }

    # Resume from a checkpoint (top-level runs only): a requeued job — handed back
    # to the queue by a deploy drain or the reaper after its worker died — whose
    # previous attempt checkpointed restores the conversation from the last clean
    # turn and CONTINUES, instead of restarting from scratch. The restored
    # `messages` already hold the staged inputs + memory digest + every prior turn
    # and the drive persists, so on resume we skip re-staging and memory injection.
    ckpt = await ctx.load_checkpoint() if depth == 1 else None
    resumed = ckpt is not None
    start_step = ckpt["step"] if resumed else 0

    if resumed:
        staged_inputs, attachment_labels = {}, []
        initial_content = ""
    else:
        try:
            staged_inputs, attachment_labels = _stage_file_inputs(
                inputs, str(workspace_id), str(job_id), skill.input_schema,
                heuristic=skill.is_adhoc,
            )
        except (ValueError, FileNotFoundError) as e:
            await ctx.emit_event("attachment_error", {"error": str(e)})
            await ctx.commit()
            raise RuntimeError(f"invalid input file: {e}") from e
        initial_content = _build_initial_user_text(staged_inputs, adhoc=skill.is_adhoc)

    # ── Workspace memory injection (top-level jobs only) ──────────────────
    # Derive the subject's identity from inputs, pull the workspace's pinned
    # preferences / brand kit + any prior brief for THIS subject, and prepend a
    # digest to the first user turn so the skill REUSES it (and skips the
    # researcher subagent) instead of re-researching from scratch. The identity
    # hints handed over here are the same keys the skill should `memory_put`
    # with, so the next run matches. Selective (pinned + entity matches, not the
    # whole store) and best-effort: any failure must never break the run.
    # Subagents (depth>1) inherit the digest via the parent's first message and
    # share these built-in tools, so we only inject once at the top. Skipped on a
    # local run — workspace memory is a hosted (Postgres) value-add.
    if depth == 1 and not skill.is_adhoc and not resumed and ctx.platform_enabled:
        try:
            from .memory_store import memory_context

            identity = derive_identity(staged_inputs, str(workspace_id))
            mem = await memory_context(
                session,
                workspace_id,
                entity_keys=identity.get("keys"),
                content_hashes=(
                    [identity["fingerprint"]] if identity.get("fingerprint") else None
                ),
            )
            entity_hits = mem.get("entity", [])
            pinned_hits = mem.get("pinned", [])
            # Only inject when there's something to act on: a memory hit, or at
            # least a derivable identity key the skill can cache against. Skips
            # the block (and its tokens) on keyless jobs in an empty workspace.
            if entity_hits or pinned_hits or identity.get("keys"):
                digest = format_memory_digest(pinned_hits, entity_hits, identity)
                if digest:
                    initial_content = digest + "\n\n---\n\n" + initial_content
            if entity_hits or pinned_hits:
                await ctx.emit_event("memory_injected",
                    {
                        "entity_count": len(entity_hits),
                        "pinned_count": len(pinned_hits),
                        "memory_ids": [r["id"] for r in entity_hits]
                        + [r["id"] for r in pinned_hits],
                        "kinds": sorted({r["kind"] for r in (entity_hits + pinned_hits)}),
                    },
                )
            elif identity.get("keys"):
                await ctx.emit_event("memory_miss", {"keys": identity.get("keys")})
        except Exception as e:  # never let memory break a job
            log.warning("memory_injection_failed", error=str(e), job_id=str(job_id))

    await ctx.emit_event("agent_start",
        {
            "provider": family,
            "model": variant,
            "tool_count": len(tools),
            "skill": skill.name,
            "deployment_id": deployment.deployment_id,
            "workdir": str(workdir),
            "bash_enabled": not skill.disable_bash,
            "has_output_schema": has_set_output,
            "attachments": attachment_labels,
            "resumed": resumed,
        },
    )
    if resumed:
        await ctx.emit_event("resumed", {"from_step": start_step})
    await ctx.commit()

    if resumed:
        messages: list[dict] = ckpt["messages"]
        final_text_parts: list[str] = list(ckpt["final_text_parts"])
        structured_output: dict | None = ckpt["structured_output"]
    else:
        messages = [{"role": "user", "content": initial_content}]
        final_text_parts = []
        structured_output = None
    started = time.monotonic()

    async def _run_media(tu) -> tuple[dict, str | None, str | None]:
        """Dispatch one media verb (generate_image/video/audio | transcribe) and
        return (raw_result, media_verb, media_slug) WITHOUT emitting any events.
        It only does the remote call (in a thread) and never touches the DB
        session, so a batch of these is safe to run concurrently for a same-turn
        media fan-out. The caller emits the tool_use/tool_result in order."""
        args = tu.input if isinstance(tu.input, dict) else {}
        media_verb: str | None
        if tu.name == "transcribe":
            # Speech-to-text: fixed model, returns transcript text+words (no file).
            media_verb = None
            media_slug = "elevenlabs/scribe-v2"
            media_inputs = {"audio_url": args.get("audio")}
            if args.get("keyterms"):
                media_inputs["keyterms"] = _coerce_json_arg(args.get("keyterms"))
            if args.get("language"):
                media_inputs["language_code"] = args.get("language")
        else:
            # Verb call: `model` is a family/auto; the rest is the verb's input bag.
            media_verb = tu.name[len("generate_"):]
            media_slug = args.get("model") or "auto"
            media_inputs = {
                k: v
                for k, v in args.items()
                if k not in ("model", "output_path", "_label")
            }
            # Array fields (refs / keyterms) often arrive as a JSON string from
            # the model — parse them back so the verb adapter sees a real list.
            for _af in ("refs", "keyterms"):
                if _af in media_inputs:
                    media_inputs[_af] = _coerce_json_arg(media_inputs[_af])
            # `prompt_path`: the prompt lives in a drive file the agent already
            # wrote with `file_write` (whose result reported the exact char
            # count) — resolve it here so a long prompt is emitted once, not
            # re-pasted into this call. Exactly one of prompt / prompt_path.
            pp = media_inputs.pop("prompt_path", None)
            if pp is not None:
                if tu.name == "generate_audio":
                    return (
                        {"ok": False,
                         "error": "generate_audio takes `text`, not `prompt_path`"},
                        media_verb,
                        media_slug if isinstance(media_slug, str) else None,
                    )
                if media_inputs.get("prompt"):
                    return (
                        {"ok": False,
                         "error": "pass exactly one of `prompt` / `prompt_path`, not both"},
                        media_verb,
                        media_slug if isinstance(media_slug, str) else None,
                    )
                resolved_prompt, perr = await asyncio.to_thread(
                    _resolve_prompt_path, pp, str(workspace_id)
                )
                if perr:
                    return (
                        {"ok": False, "error": perr},
                        media_verb,
                        media_slug if isinstance(media_slug, str) else None,
                    )
                media_inputs["prompt"] = resolved_prompt
        if not isinstance(media_slug, str) or not media_slug:
            return (
                {"ok": False,
                 "error": "media generation requires a model/family (e.g. 'auto')"},
                media_verb,
                None,
            )
        # An explicit output_path wins; otherwise the API defaults the file into
        # this run's folder (output_dir). A stray leading `drive/` is stripped so
        # the path resolves under the workspace root, not `<root>/drive/…`. Wherever
        # it lands, the platform files the deliverable into the run folder at
        # end-of-job, so the skill never has to know about the folder.
        op = args.get("output_path")
        if isinstance(op, str) and op.startswith("drive/"):
            op = op[len("drive/"):]
        out = await asyncio.to_thread(
            _call_media,
            media_slug,
            media_inputs if isinstance(media_inputs, dict) else {},
            str(workspace_id),
            str(job_id),
            verb=media_verb,
            output_path=op,
            output_dir=out_dir,
            output_url_path=None,
            kind="auto",
        )
        return out, media_verb, media_slug

    # ── Tracing: the run (root) span + a span per step (P0-3) ──
    # The run span is the trace root (or, for an in-proc subagent, a child of the
    # tool span that spawned it — captured here before we touch the var). Each
    # iteration opens a step span whose children are the model call + tool spans;
    # we record them at the loop's exits rather than wrapping the (huge, nested-
    # def) body. trace_id is the job id (RunContext.record_span supplies it).
    # Per-run, per-tool call counts (P2-9): enforces the skill's `tool_limits`
    # and the global per-run cap. Incremented synchronously at dispatch (no await
    # between read+write), so the concurrent dispatch can't race it.
    tool_call_counts: dict[str, int] = {}

    _run_span_id = uuid4().hex[:16]
    _run_parent_span = _current_span.get()
    _run_started = time.monotonic()
    _step_span_id = _run_span_id
    _step_started = _run_started

    async def _close_step() -> None:
        await ctx.record_span(
            span_id=_step_span_id, parent_span_id=_run_span_id, kind="step",
            name=f"step {step}",
            duration_ms=int((time.monotonic() - _step_started) * 1000),
            attributes={"step": step},
        )

    async def _close_run() -> None:
        await ctx.record_span(
            span_id=_run_span_id, parent_span_id=_run_parent_span, kind="run",
            name=f"run:{skill.name}",
            duration_ms=int((time.monotonic() - _run_started) * 1000),
            attributes={"skill": skill.name, "depth": depth, "model": model_slug},
        )

    for step in range(start_step, s.max_agent_steps):
        # Open this step's span: model + tool spans nest under it via the ctxvar.
        _step_span_id = uuid4().hex[:16]
        _step_started = time.monotonic()
        _current_span.set(_step_span_id)
        if time.monotonic() - started > s.max_agent_seconds:
            await ctx.emit_event("timeout", {"step": step})
            await ctx.commit()
            raise TimeoutError(f"agent exceeded {s.max_agent_seconds}s")
        if await ctx.is_cancelled():
            await ctx.emit_event("cancelled", {"step": step})
            await ctx.commit()
            raise RuntimeError("job cancelled")
        # Per-job spend ceiling: read the cost accrued so far (LLM + media, both
        # write jobs.cost_micros) and stop the run before it can overshoot. Caps
        # the blast radius of one expensive run against a user's free credit.
        if s.max_job_cost_micros > 0:
            spent = await ctx.get_job_cost()
            if spent >= s.max_job_cost_micros:
                await ctx.emit_event("cost_capped",
                    {"step": step, "spent_micros": spent,
                     "cap_micros": s.max_job_cost_micros},
                )
                await ctx.commit()
                raise RuntimeError(
                    "This run reached its per-run cost limit and was stopped."
                )

        # Model escalation: when the cheap model has failed the trigger enough
        # times, swap to the premium model for the REST of the run (one-way).
        # Today the only trigger is `schema_fail` — the cheap model couldn't
        # produce a schema-valid set_output after `after` attempts. Building the
        # premium provider lazily here means a run that never escalates never
        # pays for it.
        if (
            escalation
            and not escalated
            and "schema_fail" in escalation["on"]
            and schema_fail_count >= escalation["after"]
        ):
            try:
                if escalation["_provider"] is None:
                    esc_info = resolve_model(escalation["slug"])
                    escalation["_provider"] = make_provider(
                        esc_info.upstream_provider, esc_info.upstream_id
                    )
                prev_slug = model_slug
                provider = escalation["_provider"]
                model_slug = escalation["slug"]
                family, variant = escalation["_family"], escalation["_variant"]
                escalated = True
                await ctx.emit_event("model_escalated",
                    {
                        "from": prev_slug,
                        "to": model_slug,
                        "reason": "schema_fail",
                        "after": schema_fail_count,
                        "step": step,
                    },
                )
                await ctx.commit()
            except Exception:
                # A bad escalation target must never break the run — keep going on
                # the cheap model. Mark escalated so we don't retry every step.
                log.warning("model_escalation_failed", job_id=str(job_id), exc_info=True)
                escalated = True

        # cache_messages: roll a prompt-cache breakpoint forward on the last
        # message each iteration. On the next call the breakpoint becomes the
        # longest-match prefix, so each step only pays write cost on its own
        # delta. No-op on providers without Anthropic-style prompt caching.
        # Exact-match prompt cache (P0-2a): an identical request (same model +
        # system + messages + tools + max_tokens, same workspace) is served from
        # the store with no upstream call and billed 0. `use_cache` is False for
        # eval-suite variance runs so their repeats stay independent.
        async with _span(ctx, "llm", f"model:{model_slug}") as _msp:
            resp, cache_hit = await cached_messages_create(
                session,
                provider,
                model_slug=model_slug,
                workspace_id=workspace_id,
                system=system_prompt,
                messages=messages,
                tools=tools or None,
                max_tokens=16384,
                cache_messages=True,
                # The exact-match prompt cache is Postgres-backed — off for a local
                # run (no DB); the BYO-key user pays their own provider directly.
                use_cache=use_cache and ctx.platform_enabled,
            )
            _msp.attrs.update(
                {
                    "step": step,
                    "model": model_slug,
                    "stop_reason": resp.stop_reason,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cache_hit": cache_hit,
                }
            )

        if cache_hit:
            # Served from the store: no upstream spend. The cached call's cost is
            # what this step SAVED — surfaced on the event/usage row for hit-rate
            # and savings reporting; the wallet is charged 0.
            saved_micros = with_margin(resp.upstream_cost_micros)
            upstream_micros = 0
            billed_micros = 0
        else:
            saved_micros = 0
            upstream_micros = resp.upstream_cost_micros
            billed_micros = with_margin(upstream_micros)
        await ctx.record_usage(
            provider=family,
            model=variant,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            upstream_micros=upstream_micros,
            billed_micros=billed_micros,
            cache_hit=cache_hit,
            meta={
                "step": step,
                "cache_creation_input_tokens": resp.cache_creation_input_tokens,
                "cache_read_input_tokens": resp.cache_read_input_tokens,
                **({"cache_hit": True, "saved_micros": saved_micros} if cache_hit else {}),
                **(
                    {"context_management": resp.context_management_applied}
                    if resp.context_management_applied
                    else {}
                ),
            },
        )
        await ctx.emit_event("model_response",
            {
                "step": step,
                "stop_reason": resp.stop_reason,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "cache_creation_input_tokens": resp.cache_creation_input_tokens,
                "cache_read_input_tokens": resp.cache_read_input_tokens,
                "upstream_cost_micros": upstream_micros,
                "billed_micros": billed_micros,
                "cache_hit": cache_hit,
                **({"saved_micros": saved_micros} if cache_hit else {}),
                **(
                    {"context_management": resp.context_management_applied}
                    if resp.context_management_applied
                    else {}
                ),
            },
        )
        await ctx.commit()

        assistant_blocks: list[dict] = []
        for t in resp.text_blocks:
            assistant_blocks.append({"type": "text", "text": t})
            final_text_parts.append(t)
        for tu in resp.tool_uses:
            assistant_blocks.append(
                {"type": "tool_use", "id": tu.id, "name": tu.name, "input": tu.input}
            )
        messages.append({"role": "assistant", "content": assistant_blocks})
        tool_uses = resp.tool_uses

        if resp.stop_reason != "tool_use" or not tool_uses:
            # Done — close this step's and the run's span before returning.
            await _close_step()
            await _close_run()
            # If schema was required and set_output was never called, fail.
            if has_set_output and structured_output is None:
                raise RuntimeError(
                    "agent finished without calling set_output (skill declares output_schema)"
                )
            if structured_output is not None:
                return {"output": structured_output, "steps": step + 1}
            return {"output": "".join(final_text_parts), "steps": step + 1}

        tool_results: list[dict] = []
        early_return: dict | None = None

        # Dispatch one tool call end-to-end (emit tool_use → run → emit
        # tool_result) on the given `ctx`, returning (content, set_output|None).
        # Factored out so the SAME body serves both the serial path (the main
        # ctx) and the concurrent path (a per-tool ctx bound to a fresh session —
        # see below). Platform I/O (events/usage/cost/cancel) goes through `ctx`;
        # the not-yet-abstracted platform-only DB tools (memory, approvals,
        # cross-skillpack subagents) still reach the raw session via `ctx.session`
        # (aliased here so those call sites read unchanged) — PR3 gates them on
        # ctx.platform_enabled for the local runner.
        async def _dispatch_one_tool(tu, ctx):
            session = ctx.session
            # Count set_output schema failures across the whole run (drives model
            # escalation). set_output is parallel-unsafe → always on the main
            # ctx/serial path, so this increment never races.
            nonlocal schema_fail_count
            # Pop the auto-injected `_label` field before dispatch so handlers
            # and user-tool schema validators never see it. Mutating tu.input
            # also drops it from the assistant message we already appended,
            # which is fine — the model doesn't need to see past labels.
            # `_label` is optional: the agent omits it intentionally for
            # internal calls it doesn't want surfaced to the user (e.g.
            # resolving a drive URL). When missing we still dispatch — the
            # UI uses a fallback title or hides the step.
            label: str | None = None
            if isinstance(tu.input, dict):
                raw = tu.input.pop("_label", None)
                if isinstance(raw, str) and raw.strip():
                    label = raw.strip()[:80]
            event_payload: dict[str, Any] = {
                "name": tu.name,
                "input": tu.input,
                "tool_use_id": tu.id,
            }
            if label:
                event_payload["label"] = label
            await ctx.emit_event("tool_use", event_payload)
            await ctx.commit()

            # Tool scope + rate limit (P1-5 / P2-9). set_output is reserved run
            # infrastructure — never gated. Everything else is checked against the
            # skill's allowlist (defense in depth beyond _build_tools) and its
            # per-run call caps; a violation becomes a soft tool error the model
            # can react to, never a side effect. The count is read+incremented
            # synchronously here (no await between), so concurrent dispatch is safe.
            if tu.name != "set_output":
                if skill.allowed_tools is not None and tu.name not in skill.allowed_tools:
                    content = (
                        f"ERROR: tool '{tu.name}' is not in this skill's allowed_tools "
                        f"scope. Allowed: {', '.join(sorted(skill.allowed_tools)) or '(none)'}."
                    )
                    await ctx.emit_event("tool_result",
                        {"tool_use_id": tu.id, "ok": False, "preview": content[:512]},
                    )
                    await ctx.commit()
                    return content, None
                cap = skill.tool_limits.get(tu.name)
                if cap is None and s.max_tool_calls_per_run > 0:
                    cap = s.max_tool_calls_per_run
                n = tool_call_counts.get(tu.name, 0) + 1
                tool_call_counts[tu.name] = n
                if cap is not None and n > cap:
                    content = (
                        f"ERROR: tool '{tu.name}' hit its per-run limit of {cap} call(s). "
                        f"Stop calling it and finish with what you have."
                    )
                    await ctx.emit_event("tool_result",
                        {"tool_use_id": tu.id, "ok": False, "preview": content[:512]},
                    )
                    await ctx.commit()
                    return content, None

            if tu.name == "set_output":
                _so = (
                    dict(tu.input)
                    if isinstance(tu.input, dict)
                    else {"value": tu.input}
                )
                # Enforce the output schema HERE, while the model can still
                # react: un-stringify any object/array field it serialized as
                # JSON text, then validate. A bad payload becomes a tool error
                # the model retries from — previously it sailed through, ended
                # the run, and main.py's post-run check failed the whole job
                # with no chance to correct (release #10 frontend smoke).
                # Ad-hoc subagents are free-form: record verbatim, no check.
                if skill.output_schema is not None and not skill.is_adhoc:
                    _so = _coerce_output_args(skill.output_schema, _so)
                    err = _check_output_payload(skill.output_schema, _so)
                    if err is not None:
                        # A schema-invalid set_output — the model fumbled the
                        # output contract. Count it; enough of these escalate the
                        # run to the premium model (see the loop-top trigger).
                        schema_fail_count += 1
                        content = (
                            f"ERROR: {err}. Output NOT recorded — fix the "
                            "arguments (pass real JSON types, not stringified "
                            "JSON) and call set_output again."
                        )
                        await ctx.emit_event("tool_result",
                            {
                                "tool_use_id": tu.id,
                                "ok": False,
                                "preview": content[:512],
                            },
                        )
                        await ctx.commit()
                        return content, None
                content = "output recorded"
                await ctx.emit_event("tool_result",
                    {"tool_use_id": tu.id, "ok": True, "preview": content},
                )
                await ctx.commit()
                # set_output ends the run; the caller turns this into early_return.
                # (set_output is parallel-unsafe, so it's only ever on the main
                # session in the serial path.)
                return content, _so

            if tu.name == "bash":
                cmd = (tu.input or {}).get("command", "")
                to = int((tu.input or {}).get("timeout") or 0)
                result = await asyncio.to_thread(_run_bash, cmd, to, workdir, bash_env)
                content = f"exit={result['exit']}\n{result['output']}"
                await ctx.emit_event("tool_result",
                    {
                        "tool_use_id": tu.id,
                        "ok": result["exit"] == 0,
                        "preview": content[:512],
                    },
                )
            elif tu.name == "todo_write":
                # Stateless plan checklist: the model sends the full list each
                # call and we just echo a summary + surface the items on the
                # event so the UI can render a live progress checklist. Nothing
                # is executed; bad shapes are normalized rather than erroring,
                # so a planning misstep never fails the job.
                args = tu.input if isinstance(tu.input, dict) else {}
                raw = _coerce_json_arg(args.get("todos"))
                todos: list[dict] = []
                if isinstance(raw, list):
                    for it in raw:
                        if not isinstance(it, dict):
                            continue
                        c = it.get("content")
                        st = it.get("status")
                        if not isinstance(c, str) or not c.strip():
                            continue
                        if st not in ("pending", "in_progress", "completed"):
                            st = "pending"
                        todos.append({"content": c.strip()[:200], "status": st})
                done = sum(1 for t in todos if t["status"] == "completed")
                in_prog = next(
                    (t["content"] for t in todos if t["status"] == "in_progress"),
                    None,
                )
                content = json.dumps(
                    {"ok": True, "total": len(todos), "completed": done,
                     "in_progress": in_prog}
                )
                await ctx.emit_event("tool_result",
                    {
                        "tool_use_id": tu.id,
                        "ok": True,
                        "preview": content[:512],
                        "tool": "todo_write",
                        "todos": todos,
                    },
                )
            elif tu.name == "file_write":
                args = tu.input if isinstance(tu.input, dict) else {}
                out = await asyncio.to_thread(
                    _run_file_write, args.get("path"), args.get("content"),
                    str(workspace_id),
                )
                ok = bool(out.get("ok"))
                content = json.dumps(out) if ok else f"ERROR: {out.get('error')}"
                await ctx.emit_event("tool_result",
                    {
                        "tool_use_id": tu.id,
                        "ok": ok,
                        "preview": content[:512],
                        "tool": "file_write",
                    },
                )
            elif tu.name == "file_edit":
                args = tu.input if isinstance(tu.input, dict) else {}
                out = await asyncio.to_thread(
                    _run_file_edit, args, str(workspace_id)
                )
                ok = bool(out.get("ok"))
                content = json.dumps(out) if ok else f"ERROR: {out.get('error')}"
                await ctx.emit_event("tool_result",
                    {
                        "tool_use_id": tu.id,
                        "ok": ok,
                        "preview": content[:512],
                        "tool": "file_edit",
                    },
                )
            elif tu.name in ("generate_image", "generate_video", "generate_audio", "transcribe"):
                # Remote generation (no session touch); concurrency comes from
                # this whole dispatch running in its own task on the parallel path.
                out, media_verb, media_slug = await _run_media(tu)
                if out.get("ok"):
                    # The file lives in the drive — return its path, not a URL.
                    # If a downstream step needs a URL (e.g. feeding this back
                    # into another media call's image_url), the agent mints one
                    # with `drive_url`. STT models have no file: surface the
                    # transcript text/words instead.
                    payload: dict[str, Any] = {
                        "drive_path": out.get("drive_path"),
                        "billed_usd": out.get("billed_usd"),
                        "kind": out.get("kind"),
                    }
                    # For verb calls, tell the agent which concrete model ran +
                    # any caps coercion warnings (e.g. a snapped duration).
                    if media_verb and out.get("model"):
                        payload["model"] = out.get("model")
                    _media_warns = (out.get("meta") or {}).get("warnings")
                    if _media_warns:
                        payload["warnings"] = _media_warns
                    if out.get("text") is not None:
                        payload["text"] = out.get("text")
                    if out.get("words") is not None:
                        payload["words"] = out.get("words")
                    content = json.dumps(payload, default=str)
                    await ctx.emit_event("tool_result",
                        {
                            "tool_use_id": tu.id,
                            "ok": True,
                            "preview": content[:512],
                            "billed_micros": out.get("billed_micros"),
                            "drive_path": out.get("drive_path"),
                            "model": out.get("model") or media_slug,
                        },
                    )
                else:
                    content = f"ERROR: {out.get('error')}"
                    await ctx.emit_event("tool_result",
                        {"tool_use_id": tu.id, "ok": False, "preview": content[:512]},
                    )
            elif tu.name == "web_screenshot":
                args = tu.input if isinstance(tu.input, dict) else {}
                out = await asyncio.to_thread(
                    _run_web_screenshot,
                    args,
                    str(workspace_id),
                    model_slug,
                    out_dir,
                )
                content = out["content"]
                preview = (
                    content[:512]
                    if isinstance(content, str)
                    else json.dumps(
                        {
                            "drive_path": out.get("drive_path"),
                            "console_errors": out.get("console_errors"),
                        },
                        default=str,
                    )[:512]
                )
                await ctx.emit_event("tool_result",
                    {
                        "tool_use_id": tu.id,
                        "ok": bool(out.get("ok")),
                        "preview": preview,
                        "tool": "web_screenshot",
                        "drive_path": out.get("drive_path"),
                    },
                )
            elif tu.name in ("web_search", "image_search", "web_fetch"):
                args = tu.input if isinstance(tu.input, dict) else {}
                req: dict[str, Any] = {
                    "workspace_id": str(workspace_id),
                    "job_id": str(job_id),
                }
                endpoint_path: str
                if tu.name == "web_search":
                    endpoint_path = "search"
                    req["query"] = args.get("query", "")
                    if "max_results" in args:
                        req["max_results"] = args["max_results"]
                elif tu.name == "image_search":
                    endpoint_path = "image-search"
                    req["query"] = args.get("query", "")
                    if "max_results" in args:
                        req["max_results"] = args["max_results"]
                else:  # web_fetch
                    endpoint_path = "fetch"
                    req["url"] = sanitize_url(args.get("url", ""))
                    if "max_chars" in args:
                        req["max_chars"] = args["max_chars"]
                # web_fetch(render_js=true) renders in the worker's headless
                # browser instead of the API's plain HTTP fetch.
                if tu.name == "web_fetch" and args.get("render_js"):
                    max_chars = int(args.get("max_chars") or 20000)
                    out = await asyncio.to_thread(
                        _run_web_fetch_js, req["url"], max_chars
                    )
                else:
                    out = await asyncio.to_thread(_call_web, endpoint_path, req)
                if out.get("ok"):
                    payload = {k: v for k, v in out.items() if k != "ok"}
                    content = json.dumps(payload, default=str)
                    if len(content) > 30000:
                        content = content[:30000] + "\n…[truncated]"
                    await ctx.emit_event("tool_result",
                        {
                            "tool_use_id": tu.id,
                            "ok": True,
                            "preview": content[:512],
                            "billed_micros": payload.get("billed_micros"),
                            "tool": tu.name,
                        },
                    )
                else:
                    content = f"ERROR: {out.get('error')}"
                    await ctx.emit_event("tool_result",
                        {"tool_use_id": tu.id, "ok": False, "preview": content[:512]},
                    )
            elif tu.name == "download_url":
                # Runs in-worker (not via the API) so the file lands on local
                # disk — a follow-up file_read/bash reads it immediately — then
                # gets pushed to the bucket. See _run_download_url.
                args = tu.input if isinstance(tu.input, dict) else {}
                out = await asyncio.to_thread(
                    _run_download_url,
                    args.get("url"),
                    args.get("path"),
                    str(workspace_id),
                )
                if out.get("ok"):
                    payload = {
                        "drive_path": out["drive_path"],
                        "bytes": out["bytes"],
                        "content_type": out["content_type"],
                        "billed_micros": out["billed_micros"],
                    }
                    # 0-cost usage_event for analytics/audit parity with the old
                    # API endpoint (op=download_url). Doesn't touch the wallet.
                    await ctx.record_usage(
                        provider="web", model="download_url",
                        input_tokens=0, output_tokens=0,
                        upstream_micros=0, billed_micros=0,
                        meta={"op": "download_url", "url": out.get("url"),
                              "bytes": out["bytes"], "drive_path": out["drive_path"]},
                    )
                    content = json.dumps(payload, default=str)
                    await ctx.emit_event("tool_result",
                        {
                            "tool_use_id": tu.id,
                            "ok": True,
                            "preview": content[:512],
                            "billed_micros": out["billed_micros"],
                            "drive_path": out["drive_path"],
                            "tool": "download_url",
                        },
                    )
                else:
                    content = f"ERROR: {out.get('error')}"
                    await ctx.emit_event("tool_result",
                        {"tool_use_id": tu.id, "ok": False, "preview": content[:512]},
                    )
            elif tu.name == "file_read":
                args = tu.input if isinstance(tu.input, dict) else {}
                paths = _coerce_json_arg(args.get("paths"))
                content = await asyncio.to_thread(
                    _run_file_read,
                    paths,
                    str(workspace_id),
                    model_slug,
                )
                ok = isinstance(content, list)
                preview = (
                    content[:512]
                    if isinstance(content, str)
                    else json.dumps(
                        [
                            (
                                b
                                if b.get("type") == "text"
                                else {
                                    "type": b.get("type"),
                                    "media_type": (b.get("source") or {}).get(
                                        "media_type"
                                    ),
                                }
                            )
                            for b in content
                        ]
                    )[:512]
                )
                await ctx.emit_event("tool_result",
                    {
                        "tool_use_id": tu.id,
                        "ok": ok,
                        "preview": preview,
                        "tool": "file_read",
                        "paths": paths if isinstance(paths, list) else [],
                    },
                )
            elif tu.name == "drive_url":
                args = tu.input if isinstance(tu.input, dict) else {}
                out = await asyncio.to_thread(
                    _resolve_drive_url,
                    args.get("path"),
                    args.get("ttl"),
                    str(workspace_id),
                )
                if out.get("ok"):
                    content = json.dumps(
                        {
                            "url": out["url"],
                            "path": out["path"],
                            "expires_in": out["expires_in"],
                        }
                    )
                else:
                    content = f"ERROR: {out['error']}"
                await ctx.emit_event("tool_result",
                    {
                        "tool_use_id": tu.id,
                        "ok": bool(out.get("ok")),
                        "preview": content[:512],
                        "tool": "drive_url",
                    },
                )
            elif tu.name == "drive_pull":
                # Pull a workspace-drive file from the bucket onto local disk so
                # raw bash (cat/ffmpeg/script) can read it. Tool-layer reads
                # (file_read, generate_, subagent inputs) already pull on-miss;
                # bash can't be intercepted, so the agent asks explicitly.
                args = tu.input if isinstance(tu.input, dict) else {}
                clean, err = _clean_drive_path(args.get("path"))
                if err:
                    ok = False
                    content = f"ERROR: bad path: {err}"
                else:
                    ok = await asyncio.to_thread(
                        ensure_local_drive_file, str(workspace_id), clean
                    )
                    content = (
                        json.dumps({"drive_path": clean, "local_path": f"drive/{clean}"})
                        if ok
                        else f"ERROR: drive file not found: {clean}"
                    )
                await ctx.emit_event("tool_result",
                    {
                        "tool_use_id": tu.id,
                        "ok": bool(ok),
                        "preview": content[:512],
                        "tool": "drive_pull",
                    },
                )
            elif tu.name == "memory_search":
                # Workspace shared-brain lookup. Read-only, hard-scoped to this
                # workspace; inputs are guarded (no raw cast) so a bad arg never
                # aborts the session's transaction.
                from .memory_store import memory_search

                args = tu.input if isinstance(tu.input, dict) else {}
                rows = await memory_search(
                    session,
                    workspace_id,
                    kind=args.get("kind"),
                    entity_key=args.get("key"),
                    query=args.get("query"),
                    content_hash=args.get("content_hash"),
                    mtype=args.get("mtype") if args.get("mtype") in (
                        "semantic", "episodic", "procedural"
                    ) else None,
                    scope=args.get("scope") if args.get("scope") in (
                        "entity", "skillpack", "workspace"
                    ) else None,
                    limit=int(args.get("limit") or 8) if str(args.get("limit") or "").strip().lstrip("-").isdigit() else 8,
                )
                fresh = sum(1 for r in rows if r.get("fresh"))
                content = json.dumps({"count": len(rows), "results": rows}, default=str)
                if len(content) > 30000:
                    content = content[:30000] + "\n…[truncated; memory_get an id for the full record]"
                await ctx.emit_event("tool_result",
                    {
                        "tool_use_id": tu.id,
                        "ok": True,
                        "preview": content[:512],
                        "tool": "memory_search",
                        "count": len(rows),
                        "fresh": fresh,
                    },
                )
            elif tu.name == "memory_get":
                from .memory_store import memory_get

                args = tu.input if isinstance(tu.input, dict) else {}
                raw_id = args.get("id")
                rec = None
                ok = True
                try:
                    UUID(str(raw_id))
                except (ValueError, TypeError):
                    content = "ERROR: memory_get requires a valid `id` (uuid)"
                    ok = False
                else:
                    rec = await memory_get(session, workspace_id, raw_id)
                    if rec is None:
                        content = "ERROR: no memory record with that id in this workspace"
                        ok = False
                    else:
                        content = json.dumps(rec, default=str)
                await ctx.emit_event("tool_result",
                    {
                        "tool_use_id": tu.id,
                        "ok": ok,
                        "preview": content[:512],
                        "tool": "memory_get",
                    },
                )
            elif tu.name == "memory_put":
                from .memory_store import memory_put

                args = tu.input if isinstance(tu.input, dict) else {}
                record = _coerce_json_arg(args.get("record"))
                kind = args.get("kind")
                ok = True
                if not isinstance(kind, str) or not kind.strip():
                    content = "ERROR: memory_put requires a `kind` string"
                    ok = False
                elif not isinstance(record, dict) or not record:
                    content = "ERROR: memory_put requires `record` as a non-empty JSON object"
                    ok = False
                else:
                    aliases = _coerce_json_arg(args.get("aliases"))
                    tags = _coerce_json_arg(args.get("tags"))
                    # Parse stale_at to a real datetime up front so a malformed
                    # string can't raise a DB cast error mid-transaction.
                    stale = None
                    stale_raw = args.get("stale_at")
                    if isinstance(stale_raw, str) and stale_raw.strip():
                        from datetime import datetime as _dt
                        try:
                            stale = _dt.fromisoformat(stale_raw.strip().replace("Z", "+00:00"))
                        except ValueError:
                            stale = None
                    # `supersedes` must be a real uuid or it's dropped — a bad
                    # value must never abort the session's transaction.
                    supersedes = None
                    try:
                        supersedes = str(UUID(str(args.get("supersedes"))))
                    except (ValueError, TypeError):
                        pass
                    importance = args.get("importance")
                    if not isinstance(importance, (int, float)):
                        importance = None
                    res = await memory_put(
                        session,
                        workspace_id,
                        kind=kind.strip(),
                        record=record,
                        entity_key=(args.get("entity_key") or None),
                        mtype=args.get("mtype") if args.get("mtype") in (
                            "semantic", "episodic", "procedural"
                        ) else "semantic",
                        scope=args.get("scope") if args.get("scope") in (
                            "entity", "skillpack", "workspace"
                        ) else "entity",
                        title=args.get("title"),
                        summary=(args.get("summary") or None),
                        tags=tags if isinstance(tags, list) else None,
                        content_hash=args.get("content_hash"),
                        source_url=args.get("source_url"),
                        source_type="agent",
                        aliases=aliases if isinstance(aliases, list) else None,
                        pinned=bool(args.get("pinned")),
                        importance=importance,
                        supersedes=supersedes,
                        stale_at=stale,
                        source_job_id=job_id,
                    )
                    content = json.dumps(res, default=str)
                await ctx.emit_event("tool_result",
                    {
                        "tool_use_id": tu.id,
                        "ok": ok,
                        "preview": content[:512],
                        "tool": "memory_put",
                        **({"memory_id": json.loads(content).get("id")} if ok else {}),
                    },
                )
            elif tu.name == "memory_forget":
                from .memory_store import memory_forget

                args = tu.input if isinstance(tu.input, dict) else {}
                ok = True
                try:
                    forget_id = str(UUID(str(args.get("id"))))
                except (ValueError, TypeError):
                    content = "ERROR: memory_forget requires a valid `id` (uuid)"
                    ok = False
                else:
                    forgotten = await memory_forget(
                        session,
                        workspace_id,
                        forget_id,
                        reason=(args.get("reason") or None),
                    )
                    if forgotten:
                        content = json.dumps({"id": forget_id, "forgotten": True})
                    else:
                        content = "ERROR: no active memory record with that id in this workspace"
                        ok = False
                await ctx.emit_event("tool_result",
                    {
                        "tool_use_id": tu.id,
                        "ok": ok,
                        "preview": content[:512],
                        "tool": "memory_forget",
                    },
                )
            elif tu.name == "describe_subagent":
                # Introspect a subagent target's input/output schema so the
                # agent can shape `inputs` correctly BEFORE run_subagent. The
                # run_subagent tool never advertises the target's schema, so
                # this closes that gap and prevents guess-and-retry loops.
                args = tu.input if isinstance(tu.input, dict) else {}
                d_target = args.get("target")
                d_target = d_target.strip() if isinstance(d_target, str) else None
                child = None
                if not d_target:
                    content = (
                        "ERROR: describe_subagent requires a `target` "
                        "(skill ref or `references/*.md` path)"
                    )
                else:
                    try:
                        child = _resolve_local_subagent(
                            deployment, skill, target=d_target, prompt=None
                        )
                    except ValueError as e:
                        content = f"ERROR: {e}"
                    else:
                        if child is None:
                            content = (
                                f"`{d_target}` resolves to a skill in another "
                                "skillpack/workspace; its schema can't be "
                                "introspected here. Pass `inputs` as that "
                                "skill's documented inputs."
                            )
                        elif child.input_schema:
                            payload = {
                                "target": d_target,
                                "input_schema": to_jsonschema(child.input_schema),
                                "required": list(
                                    child.input_schema.get("required") or []
                                ),
                                "inputs_summary": _input_schema_summary(
                                    child.input_schema
                                ),
                            }
                            if child.output_schema:
                                payload["output_schema"] = to_output_jsonschema(
                                    child.output_schema
                                )
                            content = json.dumps(payload, default=str)
                        else:
                            content = (
                                f"`{d_target}` is a free-form subagent (a `.md` "
                                "bundle prompt or inline prompt) — no declared "
                                "input schema. Pass whatever `inputs` that prompt "
                                "expects; file-shaped values (URLs / drive paths) "
                                "are staged and attached automatically."
                            )
                await ctx.emit_event("tool_result",
                    {
                        "tool_use_id": tu.id,
                        "ok": not content.startswith("ERROR:"),
                        "preview": content[:512],
                        "tool": "describe_subagent",
                    },
                )
            elif tu.name == "run_subagent":
                # Isolated subagent run. `target` may be a `references/*.md`
                # bundle path (→ ad-hoc subagent in this skillpack) or a skill
                # ref (same/other skillpack, incl. cross-workspace); or pass
                # `prompt` for a one-off inline subagent in this bundle.
                # Same-deployment dispatches run IN-PROCESS as a nested agent
                # (no queued child job → no slot deadlock); only cross-skillpack
                # refs fall back to the queued /v1/subagent/invoke path. See
                # _dispatch_subagent.
                args = tu.input if isinstance(tu.input, dict) else {}
                target = args.get("target")
                prompt = args.get("prompt")
                # The model frequently serializes the nested `inputs` object as a
                # JSON *string* (e.g. inputs="{\"video\": ...}"). Parse it before
                # the dict check — otherwise it silently collapses to {} and the
                # subagent's schema validation reports a misleading
                # "'<field>' is a required property", which sends the agent into a
                # retry death-spiral that burns tokens. Tolerate stringy input the
                # same way `version` is tolerated below.
                raw_inputs = args.get("inputs")
                if isinstance(raw_inputs, str) and raw_inputs.strip():
                    try:
                        raw_inputs = json.loads(raw_inputs)
                    except (ValueError, TypeError):
                        pass
                child_inputs = raw_inputs if isinstance(raw_inputs, dict) else {}
                timeout = _clamp_invoke_timeout(args.get("timeout"))
                target = target.strip() if isinstance(target, str) else None
                prompt = prompt if isinstance(prompt, str) and prompt.strip() else None
                # `version` pins a skill ref to a specific deployment; tolerate a
                # stringy value from the model. None / unparseable → active.
                raw_version = args.get("version")
                try:
                    version = (
                        int(raw_version)
                        if raw_version not in (None, "")
                        and not isinstance(raw_version, bool)
                        else None
                    )
                except (TypeError, ValueError):
                    version = None
                label = target or "inline prompt"
                if (target is None) == (prompt is None):
                    content = (
                        "ERROR: run_subagent requires exactly one of `target` "
                        "(skill ref or bundle `*.md` path) or `prompt`"
                    )
                elif version is not None and (target is None or target.endswith(".md")):
                    content = (
                        "ERROR: `version` only pins a declared skill `target` "
                        "(e.g. `skillpack/skill`); it can't be used with an "
                        "inline `prompt` or a bundle `*.md` path"
                    )
                elif depth >= MAX_SUBAGENT_DEPTH:
                    content = (
                        f"ERROR: subagent depth limit ({MAX_SUBAGENT_DEPTH}) reached — "
                        f"the call graph rooted at this job is too deep to extend"
                    )
                else:
                    invoke_out = await _dispatch_subagent(
                        ctx=ctx, job_id=job_id, workspace_id=workspace_id,
                        deployment=deployment, parent_skill=skill, secrets=secrets,
                        target=target, prompt=prompt, child_inputs=child_inputs,
                        timeout=timeout, depth=depth, unique_key=tu.id, label=label,
                        version=version, out_dir=out_dir, use_cache=use_cache,
                    )
                    if not invoke_out.get("ok"):
                        content = f"ERROR: {invoke_out.get('error')}"
                    elif invoke_out.get("status") != "succeeded":
                        content = (
                            f"ERROR: subagent `{label}` ended "
                            f"{invoke_out.get('status')}: "
                            f"{invoke_out.get('error') or '(no error)'}"
                        )
                    else:
                        content = json.dumps(invoke_out.get("result"), default=str)
                await ctx.emit_event("tool_result",
                    {
                        "tool_use_id": tu.id,
                        "ok": not content.startswith("ERROR:"),
                        "preview": content[:512],
                        "tool": "run_subagent",
                    },
                )
            elif tu.name in tools_by_name:
                t = tools_by_name[tu.name]
                tool_inputs = (
                    tu.input if isinstance(tu.input, dict) else {"input": tu.input}
                )
                schema = to_jsonschema(t.input_schema)
                # The model may serialize a nested object/array (or file) field
                # as a JSON string. Parse those back BEFORE validation, else a
                # stringified value yields a misleading "X is not of type
                # array/object" and the agent retries the same shape forever.
                # Only coerce fields whose declared type is object/array or a
                # file (oneOf/anyOf of string|object) — plain string fields are
                # left untouched.
                props = (schema or {}).get("properties") or {}
                if isinstance(tool_inputs, dict):
                    for _k, _ps in props.items():
                        if _k not in tool_inputs or not isinstance(_ps, dict):
                            continue
                        if (
                            _ps.get("type") in ("object", "array")
                            or "oneOf" in _ps
                            or "anyOf" in _ps
                        ):
                            tool_inputs[_k] = _coerce_json_arg(tool_inputs[_k])
                ok = True
                tool_media: list[dict] = []
                # Validate inputs against the tool's declared schema before
                # running it; surface schema errors back to the agent so it
                # can correct itself rather than crashing the job.
                try:
                    Draft202012Validator(schema).validate(
                        tool_inputs
                    )
                except ValidationError as ve:
                    path = ".".join(str(p) for p in ve.absolute_path) or "<root>"
                    content = f"ERROR: tool `{t.name}` input invalid at `{path}`: {ve.message}"
                    ok = False
                if ok and t.confirm:
                    # Human-in-the-loop confirmation gate (P1-5): a side-effectful
                    # tool the skill marked `confirm: true` pauses here for an
                    # explicit approve/deny before it runs. The gate is enforced
                    # off the deploy-time flag, so the model (or content it
                    # fetched) can never bypass it. Fail-closed: a deny/timeout
                    # becomes a tool error the agent must react to, not a silent
                    # side effect.
                    if not ctx.platform_enabled:
                        # No dashboard offline — ask the operator on the console.
                        # Fail-closed too: a non-interactive stdin denies.
                        decision, reason = await _local_confirm(
                            ctx, t.name, tool_inputs
                        )
                    else:
                        from .approvals import request_approval, wait_for_decision

                        approval_id = await request_approval(
                            session, job_id, t.name, tool_inputs
                        )
                        if approval_id is None:
                            decision, reason = "expired", "could not create approval"
                        else:
                            decision, reason = await wait_for_decision(
                                session, job_id, approval_id
                            )
                    if decision != "approved":
                        verb = {
                            "denied": "denied by a reviewer",
                            "expired": "not approved in time",
                        }.get(decision, decision)
                        content = (
                            f"ERROR: the `{t.name}` action needs human approval and "
                            f"was {verb}"
                            + (f" ({reason})" if reason else "")
                            + ". Do not retry it — continue without it or stop."
                        )
                        ok = False
                if ok and t.is_skill_tool and depth >= MAX_SUBAGENT_DEPTH:
                    content = (
                        f"ERROR: subagent depth limit ({MAX_SUBAGENT_DEPTH}) reached"
                    )
                    ok = False
                elif ok and t.is_skill_tool:
                    # Skill-tool: t.skill_ref was resolved against THIS
                    # deployment's manifest at load time, so it always runs
                    # in-process as a nested agent (parent's job_id → shared
                    # events + cost; no queued child job → no slot deadlock).
                    invoke_out = await _dispatch_subagent(
                        ctx=ctx, job_id=job_id, workspace_id=workspace_id,
                        deployment=deployment, parent_skill=skill, secrets=secrets,
                        target=t.skill_ref or "", prompt=None,
                        child_inputs=tool_inputs, timeout=s.max_agent_seconds, depth=depth,
                        unique_key=tu.id, label=t.skill_ref or t.name, out_dir=out_dir,
                        use_cache=use_cache,
                    )
                    if not invoke_out.get("ok"):
                        content = f"ERROR: {invoke_out.get('error')}"
                        ok = False
                    elif invoke_out.get("status") != "succeeded":
                        content = (
                            f"ERROR: child skill `{t.skill_ref}` ended "
                            f"{invoke_out.get('status')}: "
                            f"{invoke_out.get('error') or '(no error)'}"
                        )
                        ok = False
                    else:
                        content = json.dumps(invoke_out.get("result"), default=str)
                elif ok:
                    out = await asyncio.to_thread(
                        run_function,
                        f"{t.module}:{t.func}",
                        tool_inputs,
                        workdir,
                        deployment.root,
                        python_exe,
                        secrets,
                        str(workspace_id),
                        str(job_id),
                    )
                    if not out.get("ok"):
                        content = f"ERROR: {out.get('error')}"
                        ok = False
                    else:
                        result_value = out.get("result")
                        # Tool outputs follow the same contract as skill outputs:
                        # drop undeclared keys, then require every declared one.
                        result_value = prune_extras(t.output_schema, result_value)
                        try:
                            Draft202012Validator(
                                to_output_jsonschema(t.output_schema)
                            ).validate(result_value)
                        except ValidationError as ve:
                            path = (
                                ".".join(str(p) for p in ve.absolute_path) or "<root>"
                            )
                            content = (
                                f"ERROR: tool `{t.name}` output failed schema at "
                                f"`{path}`: {ve.message}"
                            )
                            ok = False
                        else:
                            content = json.dumps(result_value, default=str)
                            tool_media = _collect_tool_media(result_value)
                await ctx.emit_event("tool_result",
                    {
                        "tool_use_id": tu.id,
                        "ok": ok,
                        "preview": content[:512],
                        **({"media": tool_media} if tool_media else {}),
                    },
                )
            else:
                content = (
                    f"Tool '{tu.name}' is declared in the skill but no implementation "
                    f"is registered. Inputs: {tu.input}"
                )
                await ctx.emit_event("tool_result",
                    {"tool_use_id": tu.id, "ok": False, "preview": content[:512]},
                )
            await ctx.commit()
            return content, None

        # Run one tool with hard error containment: a tool's own crash must never
        # abort its siblings or fail the job — it becomes a soft tool error (the
        # model can react) and the matching tool_result still lands, so the
        # Anthropic messages contract holds. Used by both the serial and the
        # concurrent path.
        async def _safe_dispatch(tu, ctx):
            # One tool span per call (P0-3): nests under the step span, and any
            # in-proc subagent run it spawns nests under THIS span.
            async with _span(
                ctx, "tool", f"tool:{tu.name}",
                {"tool": tu.name, "tool_use_id": tu.id},
            ) as _tsp:
                try:
                    content, _so = await _dispatch_one_tool(tu, ctx)
                    _tsp.attrs["ok"] = not (
                        isinstance(content, str) and content.startswith("ERROR")
                    )
                    return content, _so
                except Exception as e:
                    content = f"ERROR: {type(e).__name__}: {e}"
                    _tsp.attrs["ok"] = False
                    # A handler may have aborted the transaction (e.g. a DB error
                    # in a tool query); roll it back so this session is usable
                    # again. Without this, the failed tx poisons the SHARED
                    # serial-path session and every subsequent query (the next
                    # is_cancelled / emit_event) raises InFailedSQLTransactionError
                    # — turning one tool's failure into a dead job. Best-effort.
                    try:
                        await ctx.session.rollback()
                    except Exception:
                        pass
                    try:
                        await ctx.emit_event("tool_result",
                            {"tool_use_id": tu.id, "ok": False, "preview": content[:512]},
                        )
                        await ctx.commit()
                    except Exception:
                        pass
                    return content, None

        # ── Run the turn's tool calls ──
        # Independent calls (everything but set_output) run concurrently — each in
        # its own task on its OWN db session (the worker session isn't
        # concurrency-safe) with its own event ctx — so subagents + media + leaf
        # tools the model requested together actually overlap. We only fan out at
        # shallow depth (see _PARALLEL_TOOL_MAX_DEPTH) so the per-task sessions
        # can't outrun the DB pool. A turn with set_output, or a deeper subagent,
        # runs serially. Results reassemble in tool_use order so messages stay
        # valid.
        # A confirm-gated tool in the turn forces the whole turn serial, so the
        # approval pause + its events stay on the main session in a clean order.
        _has_confirm = any(
            (tn := tools_by_name.get(t.name)) is not None and tn.confirm
            for t in tool_uses
        )
        if (
            depth <= _PARALLEL_TOOL_MAX_DEPTH
            and len(tool_uses) > 1
            and all(t.name not in _PARALLEL_UNSAFE for t in tool_uses)
            and not _has_confirm
        ):
            _sem = asyncio.Semaphore(_PARALLEL_TOOL_LIMIT)

            async def _run_parallel(tu):
                # Offline: there is no DB, so the per-task tools just share the
                # (DB-less) ctx — no fresh session to check out, nothing to make
                # concurrency-unsafe. This also keeps the DB stack off the local
                # import path (`db` is imported lazily only on the hosted branch).
                if not ctx.platform_enabled:
                    async with _sem:
                        return await _safe_dispatch(tu, ctx)
                from .db import session as db_session

                # The session CHECKOUT is inside the try too: under pool pressure
                # db_session() can raise a pool timeout, which must degrade to a
                # tool error rather than crash the batch (and thus the job).
                async with _sem:
                    try:
                        async with db_session() as _sess:
                            # Each concurrent tool runs on its OWN ctx, bound to a
                            # fresh session (the worker session isn't concurrency-
                            # safe); with_session rebinds while keeping job/ws ids.
                            return await _safe_dispatch(tu, ctx.with_session(_sess))
                    except Exception as e:
                        return (f"ERROR: {type(e).__name__}: {e}", None)

            _results = await asyncio.gather(
                *[_run_parallel(tu) for tu in tool_uses],
                return_exceptions=True,
            )
            # Belt-and-suspenders: gather should never surface an exception (every
            # task returns a tuple), but map any stray one to a soft error so the
            # zip below always gets (content, set_output) pairs.
            _results = [
                r
                if not isinstance(r, BaseException)
                else (f"ERROR: {type(r).__name__}: {r}", None)
                for r in _results
            ]
        else:
            _results = []
            for tu in tool_uses:
                _results.append(await _safe_dispatch(tu, ctx))

        for tu, (content, _so) in zip(tool_uses, _results):
            # Offload oversized results to the drive before they enter history,
            # so the agent doesn't re-read a full fetched page / long stdout on
            # every subsequent turn (token economy / P1). No-op for small or
            # multimodal results; cache-safe because it's append-only.
            content = await asyncio.to_thread(
                _offload_tool_result, tu.name, tu.id, content, job_id, workspace_id
            )
            tool_results.append(
                {"type": "tool_result", "tool_use_id": tu.id, "content": content}
            )
            if _so is not None:
                # set_output fired this turn → record it and end the run.
                structured_output = _so
                early_return = {"output": structured_output, "steps": step + 1}

        if early_return is not None:
            await _close_step()
            await _close_run()
            return early_return
        messages.append({"role": "user", "content": tool_results})

        # Checkpoint at this CLEAN turn boundary (top-level runs only): `messages`
        # now ends at a user(tool_results) turn, so a resume from `step + 1` is a
        # valid next model call. If the worker dies before the next checkpoint,
        # only this one in-flight step re-runs. Best-effort — never blocks the run.
        if depth == 1:
            await ctx.save_checkpoint(
                step=step + 1,
                messages=messages,
                final_text_parts=final_text_parts,
                structured_output=structured_output,
            )
            await ctx.commit()

        # Tools ran and the run continues — close this step's span (the run span
        # stays open across iterations).
        await _close_step()

    await _close_run()
    raise RuntimeError(f"agent exceeded {s.max_agent_steps} steps without finishing")
