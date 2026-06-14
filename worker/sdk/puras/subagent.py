"""puras.subagent — run an isolated subagent from inside a running job.

A *subagent run* is an isolated agent run — its own job, own context window,
own `cost_micros` row, linked back to the caller via `jobs.parent_job_id`.
Think of it like Claude Code's subagents: hand a self-contained stage of work
to a fresh agent instead of doing everything in the current context. Use it to
build pipelines — one deterministic Python skill orchestrates calls into
agentic or deterministic subagents (research + render, plan + execute, N
parallel renders, …).

A subagent's `target` can be one of three things:

  - a **skill** in the same skillpack, another skillpack, or a public skillpack
    in another workspace:
      - `"skill_name"` — a skill in the caller's own skillpack.
      - `"skillpack_slug/skill_name"` — a skill in another skillpack the
        caller workspace has access to.
      - `"workspace_slug/skillpack_slug/skill_name"` — fully qualified, for a
        public skillpack in another workspace.
  - a **bundle markdown file** — `"references/foo.md"` (any bundle-relative
    `*.md` path): run that file as the system prompt of a fresh subagent in
    the caller's own skillpack. No manifest entry or schema needed.
  - an **inline prompt** — pass `prompt="…"` instead of a `target`: run a raw
    prompt string as an isolated subagent in the caller's bundle context, with
    the built-in tools (bash, media, web, file_read, run_subagent, …) and a
    free-form `set_output`. No file or manifest entry needed.

The child runs as a real job in the queue. Cost accrues to the same workspace
wallet the parent is billing to. The platform caps the call-graph depth and
refuses cycles.

Example:

    from puras import subagent

    # Run a skill as a subagent.
    research = subagent.run(
        "creative-research",
        {"brief": brief, "product_image": product_image},
    )

    # Run a bundle prompt file as a subagent.
    storyboard = subagent.run(
        "references/storyboard-director.md",
        {"brief": research},
    )

    # Run a one-off inline prompt as a subagent.
    summary = subagent.run(
        prompt="You are a copy editor. Tighten the script below and "
               "call set_output with {\"script\": <edited>}.",
        inputs={"script": video["script"]},
    )

    return {"video_url": research["video_url"]}

Returns the child's `result` value (whatever its `output_schema` declares, or
whatever an ad-hoc / inline subagent passed to `set_output`). On failure,
raises `SubagentRunError` with the child job_id and the error message — so the
caller can surface a useful diagnostic without re-querying the API.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


class SubagentRunError(RuntimeError):
    """Raised when a subagent run ended in `failed` or `cancelled` status,
    or did not reach a terminal status within `timeout` seconds.

    Attributes:
        job_id: The child job's id, useful for fetching events / logs.
        status: One of `failed`, `cancelled`, `queued`, `running`.
        message: Human-readable error from the child (or a timeout note).
    """

    def __init__(self, job_id: str, status: str, message: str) -> None:
        super().__init__(f"subagent run {status}: {message} (job_id={job_id})")
        self.job_id = job_id
        self.status = status
        self.message = message


def _env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(
            f"missing {name} env var — `puras.subagent.run` only works inside a "
            f"deployed function/agent; the worker injects this. (Running outside? "
            f"You need to set it manually.)"
        )
    return v


def run(
    target: str | None = None,
    inputs: dict[str, Any] | None = None,
    *,
    prompt: str | None = None,
    version: int | None = None,
    timeout: int = 600,
) -> Any:
    """Run a subagent synchronously and return its output.

    Pass exactly one of `target` or `prompt`.

    Args:
      target: what to run —
        - `"references/foo.md"` (any bundle-relative `*.md` path) — run that
          markdown file as an isolated subagent in the caller's own skillpack.
          No manifest entry or schema needed; the path is relative to the
          caller's skill directory. Inputs are passed verbatim and any
          file-shaped values (URLs / drive paths) are staged for the child.
        - `"skill_name"` (same skillpack),
        - `"skillpack_slug/skill_name"` (another skillpack in the caller's
          workspace), or
        - `"workspace_slug/skillpack_slug/skill_name"` (a public skillpack in
          another workspace).
      inputs: dict passed straight to the subagent. For a declared skill it is
        validated against the skill's `input_schema` before the job is queued;
        for a `.md` or inline-prompt subagent it is passed verbatim.
      prompt: an inline system prompt to run as a one-off subagent (mutually
        exclusive with `target`). The subagent runs in the caller's bundle
        context with the built-in tools and a free-form `set_output`.
      version: pin a declared skill `target` to a specific deployment version
        of its skillpack (e.g. 3). Omit to use the active deployment. Only
        valid for a skill-ref `target` — passing it with an inline `prompt` or
        a `*.md` path is an error (those always run against the caller's own
        deployment).
      timeout: max seconds to wait for the child to reach a terminal state.
        The platform also enforces its own per-job timeouts; this is just
        how long the calling side will block.

    Returns:
      The child's `result` value. For deterministic skills that's whatever
      their `:run` function returned. For agentic / ad-hoc / inline subagents
      it's whatever was passed to `set_output`.

    Raises:
      SubagentRunError: child failed, was cancelled, or didn't finish in time.
      ValueError: neither or both of `target` / `prompt` were given.
    """
    if (target is None) == (prompt is None):
        raise ValueError(
            "puras.subagent.run requires exactly one of `target` (a skill ref "
            "or a bundle `*.md` path) or `prompt` (an inline system prompt)"
        )
    if version is not None and (target is None or target.strip().endswith(".md")):
        raise ValueError(
            "puras.subagent.run: `version` only pins a declared skill `target` "
            '(e.g. "skillpack/skill"); it cannot be used with an inline `prompt` '
            "or a bundle `*.md` path, which run against the caller's own deployment"
        )
    api_base = _env("PURAS_API_BASE").rstrip("/")
    token = _env("PURAS_SERVICE_TOKEN")
    parent_job_id = _env("PURAS_JOB_ID")
    body = {
        "parent_job_id": parent_job_id,
        "target": target,
        "prompt": prompt,
        "inputs": inputs or {},
        "version": version,
        "timeout": timeout,
    }
    # Generous client-side timeout so the server-side wait can run to
    # completion without httpx tripping first. +30s headroom for queue
    # latency, polling jitter, and the response itself.
    r = httpx.post(
        f"{api_base}/v1/subagent/invoke",
        headers={"X-Puras-Service-Token": token, "Content-Type": "application/json"},
        json=body,
        timeout=timeout + 30,
    )
    if not r.is_success:
        raise SubagentRunError(
            job_id="",
            status="failed",
            message=f"subagent/invoke {r.status_code}: {r.text[:500]}",
        )
    payload = r.json()
    status_ = payload.get("status")
    job_id = payload.get("job_id", "")
    if status_ == "succeeded":
        # API has already unwrapped the worker's storage envelope, so this
        # value matches the child subagent's `output_schema` directly.
        return payload.get("result")
    raise SubagentRunError(
        job_id=job_id,
        status=status_ or "unknown",
        message=payload.get("error") or f"child did not succeed (status={status_})",
    )
