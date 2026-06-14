"""External client — call your deployed Puras skills from any app.

This is the counterpart to the in-skill runtime SDK: instead of running *inside*
a job, it submits jobs to a skillpack you've deployed and hands back the result.

Address a skill the same way you see it on its page — a `workspace/skillpack/skill`
path you can copy straight from the playground:

    import puras

    client = puras.Client()   # PURAS_API_KEY from env
    ad = client.run("acme/ugc-ads/ugc-ad", {"product": product_url, "duration": 15})
    print(ad["video"])

For your own skills you can drop the workspace (`skillpack/skill`), or set a
default skillpack once and call skills bare:

    client = puras.Client(skillpack="ugc-ads")   # slug or UUID
    ad = client.run("ugc-ad", {"product": product_url, "duration": 15})

Auth is a workspace API key: pass `api_key=` or set the `PURAS_API_KEY` env
var. Use a secret key (`puras_live_…`) server-side; a publishable key
(`puras_pub_…`, job submit + own-job reads only) also works if the code ships
to untrusted environments. It wraps `POST /v1/jobs?skillpack=…` — exactly
what the `puras run` CLI and a raw curl do.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

import httpx

DEFAULT_API_BASE = "https://api.puras.co"
_TERMINAL = ("succeeded", "failed", "cancelled")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class PurasAPIError(RuntimeError):
    """Non-2xx response from the Puras API."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class JobError(RuntimeError):
    """A job finished in a non-succeeded state (failed/cancelled/timeout)."""

    def __init__(self, job: dict):
        self.job = job
        status = job.get("status")
        super().__init__(f"job {job.get('id')} {status}: {job.get('error') or ''}".rstrip())


def _detail(r: httpx.Response) -> str:
    try:
        j = r.json()
        if isinstance(j, dict):
            return str(j.get("detail") or j.get("message") or j)
        return str(j)
    except ValueError:
        return r.text or r.reason_phrase


class Client:
    """Call deployed skills.

    Pass a fully-qualified `workspace/skillpack/skill` path to `run`/`submit`
    (copyable from a skill's page), or set a default `skillpack` once — a slug
    or UUID — and call skills by bare name."""

    def __init__(
        self,
        api_key: str | None = None,
        skillpack: str | None = None,
        *,
        api_base: str | None = None,
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.environ.get("PURAS_API_KEY")
        if not self.api_key:
            raise ValueError("no API key — pass api_key=… or set PURAS_API_KEY")
        self.skillpack = skillpack
        self.api_base = (
            api_base or os.environ.get("PURAS_API_BASE") or DEFAULT_API_BASE
        ).rstrip("/")
        self.timeout = timeout

    # ── public API ───────────────────────────────────────────────────────────
    def submit(
        self,
        skill: str,
        inputs: dict | None = None,
        *,
        skillpack: str | None = None,
        version: int | None = None,
        wait: bool = False,
        timeout: int = 60,
    ) -> dict:
        """Submit a job. `wait=True` blocks server-side up to `timeout` seconds
        (the API caps the server-side wait at 60 — short jobs only; for longer
        runs use `run()` or `wait()`, which poll client-side).
        Returns the job object (`{id, status, result, error, …}`).

        `skill` may be a fully-qualified path —
        `"workspace/skillpack/skill"` (copyable from the skill's page) or
        `"skillpack/skill"` for one of your own — in which case the skillpack
        is taken from the path. A bare skill name uses the default `skillpack`
        passed here or to `Client(...)`.

        `version` pins the run to a specific deployment version of the
        skillpack (e.g. `version=3`); omit it to use the active deployment,
        which follows new deploys. Pinning lets you keep running a release
        you've validated even after newer versions are deployed."""
        ref, skill_name = self._target(skill, skillpack)
        params: dict[str, Any] = {}
        # A UUID goes on the legacy `skillpack_id` param; a slug path on
        # `skillpack`. Either resolves server-side, but keeping UUIDs on the
        # old param means an older API still understands them.
        params["skillpack_id" if _UUID_RE.match(ref) else "skillpack"] = ref
        if version is not None:
            params["version"] = str(int(version))
        if wait:
            params["wait"] = "true"
            params["timeout"] = str(int(timeout))
        http_timeout = (timeout + 30) if wait else self.timeout
        return self._request(
            "POST",
            "/v1/jobs",
            params=params,
            json_body={"skill": skill_name, "inputs": inputs or {}},
            timeout=http_timeout,
        )

    def run(
        self,
        skill: str,
        inputs: dict | None = None,
        *,
        skillpack: str | None = None,
        version: int | None = None,
        timeout: float = 600,
        poll_interval: float = 2.0,
    ) -> dict:
        """Submit + wait, returning the skill's `result`. Raises `JobError` if it
        didn't succeed (including still-running when `timeout` elapses).

        Waiting is client-side polling (`poll_interval` seconds between
        checks), so multi-minute media jobs work; `timeout` defaults to 10
        minutes.

        `skill` may be a `"workspace/skillpack/skill"` path (copyable from the
        skill's page) or a bare name resolved against the default skillpack.

        Pass `version=N` to pin the run to a specific deployment version of the
        skillpack; omit it to use the active deployment."""
        job = self.submit(skill, inputs, skillpack=skillpack, version=version)
        job = self.wait(job["id"], timeout=timeout, poll_interval=poll_interval)
        if job.get("status") != "succeeded":
            raise JobError(job)
        return job.get("result") or {}

    def get(self, job_id: str) -> dict:
        """Fetch a job by id."""
        return self._request("GET", f"/v1/jobs/{job_id}", timeout=self.timeout)

    def wait(
        self,
        job_id: str,
        *,
        timeout: float = 600,
        poll_interval: float = 2.0,
    ) -> dict:
        """Poll a job until it reaches a terminal status (succeeded / failed /
        cancelled) or `timeout` seconds elapse. Returns the last job object —
        check `status`; a job still running at timeout is returned as-is."""
        deadline = time.monotonic() + timeout
        job = self.get(job_id)
        while job.get("status") not in _TERMINAL:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(poll_interval, remaining))
            job = self.get(job_id)
        return job

    # ── internals ────────────────────────────────────────────────────────────
    def _target(self, skill: str, skillpack: str | None) -> tuple[str, str]:
        """Split a call into `(skillpack_ref, skill_name)`.

        A qualified `skill` — `workspace/skillpack/skill` or `skillpack/skill`
        — carries its own skillpack (everything before the last segment); the
        default `skillpack` is ignored. A bare skill name needs a default
        skillpack from `skillpack=` here or on `Client(...)`."""
        s = (skill or "").strip().strip("/")
        parts = [p for p in s.split("/") if p]
        if len(parts) >= 2:
            return "/".join(parts[:-1]), parts[-1]
        sp = skillpack or self.skillpack
        if not sp:
            raise ValueError(
                "no skillpack — call a fully-qualified skill "
                '("workspace/skillpack/skill"), or set skillpack=… here or on '
                "Client(...)"
            )
        return sp, (parts[0] if parts else s)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: Any = None,
        timeout: float = 120.0,
    ) -> dict:
        with httpx.Client(timeout=timeout, follow_redirects=True) as c:
            r = c.request(
                method,
                f"{self.api_base}{path}",
                params=params,
                json=json_body,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        if r.status_code >= 400:
            raise PurasAPIError(r.status_code, _detail(r))
        return r.json()
