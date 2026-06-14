"""Local API server — `puras serve` (open-core).

Serves a local HTTP API that mirrors the hosted Puras *job* contract, backed by
the offline runner (`run_local`). Point any Puras SDK at the server's base URL
(set `apiBase` / `PURAS_API_BASE` to `http://localhost:<port>`) and build your
whole integration offline, on your own LLM key, with NO puras.co account — then
flip the base URL to `https://api.puras.co` and `puras deploy` to ship the
identical code.

Where `puras run --local` answers "does my skill work?", `puras serve` answers
"does my *app's* integration work?": your application talks HTTP to localhost
exactly as it will to production.

Endpoints (the SDK-driving subset of the hosted `/v1/jobs` surface):

    POST   /v1/jobs                 submit a job (?wait=true&timeout=N supported)
    GET    /v1/jobs                 list jobs in this server
    GET    /v1/jobs/{id}            fetch one job
    GET    /v1/jobs/{id}/events     job events (?after=<id> for the tail)
    GET    /v1/jobs/{id}/spans      OTel-style trace spans
    GET    /health                  liveness + the served skills

Zero extra dependencies: stdlib `http.server`, threaded, an in-memory job store,
and client-side polling — the default transport of the Python and React-Native
SDKs, so they work fully. Live SSE (`/stream`) is intentionally not implemented
yet; it can land later behind a `puras-runner[serve]` extra.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.parse
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from . import local_run

# A stable synthetic workspace — there is no real tenant offline. Matches the id
# the offline runner uses so a job's drive lines up with `puras run --local`.
_LOCAL_WORKSPACE_ID = "00000000-0000-0000-0000-000000000000"
_TERMINAL = ("succeeded", "failed", "cancelled")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _skill_name(raw: str | None) -> str | None:
    """A skill may arrive fully-qualified (`workspace/skillpack/skill`) like the
    SDK sends it; locally we serve a single bundle, so take the last segment."""
    if not raw:
        return None
    parts = [p for p in str(raw).strip().strip("/").split("/") if p]
    return parts[-1] if parts else None


class _BadRequest(ValueError):
    """A malformed request — surfaced to the client as a 400."""


class LocalServer:
    """In-memory, thread-safe job store + HTTP runtime for one served bundle.

    Construct it with the bundle dir, then `serve_forever(host, port)` (blocking)
    — or `make_server(host, port)` to drive the lifecycle yourself (tests)."""

    def __init__(
        self,
        bundle_dir: str | Path,
        *,
        model: str | None = None,
        api_key: str | None = None,
        require_key: str | None = None,
        on_log: Callable[[str], None] | None = None,
    ):
        self.bundle_dir = str(Path(bundle_dir).expanduser().resolve())
        self.model = model
        self.api_key = api_key  # the user's own LLM key (BYO); may be None → env
        self.require_key = require_key  # optional emulated client Bearer token
        self.on_log = on_log
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._events: dict[str, list[dict]] = {}
        self._spans: dict[str, list[dict]] = {}

    # ── discovery ──────────────────────────────────────────────────────────────
    def discover_skills(self) -> list[str]:
        """Top-level skill names in the served bundle. Raises on a bad bundle."""
        from .manifest import parse_bundle_dir

        manifest = parse_bundle_dir(Path(self.bundle_dir))
        return [s.name for s in manifest.skills if s.parent_skill is None]

    # ── store ──────────────────────────────────────────────────────────────────
    def authorized(self, auth_header: str | None) -> bool:
        if not self.require_key:
            return True
        return (auth_header or "") == f"Bearer {self.require_key}"

    def health(self) -> dict:
        try:
            skills = self.discover_skills()
            err = None
        except Exception as e:  # noqa: BLE001 — report, don't crash liveness
            skills, err = [], f"{type(e).__name__}: {e}"
        out = {"ok": True, "service": "puras-local", "bundle_dir": self.bundle_dir, "skills": skills}
        if err:
            out["bundle_error"] = err
        return out

    def submit(self, skill: str | None, inputs: dict, *, wait: bool, timeout: float) -> dict:
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "workspace_id": _LOCAL_WORKSPACE_ID,
            "skillpack_id": None,
            "deployment_id": None,
            "parent_job_id": None,
            "type": "agentic",  # the local runner only drives the agent loop
            "skill_name": skill or "",
            "status": "queued",
            "inputs": inputs,
            "result": None,
            "error": None,
            "outputs": None,
            "cost_micros": 0,
            "steps": None,
            "usage": None,
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._events[job_id] = []
            self._spans[job_id] = []
        threading.Thread(
            target=self._run_job, args=(job_id, skill, inputs), daemon=True
        ).start()

        if wait and timeout > 0:
            # Bound the server-side wait (the SDK also polls client-side for long
            # jobs); poll our own store until the run reaches a terminal status.
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                with self._lock:
                    if self._jobs[job_id]["status"] in _TERMINAL:
                        break
                time.sleep(0.05)
        return self.get_job(job_id)  # type: ignore[return-value]

    def _run_job(self, job_id: str, skill: str | None, inputs: dict) -> None:
        self._update(job_id, status="running", started_at=_now())
        counter = {"n": 0}

        def on_event(event_type: str, payload: dict) -> None:
            with self._lock:
                counter["n"] += 1
                self._events[job_id].append(
                    {
                        "id": counter["n"],
                        "job_id": job_id,
                        "ts": _now(),
                        "type": event_type,
                        "payload": payload,
                    }
                )

        try:
            # Attribute lookup (not a bound import) so tests can monkeypatch
            # `worker.local_run.run_local`.
            res = local_run.run_local(
                self.bundle_dir,
                inputs,
                skill=skill,
                model=self.model,
                api_key=self.api_key,
                on_event=on_event,
            )
            usage = res.get("usage") or {}
            with self._lock:
                self._spans[job_id] = res.get("spans") or []
            self._update(
                job_id,
                status="succeeded",
                result=res.get("output"),
                steps=res.get("steps"),
                usage=usage,
                cost_micros=int(usage.get("cost_micros") or 0),
                finished_at=_now(),
            )
        except local_run.LocalRunError as e:
            self._fail(job_id, str(e), on_event)
        except Exception as e:  # noqa: BLE001 — a failed run is data, not a crash
            self._fail(job_id, f"{type(e).__name__}: {e}", on_event)

    def _fail(self, job_id: str, message: str, on_event: Callable[[str, dict], None]) -> None:
        on_event("error", {"message": message})
        self._update(job_id, status="failed", error=message, finished_at=_now())

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(fields)

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [dict(j) for j in self._jobs.values()]

    def get_events(self, job_id: str, after: int = 0) -> list[dict] | None:
        with self._lock:
            if job_id not in self._jobs:
                return None
            return [dict(e) for e in self._events[job_id] if e["id"] > after]

    def get_spans(self, job_id: str) -> list[dict] | None:
        with self._lock:
            if job_id not in self._jobs:
                return None
            return [dict(s) for s in self._spans[job_id]]

    def _log(self, msg: str) -> None:
        if self.on_log:
            self.on_log(msg)

    # ── runtime ────────────────────────────────────────────────────────────────
    def make_server(self, host: str, port: int) -> ThreadingHTTPServer:
        httpd = ThreadingHTTPServer((host, port), _Handler)
        httpd.app = self  # type: ignore[attr-defined]
        return httpd

    def serve_forever(self, host: str, port: int) -> None:
        httpd = self.make_server(host, port)
        try:
            httpd.serve_forever()
        finally:
            httpd.server_close()


class _Handler(BaseHTTPRequestHandler):
    server_version = "puras-local/1"

    @property
    def app(self) -> LocalServer:
        return self.server.app  # type: ignore[attr-defined]

    # Quiet the default stderr access log; the app logs tidily via on_log.
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002
        return

    # ── helpers ────────────────────────────────────────────────────────────────
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")

    def _json(self, status: int, obj: Any) -> None:
        body = json.dumps(obj, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)
        mp = getattr(self, "_mp", None)
        if mp:
            self.app._log(f"{mp[0]} {mp[1]} -> {status}")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except ValueError as e:
            raise _BadRequest(f"body is not valid JSON: {e}") from None
        if not isinstance(data, dict):
            raise _BadRequest("request body must be a JSON object")
        return data

    # ── verbs ──────────────────────────────────────────────────────────────────
    def do_OPTIONS(self) -> None:  # noqa: N802 — CORS preflight
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        self._route("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._route("POST")

    def _route(self, method: str) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        qs = urllib.parse.parse_qs(parsed.query)
        self._mp = (method, path)

        if path.startswith("/v1/") and not self.app.authorized(self.headers.get("Authorization")):
            return self._json(401, {"detail": "missing or invalid API key"})

        try:
            if method == "GET" and path in ("/", "/health"):
                return self._json(200, self.app.health())

            if method == "POST" and path == "/v1/jobs":
                body = self._read_json()
                inputs = body.get("inputs") or {}
                if not isinstance(inputs, dict):
                    raise _BadRequest("`inputs` must be a JSON object")
                skill = _skill_name(body.get("skill"))
                wait = (qs.get("wait", ["false"])[0]).lower() in ("1", "true", "yes")
                try:
                    timeout = float(qs.get("timeout", ["60"])[0])
                except ValueError:
                    raise _BadRequest("`timeout` must be a number") from None
                job = self.app.submit(skill, inputs, wait=wait, timeout=timeout)
                return self._json(201, job)

            if method == "GET" and path == "/v1/jobs":
                return self._json(200, self.app.list_jobs())

            m = re.match(r"^/v1/jobs/([^/]+)$", path)
            if method == "GET" and m:
                job = self.app.get_job(m.group(1))
                return self._json(200, job) if job else self._json(404, {"detail": "job not found"})

            m = re.match(r"^/v1/jobs/([^/]+)/events$", path)
            if method == "GET" and m:
                try:
                    after = int(qs.get("after", ["0"])[0])
                except ValueError:
                    raise _BadRequest("`after` must be an integer") from None
                evs = self.app.get_events(m.group(1), after)
                return self._json(200, evs) if evs is not None else self._json(404, {"detail": "job not found"})

            m = re.match(r"^/v1/jobs/([^/]+)/spans$", path)
            if method == "GET" and m:
                spans = self.app.get_spans(m.group(1))
                return self._json(200, spans) if spans is not None else self._json(404, {"detail": "job not found"})

            return self._json(404, {"detail": f"no route for {method} {path}"})
        except _BadRequest as e:
            return self._json(400, {"detail": str(e)})
        except Exception as e:  # noqa: BLE001 — never leak a traceback to the client
            return self._json(500, {"detail": f"{type(e).__name__}: {e}"})


def serve(
    bundle_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 8787,
    *,
    model: str | None = None,
    api_key: str | None = None,
    require_key: str | None = None,
    on_log: Callable[[str], None] | None = None,
) -> None:
    """Build a `LocalServer` and serve it (blocking)."""
    LocalServer(
        bundle_dir, model=model, api_key=api_key, require_key=require_key, on_log=on_log
    ).serve_forever(host, port)
