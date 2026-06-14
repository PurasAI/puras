"""Tiny HTTP health server for the worker.

The worker is a background process; without an HTTP surface the host
platform (Fly, Docker, k8s) can't tell if it's alive — silent crashes
just leave the queue stuck. This serves two endpoints on `HEALTH_PORT`
(default 8080):

  GET /health  → 200 if the worker loop is running and the DB is reachable
                 within the heartbeat window, else 503.
  GET /        → same as /health (for fly's default check path).

The server runs in a daemon thread so it dies with the process. The
worker loop calls `record_heartbeat()` once per poll iteration; the
health handler checks that the last heartbeat is within the staleness
window before reporting healthy.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_state_lock = threading.Lock()
_last_heartbeat_at: float = 0.0
_started_at: float = time.monotonic()
_staleness_window_s: float = 60.0


def record_heartbeat() -> None:
    """Mark the worker loop alive. Called every iteration of the poll loop."""
    global _last_heartbeat_at
    with _state_lock:
        _last_heartbeat_at = time.monotonic()


def configure(staleness_window_s: float) -> None:
    """Override the default 60s staleness window."""
    global _staleness_window_s
    _staleness_window_s = staleness_window_s


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args, **_kwargs) -> None:
        # Silence default per-request stderr logging; fly only needs status.
        pass

    def do_GET(self) -> None:
        if self.path not in ("/", "/health"):
            self.send_response(404)
            self.end_headers()
            return
        with _state_lock:
            last = _last_heartbeat_at
            window = _staleness_window_s
        now = time.monotonic()
        age = now - last if last > 0 else None
        # Healthy if we got at least one heartbeat AND it's within window.
        # During the first few seconds (before the first poll completes) we
        # still treat the process as healthy so fly doesn't flap on startup.
        startup_grace_s = 30.0
        uptime = now - _started_at
        if last == 0.0 and uptime < startup_grace_s:
            ok = True
        else:
            ok = last > 0 and age is not None and age <= window
        status = 200 if ok else 503
        body = json.dumps(
            {
                "ok": ok,
                "uptime_s": round(uptime, 1),
                "last_heartbeat_age_s": round(age, 1) if age is not None else None,
                "staleness_window_s": window,
            }
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start(port: int = 8080) -> None:
    """Spin up the health HTTP server in a daemon thread."""

    def _run() -> None:
        server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
        server.serve_forever()

    t = threading.Thread(target=_run, name="worker-health", daemon=True)
    t.start()
