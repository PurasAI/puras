"""Headless-browser rendering for the worker, backed by Playwright (Chromium).

Two callers depend on this:

  - `web_fetch` with `render_js: true` — fetch a page, let its JavaScript run,
    then return the *rendered* visible text. Plain `web_fetch` (no JS) still goes
    through the API's httpx + BeautifulSoup path; this is only for client-side
    SPAs that come back empty otherwise.
  - `web_screenshot` — render a URL *or* a single-file HTML document from the
    workspace drive and capture a PNG. The playable-ad skill uses this to both
    runtime-validate the built game (does the canvas actually draw? any console
    errors?) and to produce its preview frame from the real game rather than an
    AI mock.

Everything runs through the synchronous Playwright API so the agent loop can
call it from `asyncio.to_thread` like the other blocking helpers. Playwright is
an optional dependency: if the package or its Chromium build is missing,
`render()` returns `{ok: False, error: ...}` instead of raising, so a worker
image built without it degrades gracefully (plain fetch keeps working).
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import structlog

log = structlog.get_logger()

# Chromium in a container has no usable sandbox and a tiny /dev/shm; these flags
# are the standard headless-in-Docker incantation.
_LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
]

# Visible text we hand back to the model is bounded — a rendered page can be
# huge, and the caller truncates again to its own max_chars anyway.
_MAX_TEXT_CHARS = 200_000

# How many console / page errors to surface. A broken playable usually trips the
# same error every frame; a handful is enough to diagnose.
_MAX_CONSOLE_ERRORS = 20


def playwright_available() -> bool:
    """True if Playwright is importable. Does NOT verify a browser is installed —
    `render()` surfaces a missing-browser launch failure with guidance."""
    try:
        import playwright.sync_api  # noqa: F401
    except Exception:
        return False
    return True


def _to_target_url(url: str | None, file_path: str | None) -> tuple[str | None, str | None]:
    """Resolve the (url, error) Playwright should navigate to.

    Exactly one of `url` / `file_path` must be set. A local file becomes a
    `file://` URL so the same render path serves both."""
    if bool(url) == bool(file_path):
        return None, "provide exactly one of 'url' or 'file_path'"
    if url:
        if not url.startswith(("http://", "https://")):
            return None, "url must start with http:// or https://"
        return url, None
    p = Path(file_path).resolve()
    if not p.is_file():
        return None, f"file not found: {file_path}"
    return p.as_uri(), None


def render(
    *,
    url: str | None = None,
    file_path: str | None = None,
    viewport_width: int = 1280,
    viewport_height: int = 800,
    full_page: bool = False,
    wait_ms: int = 1200,
    wait_until: str = "load",
    timeout_ms: int = 30_000,
    screenshot: bool = False,
    device_scale: float = 1.5,
    scroll_y: int = 0,
) -> dict:
    """Render a URL or local HTML file in headless Chromium.

    Returns a dict that is always shaped the same way:
        ok:             bool
        error:          str | None
        final_url:      str            (after redirects)
        title:          str
        text:           str            (rendered visible body text, truncated)
        console_errors: list[str]      (console.error + uncaught page errors)
        screenshot_png: bytes | None   (only when screenshot=True)

    Never raises for an expected failure (missing Playwright, nav timeout, bad
    target) — those come back as `ok: False` with a message the agent can read.
    """
    target, err = _to_target_url(url, file_path)
    if err:
        return {"ok": False, "error": err}

    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except Exception:
        return {
            "ok": False,
            "error": (
                "playwright is not installed in this worker — JS rendering and "
                "screenshots are unavailable (plain web_fetch still works)"
            ),
        }

    # Clamp the obvious knobs so a bad arg can't wedge a slot.
    viewport_width = max(64, min(3840, int(viewport_width)))
    viewport_height = max(64, min(3840, int(viewport_height)))
    wait_ms = max(0, min(15_000, int(wait_ms)))
    timeout_ms = max(1_000, min(120_000, int(timeout_ms)))

    console_errors: list[str] = []

    def _note_error(msg: str) -> None:
        if len(console_errors) < _MAX_CONSOLE_ERRORS:
            console_errors.append(msg[:500])

    try:
        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(args=_LAUNCH_ARGS, headless=True)
            except PlaywrightError as e:
                return {
                    "ok": False,
                    "error": (
                        f"could not launch Chromium ({e}); the browser binary may "
                        f"be missing — run `playwright install chromium`"
                    ),
                }
            try:
                page = browser.new_page(
                    viewport={"width": viewport_width, "height": viewport_height},
                    device_scale_factor=device_scale,
                )
                page.on(
                    "console",
                    lambda m: _note_error(f"console.{m.type}: {m.text}")
                    if m.type in ("error", "warning")
                    else None,
                )
                page.on("pageerror", lambda e: _note_error(f"pageerror: {e}"))

                try:
                    # Default wait_until is "load", NOT "networkidle": a playable
                    # game is an endless requestAnimationFrame loop (and its mraid.js
                    # 404-retries), so the network NEVER idles — "networkidle" would
                    # burn the full timeout on EVERY screenshot. That long in-Playwright
                    # stall in the to_thread worker starved the main loop's heartbeat
                    # task (sync Playwright holds the GIL), so the API reaper killed the
                    # still-alive job. "load" + an explicit wait_ms settles fast instead.
                    page.goto(target, wait_until=wait_until, timeout=timeout_ms)
                except PlaywrightError as e:
                    # A nav timeout is common and benign for pages with persistent
                    # connections (game loops, analytics beacons). Fall through and
                    # capture whatever rendered.
                    _note_error(f"navigation: {e}")

                if wait_ms:
                    page.wait_for_timeout(wait_ms)

                # Capture a lower section of a long page as a single viewport shot
                # (preferred over a giant full_page image that blows the model's
                # per-side pixel limit).
                if scroll_y and not full_page:
                    try:
                        page.evaluate(f"window.scrollTo(0, {max(0, int(scroll_y))})")
                        page.wait_for_timeout(250)
                    except PlaywrightError:
                        pass

                final_url = page.url
                title = page.title() or ""
                try:
                    text = page.inner_text("body")
                except PlaywrightError:
                    text = ""
                if len(text) > _MAX_TEXT_CHARS:
                    text = text[:_MAX_TEXT_CHARS] + "\n…[truncated]"

                png: bytes | None = None
                if screenshot:
                    png = page.screenshot(type="png", full_page=full_page)

                return {
                    "ok": True,
                    "error": None,
                    "final_url": final_url,
                    "title": title.strip(),
                    "text": text,
                    "console_errors": console_errors,
                    "screenshot_png": png,
                }
            finally:
                browser.close()
    except Exception as e:  # pragma: no cover - defensive catch-all
        log.warning("browser_render_failed", target=target, error=str(e))
        return {"ok": False, "error": f"render failed: {e}"}
