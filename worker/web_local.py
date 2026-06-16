"""Local web tools — search and fetch without the platform (open-core, BYO key).

Hosted web search/fetch go through the platform's /v1/web service. A local run
(`puras run --local` / `puras serve`) has no platform, so this module backs the
two web verbs the local runner offers directly on the user's own machine + key:

  - `fetch(url, max_chars)` — a plain Python HTTP GET (httpx) with the HTML
    reduced to readable text (scripts/styles stripped) by the stdlib parser, no
    JavaScript. Same as the hosted plain `web_fetch`; the `render_js=true` path
    renders in the local headless browser instead (agent_runner._run_web_fetch_js).
    No key, no platform.
  - `search(query, max_results)` — runs Anthropic's server-side `web_search` tool
    on the BYO ANTHROPIC_API_KEY (the same key the local LLM loop uses) and
    returns the result list. There is no local search engine, so "web search with
    Anthropic" is the offline path; it needs the key.

Both return the SAME shape the hosted `/v1/web/{search,fetch}` endpoints do (a
`results` list / a `{url,title,content,length}` page), so `agent_runner` surfaces
them to the model identically. Problems raise `LocalWebError`, which the
dispatcher turns into a soft tool error the agent can react to.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any

from .config import get_settings


class LocalWebError(RuntimeError):
    """A local web search/fetch could not be set up or completed."""


# ---------------------------------------------------------------------------
# web_fetch — direct HTTP GET + HTML→text (no JS; render_js uses the browser).
# ---------------------------------------------------------------------------
_DEFAULT_UA = "Mozilla/5.0 (compatible; PurasLocalRunner/1.0; +https://puras.co)"
# Tags whose CONTENT is never visible text. `head` is intentionally NOT here —
# it carries the <title> we want; its other children (meta/link) are void, and
# any <style>/<script> inside it are skipped on their own.
_SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}
# Block-level tags that should force a line break in the extracted text.
_BLOCK_TAGS = {
    "p", "br", "div", "li", "tr", "section", "article", "header", "footer",
    "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "table", "blockquote", "pre",
}


class _TextExtractor(HTMLParser):
    """Minimal HTML→text: drop script/style content, keep <title>, insert breaks
    on block boundaries, collapse whitespace. Dependency-free (stdlib)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._in_title = False
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            self.title += data
            return
        self._parts.append(data)

    def text(self) -> str:
        joined = "".join(self._parts)
        joined = re.sub(r"[ \t\f\v]+", " ", joined)
        joined = re.sub(r"\n[ \t]+", "\n", joined)
        joined = re.sub(r"\n{3,}", "\n\n", joined)
        return joined.strip()


def fetch(url: str, max_chars: int = 20000) -> dict:
    """Fetch a URL over plain HTTP and return its readable text. Mirrors the
    hosted `/v1/web/fetch` shape (`url`, `title`, `content`, `length`). HTML is
    reduced to text; non-HTML bodies are returned as-is. No JavaScript — the
    agent's `web_fetch(render_js=true)` renders in the headless browser instead.
    """
    import httpx

    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise LocalWebError("url must start with http:// or https://")
    try:
        n = int(max_chars)
    except (TypeError, ValueError):
        n = 20000
    n = max(500, min(n, 200000))

    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(30.0),
            headers={"User-Agent": _DEFAULT_UA},
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
    except httpx.HTTPError as e:
        raise LocalWebError(f"fetch failed: {e}") from e

    ctype = resp.headers.get("content-type", "").lower()
    body = resp.text
    title = ""
    if "html" in ctype or (not ctype and body.lstrip()[:1] == "<"):
        parser = _TextExtractor()
        try:
            parser.feed(body)
        except Exception:
            # A malformed page still gives us whatever was parsed so far.
            pass
        title = parser.title.strip()
        text = parser.text()
    else:
        text = body

    truncated = len(text) > n
    if truncated:
        text = text[:n] + "\n…[truncated]"
    return {
        "url": str(resp.url),
        "title": title,
        "content": text,
        "length": len(text),
        "truncated": truncated,
        "rendered": False,
        "billed_micros": 0,
    }


# ---------------------------------------------------------------------------
# web_search — Anthropic's server-side web_search tool, BYO ANTHROPIC_API_KEY.
# ---------------------------------------------------------------------------
def _block_field(obj: Any, key: str) -> Any:
    """Read a field off an Anthropic response block whether it's a typed SDK
    model (attr) or a plain dict (older/loosely-typed responses)."""
    v = getattr(obj, key, None)
    if v is None and isinstance(obj, dict):
        v = obj.get(key)
    return v


def _block_type(obj: Any) -> str | None:
    t = _block_field(obj, "type")
    return t if isinstance(t, str) else None


def _extract_search_results(resp: Any, limit: int) -> list[dict]:
    """Pull `web_search_result` items out of the `web_search_tool_result` blocks
    Anthropic returns when it runs the server-side web_search tool. Raises
    LocalWebError if the tool itself reported an error."""
    out: list[dict] = []
    for block in (getattr(resp, "content", None) or []):
        if _block_type(block) != "web_search_tool_result":
            continue
        content = _block_field(block, "content")
        # The whole result can be an error object instead of a result list.
        if not isinstance(content, list):
            if _block_type(content) == "web_search_tool_result_error":
                raise LocalWebError(
                    f"web search error: {_block_field(content, 'error_code')}"
                )
            continue
        for item in content:
            ityp = _block_type(item)
            if ityp == "web_search_tool_result_error":
                raise LocalWebError(
                    f"web search error: {_block_field(item, 'error_code')}"
                )
            if ityp not in (None, "web_search_result"):
                continue
            url = _block_field(item, "url")
            if not isinstance(url, str) or not url:
                continue
            res = {"title": _block_field(item, "title") or "", "url": url}
            page_age = _block_field(item, "page_age")
            if page_age:
                res["page_age"] = page_age
            out.append(res)
            if len(out) >= limit:
                return out
    return out


def search(query: str, max_results: int = 5) -> dict:
    """Run a web search through Anthropic's server-side web_search tool on the BYO
    ANTHROPIC_API_KEY and return the hosted `/v1/web/search` shape (a `results`
    list of `{title, url}`). There is no local search engine, so the offline
    web_search delegates to Anthropic."""
    if not isinstance(query, str) or not query.strip():
        raise LocalWebError("query must be a non-empty string")
    try:
        m = int(max_results)
    except (TypeError, ValueError):
        m = 5
    m = max(1, min(m, 20))

    s = get_settings()
    key = (s.anthropic_api_key or "").strip()
    if not key:
        raise LocalWebError(
            "no ANTHROPIC_API_KEY — local web_search runs through Anthropic's "
            "server-side web_search tool (BYO key, like your LLM key)."
        )
    try:
        from anthropic import Anthropic
    except ImportError as e:  # pragma: no cover - anthropic is a core dep
        raise LocalWebError(
            "web search needs the `anthropic` package (pip install anthropic)."
        ) from e

    client = Anthropic(api_key=key)
    try:
        resp = client.messages.create(
            model=s.local_web_search_model,
            max_tokens=1024,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Search the web for: {query}\n\n"
                        "Use the web_search tool once to find the most relevant "
                        "results, then stop."
                    ),
                }
            ],
        )
    except LocalWebError:
        raise
    except Exception as e:
        raise LocalWebError(f"web search failed upstream: {e}") from e

    return {"query": query, "results": _extract_search_results(resp, m), "billed_micros": 0}
