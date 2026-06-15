"""Content-aware context compression for agent tool results (token economy).

A multi-step agent re-reads its whole conversation on every turn, so a large
tool result (a fetched page, long bash stdout, a verbose subagent return, a big
JSON payload) is paid for in tokens on every later step it stays inline. This
module shrinks such a result *in place* before it enters the append-only
history: a small ContentRouter picks a specialized compressor by content kind,
each reducing tokens while preserving the information the model needs.

  - JSON:  minify, collapse long homogeneous arrays-of-objects to a sample + a
           count, truncate oversized strings.
  - Python: drop comments + blank lines via `tokenize` (exact, string-safe);
           other content falls to the text pass.
  - logs/prose: strip ANSI + trailing whitespace, fold runs of blank or
           identical lines to one + `… (×N)`.

Two properties the agent loop depends on:

  * Deterministic & pure — same input → same output. A cached/replayed run
    (worker.prompt_cache keys on the messages) reproduces identical history, and
    the provider-native KV-cache prefix stays aligned because compression is
    append-only: it only ever shapes a NEW result as it lands, never rewrites an
    earlier turn (the cache-invalidating mistake).
  * Never grows, never raises, never loses silently — any error, or a result
    that didn't shrink, returns the original untouched. A lossy pass flags
    `lossless=False` so the caller can persist the exact original to the drive
    and hand the model a `file_read` pointer (reversible retrieval).

Stdlib-only, so it stays on the dependency-light offline-runner import path
(see tests/dry/test_local_import_isolation.py).
"""

from __future__ import annotations

import io
import json
import re
import tokenize
from dataclasses import dataclass

# --- tunables (module-level so they're trivially testable/overridable) --------
# A JSON string longer than this is truncated to its head + a `…(+N chars)` tag.
_JSON_STR_MAX = 600
# Only collapse an array once it has at least this many items…
_JSON_ARRAY_MIN = 8
# …and keep this many leading items as a representative sample.
_JSON_ARRAY_SAMPLE = 3
# Stop recursing into pathologically nested JSON (cycles can't occur in parsed
# JSON, but depth guards against a stack blow-up on adversarial input).
_JSON_MAX_DEPTH = 16
# A run of this many identical consecutive lines folds to one + `… (×N …)`.
_DEDUP_MIN_RUN = 3

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
# Strong Python signals — used only to ROUTE to the code compressor, never to
# decide what to strip (that's `tokenize`'s job, which is string-safe).
_PY_SIG_RE = re.compile(r"^\s*(def |class |import |from \w[\w.]* import |@\w|async def )")


@dataclass(frozen=True)
class CompressionResult:
    """Outcome of one compression attempt.

    `applied` is True only when the text actually changed AND shrank; `lossless`
    is True when the compressed form is information-equivalent for the model
    (JSON minify, ANSI/whitespace cleanup) and so needs no reversible backup.
    """

    text: str
    kind: str  # "json" | "code" | "text" | "none"
    original_chars: int
    compressed_chars: int
    lossless: bool
    applied: bool

    @property
    def saved_chars(self) -> int:
        return max(0, self.original_chars - self.compressed_chars)

    @property
    def ratio(self) -> float:
        """Fraction of characters saved, in [0, 1)."""
        return self.saved_chars / self.original_chars if self.original_chars else 0.0


def _none_result(text: str) -> CompressionResult:
    n = len(text)
    return CompressionResult(text, "none", n, n, True, False)


def compress_text(text: str) -> CompressionResult:
    """Route `text` to a content-aware compressor and return the outcome.

    Always safe: returns an unchanged, `applied=False` result if `text` isn't a
    non-empty string, a compressor raises, or the output didn't actually shrink.
    """
    if not isinstance(text, str) or not text:
        return _none_result(text if isinstance(text, str) else "")
    n = len(text)
    try:
        kind = _route(text)
        if kind == "json":
            out, lossless = _crush_json(text)
        elif kind == "code":
            out, lossless = _crush_code(text)
        else:
            out, lossless = _crush_text(text)
    except Exception:
        return _none_result(text)
    if not isinstance(out, str) or len(out) >= n:
        return _none_result(text)
    return CompressionResult(out, kind, n, len(out), bool(lossless), True)


# --- ContentRouter ------------------------------------------------------------
def _route(text: str) -> str:
    head = text.lstrip()[:1]
    if head in "{[":
        try:
            json.loads(text)
            return "json"
        except (ValueError, TypeError):
            pass
    if _looks_like_python(text):
        return "code"
    return "text"


def _looks_like_python(text: str) -> bool:
    """Detect Python source with comments to strip. Conservative on purpose: the
    code pass only fires when there's clearly Python AND `#` comments to remove,
    so arbitrary log output is never mistaken for code."""
    lines = text.splitlines()
    if len(lines) < 5:
        return False
    sig = sum(1 for ln in lines if _PY_SIG_RE.match(ln))
    has_hash_comment = re.search(r"(?m)^\s*#", text) is not None
    return sig >= 3 and has_hash_comment


# --- JSON compressor ----------------------------------------------------------
def _crush_json(text: str) -> tuple[str, bool]:
    data = json.loads(text)
    crushed, lossy = _crush_value(data, 0)
    out = json.dumps(crushed, ensure_ascii=False, separators=(",", ":"), default=str)
    # Minification alone (whitespace) is information-equivalent; only a
    # structural collapse / string truncation makes it lossy.
    return out, not lossy


def _crush_value(v, depth: int) -> tuple[object, bool]:
    if depth > _JSON_MAX_DEPTH:
        return v, False
    if isinstance(v, str):
        if len(v) > _JSON_STR_MAX:
            return v[:_JSON_STR_MAX] + f"…(+{len(v) - _JSON_STR_MAX} chars)", True
        return v, False
    if isinstance(v, list):
        lossy = False
        new = []
        for item in v:
            cv, cl = _crush_value(item, depth + 1)
            new.append(cv)
            lossy = lossy or cl
        # Collapse a long array of OBJECTS to a leading sample + a summary. Only
        # object-arrays (the verbose case — search hits, rows, records); scalar
        # arrays are cheap and kept intact (after per-string truncation above).
        if len(new) >= _JSON_ARRAY_MIN and all(isinstance(x, dict) for x in v):
            sample = new[:_JSON_ARRAY_SAMPLE]
            summary = {
                "__compressed__": (
                    f"{len(new)} items total; first {len(sample)} shown, "
                    f"{len(new) - len(sample)} elided (file_read the original for all)"
                ),
                "item_keys": sorted(v[0].keys()),
            }
            return sample + [summary], True
        return new, lossy
    if isinstance(v, dict):
        lossy = False
        out: dict = {}
        for k, val in v.items():
            cv, cl = _crush_value(val, depth + 1)
            out[k] = cv
            lossy = lossy or cl
        return out, lossy
    return v, False


# --- Code compressor (Python) -------------------------------------------------
def _crush_code(text: str) -> tuple[str, bool]:
    stripped = _strip_python_comments(text)
    if stripped is None:
        return _crush_text(text)  # not tokenizable → safe text pass
    out = _collapse_blank_lines(stripped)
    if len(out) >= len(text):
        return _crush_text(text)
    return out, False  # comments removed → lossy (reversible via the drive copy)


def _strip_python_comments(text: str) -> str | None:
    """Remove `#` comments with `tokenize` (so `#` inside a string is untouched).
    Returns None if the text doesn't tokenize cleanly — the caller then falls
    back to the always-safe text pass."""
    try:
        kept = [
            tok
            for tok in tokenize.generate_tokens(io.StringIO(text).readline)
            if tok.type != tokenize.COMMENT
        ]
        return tokenize.untokenize(kept)
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return None


# --- Text compressor (logs / prose) -------------------------------------------
def _crush_text(text: str) -> tuple[str, bool]:
    """Strip ANSI + trailing whitespace, fold runs of blank or identical lines.

    Lossless unless an identical-line run was folded (`deduped`) — line dedup
    drops repeated copies, so we flag it lossy to keep the original retrievable.
    """
    lines = _ANSI_RE.sub("", text).split("\n")
    out: list[str] = []
    deduped = False
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i].rstrip()
        if ln == "":
            # fold a run of blank lines to a single blank
            j = i + 1
            while j < n and lines[j].strip() == "":
                j += 1
            out.append("")
            i = j
            continue
        # fold a run of identical non-blank lines
        j = i + 1
        while j < n and lines[j].rstrip() == ln:
            j += 1
        run = j - i
        if run >= _DEDUP_MIN_RUN:
            out.append(ln)
            out.append(f"… (×{run} identical lines)")
            deduped = True
        else:
            out.extend([ln] * run)
        i = j
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out), not deduped


def _collapse_blank_lines(text: str) -> str:
    out: list[str] = []
    blank = False
    for ln in text.split("\n"):
        ln = ln.rstrip()
        if ln == "":
            if blank:
                continue
            blank = True
        else:
            blank = False
        out.append(ln)
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)
