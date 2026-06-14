"""Tiny terminal output helpers — no third-party deps, ANSI only when on a TTY."""

from __future__ import annotations

import sys

_TTY = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def dim(s: str) -> str:
    return _c("2", s)


def bold(s: str) -> str:
    return _c("1", s)


def green(s: str) -> str:
    return _c("32", s)


def red(s: str) -> str:
    return _c("31", s)


def accent(s: str) -> str:
    return _c("35", s)


def ok(msg: str) -> None:
    print(f"{green('✓')} {msg}")


def info(msg: str) -> None:
    print(msg)


def warn(msg: str) -> None:
    print(f"{red('✗')} {msg}", file=sys.stderr)


def table(rows: list[list], headers: list[str]) -> None:
    """Left-aligned fixed-width columns. `rows` is a list of cell lists."""
    str_rows = [[str(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in str_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    print(dim(fmt(headers)))
    for r in str_rows:
        print(fmt(r))


def mask_key(k: str | None) -> str:
    """puras_live_<prefix8>.<secret32> → puras_live_<prefix8>.****"""
    if not k:
        return "(none)"
    if "." in k:
        head, _ = k.split(".", 1)
        return head + ".****"
    return (k[:12] + "****") if len(k) > 12 else "****"
