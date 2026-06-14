"""`check_post` — deterministic validator for one repurposed post.

The agent must not eyeball character counts or scan for stray hashtags — LLMs
are unreliable at both. This tool counts characters exactly, enforces each
platform's character limit, and flags any hashtag that also appears inside the
body. Call it once per drafted post, fix whatever it reports, and use the
returned `char_count` verbatim in the final output.

A tool entrypoint is `<file>:<func>`; the function's params are the tool's
declared `input_schema` properties and it returns a dict shaped like the tool's
`output_schema`. Pure stdlib — runs on the worker's base interpreter, so there
is no requirements.txt.
"""

from __future__ import annotations

# Character ceiling for the post BODY per platform. X is the strict one the
# agent keeps blowing; the rest are each platform's real limit (rarely hit, but
# worth catching). Hashtags live in their own field and don't count here.
LIMITS = {
    "x": 280,
    "threads": 500,
    "instagram": 2200,
    "linkedin": 3000,
    "reddit": 40000,  # Reddit self-text limit
}


def run(platform: str, body: str, hashtags: list | None = None) -> dict:
    body = body or ""
    hashtags = hashtags or []
    char_count = len(body)

    limit = LIMITS.get((platform or "").lower(), 3000)
    over_by = max(0, char_count - limit)
    within_limit = over_by == 0

    # A hashtag belongs in the `hashtags` field, never jammed into the body.
    # Flag any tag whose "#tag" form shows up in the body (case-insensitive).
    body_low = body.lower()
    duplicate_hashtags = [
        tag for tag in hashtags
        if ("#" + str(tag).lstrip("#").strip()).lower() in body_low
    ]

    issues = []
    if not within_limit:
        issues.append(
            f"{char_count} chars — {over_by} over the {limit}-char {platform} limit; tighten it."
        )
    if duplicate_hashtags:
        issues.append(
            "hashtags also appear in the body (keep them only in `hashtags`): "
            + ", ".join(duplicate_hashtags)
        )

    return {
        "char_count": char_count,
        "limit": limit,
        "within_limit": within_limit,
        "over_by": over_by,
        "duplicate_hashtags": duplicate_hashtags,
        "ok": within_limit and not duplicate_hashtags,
        "issues": issues,
    }
