"""Deterministic eval graders for `content-repurposer` — the unit-test layer.

These are to the skill what assertions are to a unit test: each grader is called
once per finished run with the run's `inputs` and `output`, and returns
`{score: 0..1, passed: bool, detail: str}`. They check only the objective,
machine-verifiable contract (one post per requested platform, each body within
its platform's character limit, no hashtags jammed into the body, the declared
`char_count` is honest, the posts aren't copies of each other). The subjective
"does it sound native / keep the same message" judgments live as `rubric` graders
in skill.yaml.

Pure stdlib — runs on the worker's base interpreter, no requirements.txt. A
grader entrypoint is `<file>:<func>`; the function's params are filled from the
eval runner as `func(inputs=<job inputs>, output=<job output>)`.
"""

from __future__ import annotations

import re

# Per-platform BODY character ceilings — mirrors tools/check_post.py LIMITS so the
# eval enforces the exact contract the skill validates against at runtime. Kept as
# an independent copy on purpose: a unit test shouldn't import the code under test
# and inherit its bugs.
LIMITS = {
    "x": 280,
    "threads": 500,
    "instagram": 2200,
    "linkedin": 3000,
    "reddit": 40000,
}

# The skill's default platform set, applied when the caller omits `platforms`
# (matches the input_schema default in skill.yaml).
DEFAULT_PLATFORMS = ["linkedin", "x", "reddit", "instagram"]


def _requested_platforms(inputs: dict) -> list[str]:
    plats = (inputs or {}).get("platforms")
    if isinstance(plats, list) and plats:
        return [str(p) for p in plats]
    return list(DEFAULT_PLATFORMS)


def _posts(output) -> list[dict]:
    posts = (output or {}).get("posts") if isinstance(output, dict) else None
    return [p for p in posts if isinstance(p, dict)] if isinstance(posts, list) else []


def _result(score: float, passed: bool, detail: str) -> dict:
    # Clamp defensively so a buggy grader can never push the aggregate out of range.
    s = 0.0 if score < 0 else 1.0 if score > 1 else float(score)
    return {"score": s, "passed": bool(passed), "detail": detail[:500]}


def platform_coverage(inputs: dict, output) -> dict:
    """Exactly one post per requested platform, in the requested order — no
    missing platforms, no extras, no duplicates."""
    requested = _requested_platforms(inputs)
    posts = _posts(output)
    got = [str(p.get("platform")) for p in posts]

    present_once = sum(1 for plat in requested if got.count(plat) == 1)
    score = present_once / len(requested) if requested else 0.0
    perfect = got == requested  # right set, right count, right order

    problems = []
    missing = [p for p in requested if p not in got]
    if missing:
        problems.append(f"missing: {', '.join(missing)}")
    extras = [p for p in got if p not in requested]
    if extras:
        problems.append(f"unexpected: {', '.join(extras)}")
    dupes = sorted({p for p in got if got.count(p) > 1})
    if dupes:
        problems.append(f"duplicated: {', '.join(dupes)}")
    if not problems and not perfect:
        problems.append(f"order {got} != requested {requested}")

    detail = "one native post per requested platform" if perfect else "; ".join(problems)
    return _result(score, perfect, detail)


def char_limits(inputs: dict, output) -> dict:
    """Every post body is within its platform's character limit (X ≤ 280 is the
    one the model keeps blowing)."""
    posts = _posts(output)
    if not posts:
        return _result(0.0, False, "no posts to check")

    within = 0
    offenders = []
    for p in posts:
        plat = str(p.get("platform") or "").lower()
        body = p.get("body") or ""
        limit = LIMITS.get(plat, 3000)
        n = len(body)
        if n <= limit:
            within += 1
        else:
            offenders.append(f"{plat} {n}/{limit} (+{n - limit})")

    score = within / len(posts)
    passed = not offenders
    detail = "all bodies within platform limits" if passed else "over limit → " + "; ".join(offenders)
    return _result(score, passed, detail)


def hashtag_hygiene(inputs: dict, output) -> dict:
    """No declared hashtag is also jammed into the body, and Reddit carries no
    hashtags at all (subreddit etiquette the skill commits to)."""
    posts = _posts(output)
    if not posts:
        return _result(0.0, False, "no posts to check")

    clean = 0
    problems = []
    for p in posts:
        plat = str(p.get("platform") or "").lower()
        body = (p.get("body") or "")
        body_low = body.lower()
        tags = p.get("hashtags") if isinstance(p.get("hashtags"), list) else []

        dupes = [
            t for t in tags
            if ("#" + str(t).lstrip("#").strip()).lower() in body_low
        ]
        reddit_tags = plat == "reddit" and len(tags) > 0

        if not dupes and not reddit_tags:
            clean += 1
            continue
        if dupes:
            problems.append(f"{plat}: hashtags in body ({', '.join(dupes)})")
        if reddit_tags:
            problems.append(f"{plat}: should carry no hashtags")

    score = clean / len(posts)
    passed = not problems
    detail = "hashtags kept out of bodies; reddit clean" if passed else "; ".join(problems)
    return _result(score, passed, detail)


def char_count_accuracy(inputs: dict, output) -> dict:
    """The declared `char_count` matches the real body length — the skill must
    report the value the validator returned, not eyeball it."""
    posts = _posts(output)
    if not posts:
        return _result(0.0, False, "no posts to check")

    accurate = 0
    problems = []
    for p in posts:
        plat = str(p.get("platform") or "").lower()
        body = p.get("body") or ""
        declared = p.get("char_count")
        actual = len(body)
        if isinstance(declared, bool) or not isinstance(declared, int):
            problems.append(f"{plat}: char_count missing/not an int")
            continue
        if declared == actual:
            accurate += 1
        else:
            problems.append(f"{plat}: declared {declared} != actual {actual}")

    score = accurate / len(posts)
    passed = not problems
    detail = "every char_count matches its body" if passed else "; ".join(problems)
    return _result(score, passed, detail)


_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(text: str) -> set:
    return set(_WORD_RE.findall((text or "").lower()))


def distinctness(inputs: dict, output) -> dict:
    """The posts aren't copy-paste of each other. Scores 1 − (max pairwise word
    Jaccard similarity); a near-duplicate pair drags the score down. The skill's
    core rule is "same message, different clothes", so two posts that read the
    same way are a failure."""
    posts = _posts(output)
    if len(posts) < 2:
        # Nothing to compare — a single (or empty) post can't be a copy.
        return _result(1.0, True, "fewer than two posts; nothing to compare")

    toks = [_tokens(p.get("body") or "") for p in posts]
    plats = [str(p.get("platform") or f"#{i}") for i, p in enumerate(posts)]

    max_sim = 0.0
    worst = ""
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            a, b = toks[i], toks[j]
            union = a | b
            if not union:
                continue
            sim = len(a & b) / len(union)
            if sim > max_sim:
                max_sim = sim
                worst = f"{plats[i]}↔{plats[j]}"

    score = 1.0 - max_sim
    passed = max_sim < 0.8
    detail = (
        f"posts are distinct (max similarity {max_sim:.0%})"
        if passed
        else f"near-duplicate posts: {worst} ({max_sim:.0%} word overlap)"
    )
    return _result(score, passed, detail)
