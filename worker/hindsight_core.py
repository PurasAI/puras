"""Hindsight core — pure, runtime-agnostic retrospective detectors (open-core).

Given a WINDOW of a skill's recent runs — each run reduced to the trace facts we
already record (job_spans + job_events: every tool call with its input, whether
it succeeded, its error preview, plus the memory_injected / memory_miss signals)
— this finds recurring inefficiencies and returns `Finding`s. It is the sibling
of `optimizer_core` and, like it, depends on NOTHING (no DB, no platform API, no
event loop, no provider): the cloud resolver (`api/app/hindsight.py`) loads the
window from Postgres and the local CLI loads it from disk, but both call the
SAME detectors here.

It is deliberately DETERMINISTIC: detection is normalization + grouping (no
embeddings, no LLM), so the same window always yields the same findings and the
unit tests are exact. The grounded LLM step that turns a `Finding` into prose +
a draft artifact (a proposed tool spec / SKILL.md patch / memory action) lives in
the cloud layer — this core only finds the pattern and attaches the evidence.

This is REPORT-ONLY: a `Finding` is a suggestion with evidence. Nothing here (or
downstream) applies anything; a human reads the finding and decides.

The detector families mirror migration 052:
  * tool        — `detect_adhoc_code`: the same ad-hoc bash/python written across
                  runs → a missing first-class tool.
  * error       — `detect_repeated_errors`: the same tool failing the same way.
  * redundancy  — `detect_redundant_calls`: a run repeating an identical call.
  * memory      — `detect_memory_misses`, `detect_duplicate_writes`,
                  `detect_unused_injected`, `detect_refetch_candidates`.

(The `prompt` family — a proposed SKILL.md improvement — is synthesized in the
cloud layer from the union of these findings, so it has no pure detector here.)
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Value types — the window the detectors consume
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """One tool invocation inside a run, joined from the `tool_use` event (name +
    input) and its matching `tool_result` (ok + error preview). `ok=None` means we
    never saw the result (e.g. the run died mid-call)."""

    job_id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    ok: bool | None = None
    error_preview: str = ""


@dataclass
class MemorySignal:
    """What memory did for one run, from the `memory_injected` / `memory_miss`
    events (see agent_runner): which rows were injected into the first-turn
    digest, and whether the lookup missed entirely (identity keys but no hits)."""

    injected_ids: list[str] = field(default_factory=list)
    injected_count: int = 0
    missed: bool = False
    miss_keys: list[str] = field(default_factory=list)


@dataclass
class RunTrace:
    """One analyzed run: its tool calls in order + its memory signal."""

    job_id: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    memory: MemorySignal = field(default_factory=MemorySignal)


@dataclass
class Window:
    """The unit a Hindsight run analyzes: a skill's last N runs."""

    skill_name: str
    runs: list[RunTrace] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        return len(self.runs)


@dataclass
class Finding:
    """One detected pattern. `evidence` carries the job_ids it spans plus a few
    representative samples; the cloud layer adds prose + a draft from this."""

    family: str  # 'tool' | 'error' | 'redundancy' | 'memory'
    kind: str
    title: str
    severity: str = "medium"  # 'low' | 'medium' | 'high'
    evidence: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


# ---------------------------------------------------------------------------
# Detection thresholds — tuned for a small window (the product default is ~10
# runs). Kept as module constants so the cloud layer and tests share one source.
# ---------------------------------------------------------------------------

# A pattern must recur across at least this many DISTINCT runs to be a finding —
# a one-off isn't a pipeline problem.
MIN_DISTINCT_RUNS = 2
# An in-run repetition (same call twice+) is redundant at this count.
MIN_INTRA_RUN_REPEATS = 2
# Severity ramps with how much of the window a pattern touches.
_HIGH_RUN_FRACTION = 0.6
_MED_RUN_FRACTION = 0.35


def _severity(distinct_runs: int, n_runs: int) -> str:
    """Severity from the share of the window a pattern touches (capped at the
    window size so a single very-busy run can't inflate it)."""
    if n_runs <= 0:
        return "low"
    frac = min(distinct_runs, n_runs) / n_runs
    if frac >= _HIGH_RUN_FRACTION:
        return "high"
    if frac >= _MED_RUN_FRACTION:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Normalization helpers (the deterministic heart: collapse incidental variation
# so "the same script with a different URL/path" groups together)
# ---------------------------------------------------------------------------

_STR_LITERAL = re.compile(r"""(['"]).*?\1""", re.DOTALL)
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_ABS_PATH = re.compile(r"(/[\w.\-]+)+")  # absolute / nested paths: /tmp/a.json
_REL_PATH = re.compile(r"\b[\w.\-]+(?:/[\w.\-]+)+\b")  # relative: dir/sub/file
_FILE = re.compile(r"\b[\w\-]+\.[A-Za-z0-9]{1,6}\b")  # bare filename: in1.csv, conv.py
_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+", re.IGNORECASE)


def normalize_command(text: str) -> str:
    """Reduce a shell/script body to its STRUCTURAL signature: URLs, paths,
    filenames, and string/numeric literals become placeholders and whitespace
    collapses. Two runs that wrote `python conv.py in1.csv` and
    `python conv.py in2.csv` (or `curl https://a -o /tmp/a.json` and
    `curl https://b -o /tmp/b.json`) normalize to the same signature, so they
    group as one recurring script. Filenames collapse too, so detection groups by
    command STRUCTURE rather than the exact data file passed each run."""
    if not isinstance(text, str):
        text = str(text)
    t = _URL.sub("§URL§", text)
    t = _STR_LITERAL.sub("§STR§", t)
    t = _ABS_PATH.sub("§PATH§", t)
    t = _REL_PATH.sub("§PATH§", t)
    t = _FILE.sub("§PATH§", t)
    t = _NUMBER.sub("§N§", t)
    t = _WS.sub(" ", t).strip()
    return t


def _signature(text: str) -> str:
    """A short stable id for a normalized body (for grouping + evidence refs)."""
    return hashlib.sha1(normalize_command(text).encode("utf-8")).hexdigest()[:12]


def normalize_error(text: str) -> str:
    """Collapse an error preview to its class: drop literals/numbers/paths so
    `file not found: /a/1` and `file not found: /b/2` group as one error."""
    if not isinstance(text, str):
        text = str(text)
    t = _URL.sub("§URL§", text)
    t = _ABS_PATH.sub("§PATH§", t)
    t = _REL_PATH.sub("§PATH§", t)
    t = _FILE.sub("§PATH§", t)
    t = _NUMBER.sub("§N§", t)
    t = _STR_LITERAL.sub("§STR§", t)
    return _WS.sub(" ", t).strip()[:200]


def _bash_command(call: ToolCall) -> str | None:
    """The shell body of a `bash` call, if this is one."""
    if call.name != "bash":
        return None
    cmd = call.input.get("command")
    return cmd if isinstance(cmd, str) and cmd.strip() else None


def _canonical_input(call: ToolCall) -> str:
    """A stable key for "the same call with the SAME args" (intra-run redundancy).
    Uses EXACT argument values (not normalized): redundancy means the agent made
    the literally identical call twice, so `web_fetch a.com` and `web_fetch b.com`
    must stay distinct. Sorted so key order can't make two identical calls differ."""
    try:
        items = sorted((str(k), str(v).strip()) for k, v in call.input.items())
    except Exception:
        items = [("_repr", str(call.input))]
    return call.name + "|" + "|".join(f"{k}={v}" for k, v in items)


# ---------------------------------------------------------------------------
# tool family
# ---------------------------------------------------------------------------


def detect_adhoc_code(window: Window) -> list[Finding]:
    """The agent kept writing the SAME ad-hoc bash/python across runs → it needs a
    first-class tool. Group `bash` bodies by structural signature; a signature
    seen in >= MIN_DISTINCT_RUNS distinct runs is a missing-tool candidate."""
    by_sig_runs: dict[str, set[str]] = defaultdict(set)
    by_sig_total: Counter = Counter()
    sample: dict[str, str] = {}
    for run in window.runs:
        for call in run.tool_calls:
            cmd = _bash_command(call)
            if cmd is None:
                continue
            sig = _signature(cmd)
            by_sig_runs[sig].add(run.job_id)
            by_sig_total[sig] += 1
            sample.setdefault(sig, cmd)

    findings: list[Finding] = []
    for sig, runs in by_sig_runs.items():
        if len(runs) < MIN_DISTINCT_RUNS:
            continue
        findings.append(
            Finding(
                family="tool",
                kind="adhoc_code",
                title="Recurring ad-hoc script — candidate for a first-class tool",
                severity=_severity(len(runs), window.n_runs),
                evidence={
                    "signature": sig,
                    "distinct_runs": len(runs),
                    "total_calls": by_sig_total[sig],
                    "job_ids": sorted(runs),
                    "sample_command": sample[sig][:2000],
                },
                recommendation=(
                    "This shell/python body was written in "
                    f"{len(runs)} separate runs. Codify it as a tool (or skill) so "
                    "the agent calls it directly instead of re-authoring it each run."
                ),
            )
        )
    findings.sort(key=lambda f: f.evidence["distinct_runs"], reverse=True)
    return findings


# ---------------------------------------------------------------------------
# error family
# ---------------------------------------------------------------------------


def detect_repeated_errors(window: Window) -> list[Finding]:
    """The same tool failed the same way across runs → a prompt/tool fix. Group
    failed calls by (tool, normalized error); >= MIN_DISTINCT_RUNS distinct runs
    is a finding."""
    by_key_runs: dict[tuple[str, str], set[str]] = defaultdict(set)
    by_key_total: Counter = Counter()
    sample: dict[tuple[str, str], str] = {}
    for run in window.runs:
        for call in run.tool_calls:
            if call.ok is not False:
                continue
            err = normalize_error(call.error_preview)
            key = (call.name, err)
            by_key_runs[key].add(run.job_id)
            by_key_total[key] += 1
            sample.setdefault(key, call.error_preview)

    findings: list[Finding] = []
    for (tool, err), runs in by_key_runs.items():
        if len(runs) < MIN_DISTINCT_RUNS:
            continue
        findings.append(
            Finding(
                family="error",
                kind="repeated_error",
                title=f"`{tool}` keeps failing the same way",
                severity=_severity(len(runs), window.n_runs),
                evidence={
                    "tool": tool,
                    "error_class": err,
                    "distinct_runs": len(runs),
                    "total_failures": by_key_total[(tool, err)],
                    "job_ids": sorted(runs),
                    "sample_error": sample[(tool, err)][:1000],
                },
                recommendation=(
                    f"`{tool}` failed with this same error in {len(runs)} runs. Add "
                    "guidance to SKILL.md (or fix the tool) so the agent avoids or "
                    "handles it instead of failing-and-retrying."
                ),
            )
        )
    findings.sort(key=lambda f: f.evidence["distinct_runs"], reverse=True)
    return findings


# ---------------------------------------------------------------------------
# redundancy family
# ---------------------------------------------------------------------------


def detect_redundant_calls(window: Window) -> list[Finding]:
    """Within a single run, the agent made the SAME call (same tool + same args)
    more than once — it already had the answer (a re-fetched URL, a re-read memory
    id, a re-run script). One finding per (tool) summarizing the wasted calls."""
    waste_by_tool: Counter = Counter()
    runs_by_tool: dict[str, set[str]] = defaultdict(set)
    sample_by_tool: dict[str, str] = {}
    for run in window.runs:
        seen: Counter = Counter()
        for call in run.tool_calls:
            if call.name == "set_output":
                continue
            key = _canonical_input(call)
            seen[key] += 1
        for key, count in seen.items():
            if count < MIN_INTRA_RUN_REPEATS:
                continue
            tool = key.split("|", 1)[0]
            waste_by_tool[tool] += count - 1  # first call is legitimate
            runs_by_tool[tool].add(run.job_id)
            sample_by_tool.setdefault(tool, key)

    findings: list[Finding] = []
    for tool, wasted in waste_by_tool.items():
        runs = runs_by_tool[tool]
        findings.append(
            Finding(
                family="redundancy",
                kind="redundant_call",
                title=f"`{tool}` called with identical arguments repeatedly",
                severity=_severity(len(runs), window.n_runs),
                evidence={
                    "tool": tool,
                    "wasted_calls": wasted,
                    "distinct_runs": len(runs),
                    "job_ids": sorted(runs),
                    "sample_call": sample_by_tool[tool][:1000],
                },
                recommendation=(
                    f"`{tool}` was invoked with the same arguments multiple times in "
                    f"{len(runs)} run(s) ({wasted} redundant call(s)). Tell the agent "
                    "to reuse the earlier result instead of repeating the call."
                ),
            )
        )
    findings.sort(key=lambda f: f.evidence["wasted_calls"], reverse=True)
    return findings


# ---------------------------------------------------------------------------
# memory family
# ---------------------------------------------------------------------------


def detect_memory_misses(window: Window) -> list[Finding]:
    """Runs where memory lookup MISSED (identity keys present, no hits) — the
    agent had to re-derive everything. Frequent misses mean the skill should be
    writing memory it currently doesn't."""
    missed = [run for run in window.runs if run.memory.missed]
    if len(missed) < MIN_DISTINCT_RUNS:
        return []
    keys: Counter = Counter()
    for run in missed:
        keys.update(run.memory.miss_keys)
    return [
        Finding(
            family="memory",
            kind="memory_miss",
            title="Memory was cold — the agent re-derived instead of recalling",
            severity=_severity(len(missed), window.n_runs),
            evidence={
                "distinct_runs": len(missed),
                "job_ids": sorted(r.job_id for r in missed),
                "top_miss_keys": [k for k, _ in keys.most_common(10)],
            },
            recommendation=(
                f"Memory lookup missed in {len(missed)} runs. Add a `memory_put` step "
                "(or fix identity-key derivation) so recurring subjects are recalled "
                "instead of recomputed every run."
            ),
        )
    ]


def detect_duplicate_writes(window: Window) -> list[Finding]:
    """The agent wrote near-duplicate `memory_put` rows for the same subject across
    runs — memory bloat that hurts retrieval. Group by entity_key/title."""
    by_subject_runs: dict[str, set[str]] = defaultdict(set)
    by_subject_total: Counter = Counter()
    for run in window.runs:
        for call in run.tool_calls:
            if call.name != "memory_put":
                continue
            subject = (
                call.input.get("entity_key")
                or call.input.get("title")
                or ""
            )
            subject = normalize_command(str(subject))
            if not subject:
                continue
            by_subject_runs[subject].add(run.job_id)
            by_subject_total[subject] += 1

    findings: list[Finding] = []
    for subject, runs in by_subject_runs.items():
        # Duplicate = written in multiple runs OR multiple times overall.
        if len(runs) < MIN_DISTINCT_RUNS and by_subject_total[subject] < MIN_INTRA_RUN_REPEATS + 1:
            continue
        findings.append(
            Finding(
                family="memory",
                kind="duplicate_write",
                title="Near-duplicate memory writes for the same subject",
                severity=_severity(len(runs), window.n_runs),
                evidence={
                    "subject": subject[:200],
                    "distinct_runs": len(runs),
                    "total_writes": by_subject_total[subject],
                    "job_ids": sorted(runs),
                },
                recommendation=(
                    f"`memory_put` wrote this subject {by_subject_total[subject]} times. "
                    "Have the agent update/supersede the existing record instead of "
                    "writing a new one, or merge the duplicates."
                ),
            )
        )
    findings.sort(key=lambda f: f.evidence["total_writes"], reverse=True)
    return findings


def detect_unused_injected(window: Window) -> list[Finding]:
    """Memory rows injected into the first-turn digest that the run NEVER read back
    (no `memory_get` for them) — digest budget (and tokens) spent on dead weight.
    Reported per memory id that was injected-but-unused across multiple runs."""
    injected_runs: dict[str, set[str]] = defaultdict(set)
    used_runs: dict[str, set[str]] = defaultdict(set)
    for run in window.runs:
        for mid in run.memory.injected_ids:
            injected_runs[mid].add(run.job_id)
        for call in run.tool_calls:
            if call.name == "memory_get":
                mid = call.input.get("id")
                if isinstance(mid, str):
                    used_runs[mid].add(run.job_id)

    findings: list[Finding] = []
    for mid, runs in injected_runs.items():
        unused = runs - used_runs.get(mid, set())
        if len(unused) < MIN_DISTINCT_RUNS:
            continue
        findings.append(
            Finding(
                family="memory",
                kind="unused_injected",
                title="Injected memory went unused — digest budget waste",
                severity=_severity(len(unused), window.n_runs),
                evidence={
                    "memory_id": mid,
                    "injected_runs": len(runs),
                    "unused_runs": len(unused),
                    "job_ids": sorted(unused),
                },
                recommendation=(
                    f"Memory {mid} was injected into the digest in {len(unused)} runs "
                    "but never read. Lower its importance (or unpin it) so it stops "
                    "consuming the digest budget."
                ),
            )
        )
    findings.sort(key=lambda f: f.evidence["unused_runs"], reverse=True)
    return findings


def detect_refetch_candidates(window: Window) -> list[Finding]:
    """A memory id `memory_get`-fetched in (almost) every run is hot context the
    digest should carry directly — a PIN candidate."""
    fetch_runs: dict[str, set[str]] = defaultdict(set)
    for run in window.runs:
        for call in run.tool_calls:
            if call.name == "memory_get":
                mid = call.input.get("id")
                if isinstance(mid, str):
                    fetch_runs[mid].add(run.job_id)

    findings: list[Finding] = []
    threshold = max(MIN_DISTINCT_RUNS, int(window.n_runs * _HIGH_RUN_FRACTION))
    for mid, runs in fetch_runs.items():
        if len(runs) < threshold:
            continue
        findings.append(
            Finding(
                family="memory",
                kind="refetch_candidate",
                title="Memory re-fetched almost every run — pin candidate",
                severity=_severity(len(runs), window.n_runs),
                evidence={
                    "memory_id": mid,
                    "fetched_runs": len(runs),
                    "job_ids": sorted(runs),
                },
                recommendation=(
                    f"Memory {mid} was fetched in {len(runs)}/{window.n_runs} runs. "
                    "Pin it (or raise its importance) so it rides in the digest and "
                    "the agent stops paying a `memory_get` round-trip each run."
                ),
            )
        )
    findings.sort(key=lambda f: f.evidence["fetched_runs"], reverse=True)
    return findings


# ---------------------------------------------------------------------------
# Aggregate entry point
# ---------------------------------------------------------------------------

ALL_DETECTORS = (
    detect_adhoc_code,
    detect_repeated_errors,
    detect_redundant_calls,
    detect_memory_misses,
    detect_duplicate_writes,
    detect_unused_injected,
    detect_refetch_candidates,
)


def analyze(window: Window) -> list[Finding]:
    """Run every detector over the window and return all findings, ordered by
    severity (high → low) then family. This is the one call the cloud resolver and
    the local CLI both make; the LLM-prose/draft step consumes the result."""
    findings: list[Finding] = []
    for detector in ALL_DETECTORS:
        findings.extend(detector(window))
    rank = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: (rank.get(f.severity, 3), f.family, f.kind))
    return findings
