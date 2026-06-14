"""Keep a runaway child subprocess from taking the whole worker machine down.

WORKER_CONCURRENCY jobs run in one process on one machine, and several of them
spawn ffmpeg (stitch, auto-caption burn, or a manual bash encode). An ffmpeg
encode can spike RAM past the VM's limit; the Linux OOM-killer then kills the
process with the worst "badness" score — which, because the worker process holds
every concurrent job's context, is often the WORKER itself. That strands all of
its jobs at once (their heartbeats stop → the API reaper fails them).

`child_preexec()` returns a `preexec_fn` for those heavy children that, in the
forked child before exec:
  - raises oom_score_adj to the max (1000) so the kernel kills THIS child (and
    the ffmpeg it forks — oom_score_adj is inherited) before the worker. Raising
    one's own badness needs no privilege, so this works as a non-root container
    too. The worker process keeps its default score (0) and survives; the one
    job that ran the encode gets a clean "killed" tool error instead.
  - optionally caps address space (RLIMIT_AS) when CHILD_MEM_LIMIT_MB > 0, as a
    secondary hard stop. Off by default — VSZ limits misfire on ffmpeg/Python.

On a host without procfs (macOS dev) and with no RLIMIT cap configured there is
nothing to set, so the factory returns None and subprocess runs unchanged.
"""

from __future__ import annotations

import os

from .config import get_settings

# Max OOM badness: "kill me first." Inherited across fork+exec to ffmpeg.
_OOM_SCORE_ADJ_MAX = b"1000"
_OOM_SCORE_ADJ_PATH = "/proc/self/oom_score_adj"


def child_preexec():
    """Return a preexec_fn for heavy child subprocesses, or None when there's
    nothing to apply on this platform.

    The returned closure captures the memory cap read here in the PARENT, so the
    child side touches no Python-level locks (it must stay fork-safe): it only
    makes os-level syscalls.
    """
    limit_mb = max(0, get_settings().child_mem_limit_mb)
    can_oom = os.path.exists(_OOM_SCORE_ADJ_PATH)
    if not can_oom and limit_mb <= 0:
        return None

    def _apply() -> None:
        if can_oom:
            try:
                fd = os.open(_OOM_SCORE_ADJ_PATH, os.O_WRONLY)
                try:
                    os.write(fd, _OOM_SCORE_ADJ_MAX)
                finally:
                    os.close(fd)
            except OSError:
                pass  # best effort — never block the exec on this
        if limit_mb > 0:
            try:
                import resource

                nbytes = limit_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
            except (ImportError, ValueError, OSError):
                pass

    return _apply
