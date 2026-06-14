"""Self-sampled machine resource health for the worker registry / admin view.

Reads the machine's OWN CPU and memory from /proc. Fly machines are Firecracker
microVMs, so /proc reflects the VM's allocated 1 vCPU / 2GB — not the host. All
functions are best-effort: any value is None where unavailable (e.g. /proc is
absent on a macOS dev worker, or the first CPU sample which needs a delta).

CPU% is computed from the delta between consecutive /proc/stat snapshots, so it
reflects utilization over the heartbeat interval (~10s) — call sample() once per
heartbeat tick and the module keeps the previous snapshot.
"""

from __future__ import annotations

# Previous /proc/stat (idle, total) jiffies, for the CPU% delta.
_prev_cpu: tuple[int, int] | None = None


def _read_cpu_pct() -> float | None:
    """Busy CPU% since the previous call (None on the first call / on error)."""
    global _prev_cpu
    try:
        with open("/proc/stat") as f:
            parts = f.readline().split()
        if not parts or parts[0] != "cpu":
            return None
        nums = [int(x) for x in parts[1:]]
        # user nice system idle iowait irq softirq steal ...
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)  # idle + iowait
        total = sum(nums)
        prev = _prev_cpu
        _prev_cpu = (idle, total)
        if prev is None:
            return None  # need two samples for a delta
        d_idle = idle - prev[0]
        d_total = total - prev[1]
        if d_total <= 0:
            return None
        return round(100.0 * (1.0 - d_idle / d_total), 1)
    except Exception:
        return None


def _read_mem() -> tuple[float | None, int | None]:
    """(used_pct, total_mb) from /proc/meminfo, or (None, None) on error."""
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, rest = line.partition(":")
                info[k] = int(rest.strip().split()[0])  # value in kB
        total_kb = info.get("MemTotal")
        avail_kb = info.get("MemAvailable")
        if not total_kb or avail_kb is None:
            return None, None
        used_pct = round(100.0 * (total_kb - avail_kb) / total_kb, 1)
        return used_pct, round(total_kb / 1024)
    except Exception:
        return None, None


def sample() -> tuple[float | None, float | None, int | None]:
    """Return (cpu_pct, mem_used_pct, mem_total_mb); any element may be None."""
    cpu = _read_cpu_pct()
    mem_pct, mem_total = _read_mem()
    return cpu, mem_pct, mem_total


def mem_used_pct() -> float | None:
    """Used-memory percentage right now, or None where /proc is absent.

    Stateless (unlike sample(), which keeps a CPU-delta snapshot) so callers
    like the claim-admission gate can poll it without disturbing the heartbeat
    loop's CPU% reading.
    """
    used_pct, _ = _read_mem()
    return used_pct
