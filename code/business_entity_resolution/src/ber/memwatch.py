"""Peak-memory sampler for a job and all of its child processes.

Usage: python -m ber.memwatch <pid> <out.json> [interval_sec]

Sums PSS (proportional set size: shared copy-on-write pages split between the
processes sharing them) over the process tree every interval, until the pid exits,
and writes the peak plus the peak system-wide used memory.
"""
import json
import os
import sys
import time


def _children(pid: int) -> list:
    out = []
    try:
        for tid in os.listdir(f"/proc/{pid}/task"):
            with open(f"/proc/{pid}/task/{tid}/children") as f:
                out += [int(x) for x in f.read().split()]
    except OSError:
        pass
    return out


def _tree(pid: int) -> list:
    seen, stack = [], [pid]
    while stack:
        p = stack.pop()
        seen.append(p)
        stack += _children(p)
    return seen


def _pss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _sys_used_kb() -> int:
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            info[k] = int(v.split()[0])
    return info["MemTotal"] - info["MemAvailable"]


def main(pid: int, out: str, interval: float = 3.0):
    t0, peak, peak_sys, base_sys = time.time(), 0, 0, _sys_used_kb()
    while os.path.exists(f"/proc/{pid}"):
        pss = sum(_pss_kb(p) for p in _tree(pid))
        peak = max(peak, pss)
        peak_sys = max(peak_sys, _sys_used_kb())
        with open(out, "w") as f:
            json.dump({"pid": pid, "elapsed_sec": round(time.time() - t0), "peak_tree_pss_gb": round(peak / 2**20, 2),
                       "current_tree_pss_gb": round(pss / 2**20, 2), "system_used_at_start_gb": round(base_sys / 2**20, 1),
                       "peak_system_used_gb": round(peak_sys / 2**20, 1)}, f, indent=1)
        time.sleep(interval)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 3.0)
