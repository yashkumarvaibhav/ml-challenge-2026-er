"""Line-based progress with ETA for long jobs.

Every update is printed to stdout (the job's own log) and appended to the shared
`logs/progress.log`, which the user watches in a terminal with
`tail -n 40 -F logs/progress.log` (see AGENTS.md).
"""
import sys
import time
from datetime import datetime, timezone

from . import config as C

PROGRESS_LOG = C.ROOT / "logs" / "progress.log"


def _fmt(sec: float) -> str:
    sec = int(max(0, sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc):%H:%M:%S}Z {msg}"
    print(line, flush=True)
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(line + "\n")


class Progress:
    """Usage: p = Progress("normalize train s2", total=n); p.update(k); ...; p.done()."""

    def __init__(self, name: str, total: int, every_sec: float = 10.0, unit: str = "items"):
        self.name, self.total, self.every, self.unit = name, max(1, total), every_sec, unit
        self.n, self.t0 = 0, time.time()
        self.last = self.t0  # first update after every_sec, so the ETA is meaningful
        log(f"[{name}] START total={total:,} {unit}")

    def update(self, k: int = 1) -> None:
        self.n += k
        now = time.time()
        if now - self.last >= self.every or self.n >= self.total:
            self.last = now
            el = now - self.t0
            eta = el / self.n * (self.total - self.n) if self.n else float("inf")
            log(f"[{self.name}] {100 * self.n / self.total:5.1f}% {self.n:,}/{self.total:,} {self.unit} "
                f"elapsed {_fmt(el)} ETA {_fmt(eta) if self.n else '?'}")

    def done(self, extra: str = "") -> None:
        log(f"[{self.name}] DONE in {_fmt(time.time() - self.t0)} {extra}".rstrip())


def step(name: str, msg: str) -> None:
    """One-off milestone line for stages without a natural item count."""
    log(f"[{name}] {msg}")


if __name__ == "__main__":  # smoke test
    p = Progress("smoke", 3, every_sec=0)
    for _ in range(3):
        p.update()
    p.done()
    sys.exit(0)
