# utils/wilson.py
from __future__ import annotations
import math
from collections import deque
from typing import Iterable, Literal, Dict, Tuple

Outcome = Literal["success", "collision", "timeout"]

def wilson_ci(k: int, n: int, conf: float = 0.95) -> Tuple[float, float]:
    """
    Wilson score interval for binomial proportion.
    Returns (lower, upper). If n==0, returns (0.0, 1.0).
    """
    if n <= 0:
        return (0.0, 1.0)
    z = {0.90:1.6448536269514722, 0.95:1.959963984540054, 0.975:2.241402727, 0.99:2.5758293035489004}.get(conf, 1.959963984540054)
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2*n)) / denom
    half = z * math.sqrt(phat*(1-phat)/n + z2/(4*n*n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)

class WilsonWindow:
    """
    维护最近 W 个 episode 的结果，并用 Wilson 区间做“训练达标”判断。
    - outcomes: "success" / "collision" / "timeout"
    """
    def __init__(self, window: int = 512, conf: float = 0.95):
        self.win = int(window)
        self.conf = float(conf)
        self.buf: deque[Outcome] = deque(maxlen=self.win)
        self.counts = {"success":0, "collision":0, "timeout":0}

    def _recount(self):
        c = {"success":0, "collision":0, "timeout":0}
        for x in self.buf:
            c[x] += 1
        self.counts = c

    def add(self, outcome: Outcome):
        if len(self.buf) == self.buf.maxlen:
            # 将要丢弃的旧样本从计数里减掉
            old = self.buf[0]
            self.counts[old] -= 1
        self.buf.append(outcome)
        self.counts[outcome] += 1

    def add_many(self, outcomes: Iterable[Outcome]):
        for x in outcomes:
            self.add(x)

    def size(self) -> int:
        return len(self.buf)

    def rates(self) -> Dict[str, float]:
        n = self.size()
        if n == 0:
            return {"success":0.0, "collision":0.0, "timeout":0.0}
        return {k: self.counts[k]/n for k in self.counts}

    def cis(self) -> Dict[str, Tuple[float,float]]:
        n = self.size()
        return {
            "success": wilson_ci(self.counts["success"], n, self.conf),
            "collision": wilson_ci(self.counts["collision"], n, self.conf),
            "timeout": wilson_ci(self.counts["timeout"], n, self.conf),
        }

    def qualified(self,
                  targets: Dict[str, float] = None,
                  mode: str = "strict") -> Tuple[bool, Dict[str, Tuple[float,float]]]:
        """
        判定是否“训练达标”：
        - success 用 下置信界 >= 目标
        - collision/timeout 用 上置信界 <= 目标
        默认 targets:
           succ >= 0.55, coll <= 0.08, tout <= 0.35 （可按需改）
        mode="strict": 三项同时满足才 PASS
        mode="soft":   满足两项以上 PASS
        """
        if targets is None:
            targets = {"success":0.55, "collision":0.08, "timeout":0.35}
        cis = self.cis()
        checks = []
        # success: lower bound >= target
        lo_s, _ = cis["success"]
        checks.append(lo_s >= targets["success"])
        # collision: upper bound <= target
        _, hi_c = cis["collision"]
        checks.append(hi_c <= targets["collision"])
        # timeout: upper bound <= target
        _, hi_t = cis["timeout"]
        checks.append(hi_t <= targets["timeout"])

        if mode == "strict":
            ok = all(checks)
        else:
            ok = (sum(1 for x in checks if x) >= 2)
        return ok, cis
