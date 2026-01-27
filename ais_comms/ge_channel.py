# ais_comms/ge_channel.py
from __future__ import annotations
import numpy as np


class GEChannel:
    """
    Gilbert–Elliott 两态信道（时间一致版本）

    设计原则（非常重要）：
    ----------------------
    1) 状态机 **只在 tick(dt) 时推进**，表示“时间流逝”
    2) pass_now() **只判断当前状态是否通过**，不改变状态
    3) burst_dur_s 语义 = 秒，不随报文密度变化
    4) 每条链路一个 GEChannel 实例（由上层保证）

    若上层误用 step_pass()，本实现仍兼容，但会警告语义不推荐。
    """

    def __init__(
        self,
        rng: np.random.Generator | None = None,
        p_g2b: float = 0.01,
        p_b2g: float = 0.2,
        drop_bad: bool = True,
    ):
        self.rng = rng or np.random.default_rng(0)

        # GE 参数
        self.p_g2b = float(p_g2b)
        self.p_b2g = float(p_b2g)
        self.drop_bad = bool(drop_bad)

        # 状态
        self.state = "G"  # "G" or "B"

        # 统计
        self._bad_time = 0.0
        self._total_time = 0.0

        # ===== burst 参数 =====
        self.burst_enable = False
        self.burst_prob = 0.0          # 每秒触发概率
        self.burst_dur_s = 0.0
        self.burst_extra_drop = 0.0

        # burst 内部状态
        self._burst_left_s = 0.0

        # 最近一次 tick 的 dt（仅调试）
        self._last_dt = 0.0

    # ------------------------------------------------------------
    # 参数设置
    # ------------------------------------------------------------
    def set_params(
        self,
        *,
        p_g2b: float,
        p_b2g: float,
        drop_bad: bool,
        burst_enable: bool = False,
        burst_prob: float = 0.0,
        burst_dur_s: float = 0.0,
        burst_extra_drop: float = 0.0,
        step_dt: float | None = None,  # 为兼容旧接口，忽略
    ):
        self.p_g2b = float(p_g2b)
        self.p_b2g = float(p_b2g)
        self.drop_bad = bool(drop_bad)

        self.burst_enable = bool(burst_enable)
        self.burst_prob = float(burst_prob)
        self.burst_dur_s = float(burst_dur_s)
        self.burst_extra_drop = float(burst_extra_drop)

    # ------------------------------------------------------------
    # 状态复位
    # ------------------------------------------------------------
    def reset_metrics(self):
        self.state = "G"
        self._bad_time = 0.0
        self._total_time = 0.0
        self._burst_left_s = 0.0
        self._last_dt = 0.0

    # ------------------------------------------------------------
    # 时间推进（核心）
    # ------------------------------------------------------------
    def tick(self, dt: float):
        """
        推进信道状态 dt 秒。
        这是 GEChannel **唯一** 改变状态的地方。
        """
        dt = float(max(0.0, dt))
        if dt <= 0.0:
            return

        self._last_dt = dt

        # ---- burst 触发 ----
        if self.burst_enable and self._burst_left_s <= 0.0:
            # burst_prob 是“每秒概率”，dt 内触发概率：
            p = 1.0 - np.exp(-self.burst_prob * dt)
            if self.rng.random() < p:
                self._burst_left_s = max(0.0, self.burst_dur_s)

        # ---- GE 状态转移（按时间近似）----
        # 使用泊松近似：p = 1 - exp(-lambda * dt)
        if self.state == "G":
            p = 1.0 - np.exp(-self.p_g2b * dt)
            if self.rng.random() < p:
                self.state = "B"
        else:
            p = 1.0 - np.exp(-self.p_b2g * dt)
            if self.rng.random() < p:
                self.state = "G"

        # ---- burst 内强制坏态 ----
        if self._burst_left_s > 0.0:
            self.state = "B"
            self._burst_left_s = max(0.0, self._burst_left_s - dt)

        # ---- 统计 ----
        self._total_time += dt
        if self.state == "B":
            self._bad_time += dt

    # ------------------------------------------------------------
    # 当前是否通过（不推进状态）
    # ------------------------------------------------------------
    def pass_now(self) -> bool:
        """
        判断当前时刻该链路是否通过。
        ❗ 不推进状态 ❗
        """
        if self.state == "G":
            return True

        # B 态
        if self.drop_bad:
            return False

        # 非强制丢包：burst 内可叠加额外丢包概率
        extra = self.burst_extra_drop if self._burst_left_s > 0.0 else 0.0
        return self.rng.random() >= float(extra)

    # ------------------------------------------------------------
    # 兼容旧接口（不推荐）
    # ------------------------------------------------------------
    def step_pass(self) -> bool:
        """
        ⚠️ 兼容旧接口：相当于 tick(1.0) + pass_now()
        ⚠️ 不推荐使用，仅用于兜底
        """
        self.tick(1.0)
        return self.pass_now()

    # ------------------------------------------------------------
    # 统计接口
    # ------------------------------------------------------------
    def bad_occupancy(self) -> float:
        """坏态时间占比"""
        if self._total_time <= 0.0:
            return 0.0
        return float(self._bad_time / self._total_time)
