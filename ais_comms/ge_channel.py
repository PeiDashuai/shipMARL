# ais_comms/ge_channel.py
from __future__ import annotations
import numpy as np
import math


class GEChannel:
    """
    Gilbert–Elliott 两态信道（时间一致版本）+ 可选距离衰减

    设计原则（非常重要）：
    ----------------------
    1) 状态机 **只在 tick(dt) 时推进**，表示"时间流逝"
    2) pass_now() **只判断当前状态是否通过**，不改变状态
    3) burst_dur_s 语义 = 秒，不随报文密度变化
    4) 每条链路一个 GEChannel 实例（由上层保证）

    距离衰减模型（可选）：
    ----------------------
    - dist_enable: 是否启用距离衰减
    - dist_ref_m: 参考距离（米），低于此距离无额外丢包
    - dist_max_m: 最大通信距离（米），超出则 100% 丢包
    - dist_loss_exp: 衰减指数，控制丢包率曲线陡峭程度
      p_dist_loss = ((d - dist_ref_m) / (dist_max_m - dist_ref_m))^dist_loss_exp

    典型 AIS 参数：
    - Class A: dist_ref_m=18520 (10nm), dist_max_m=55560 (30nm), dist_loss_exp=2.0
    - Class B: dist_ref_m=9260 (5nm), dist_max_m=18520 (10nm), dist_loss_exp=2.0

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

        # ===== 距离衰减参数 =====
        self.dist_enable = False
        self.dist_ref_m = 18520.0       # 10 nautical miles in meters
        self.dist_max_m = 55560.0       # 30 nautical miles in meters
        self.dist_loss_exp = 2.0        # 衰减指数（2.0 = 平方衰减）

        # 距离统计
        self._dist_drop_count = 0
        self._dist_total_count = 0

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
        dist_enable: bool = False,
        dist_ref_m: float = 18520.0,
        dist_max_m: float = 55560.0,
        dist_loss_exp: float = 2.0,
        step_dt: float | None = None,  # 为兼容旧接口，忽略
    ):
        self.p_g2b = float(p_g2b)
        self.p_b2g = float(p_b2g)
        self.drop_bad = bool(drop_bad)

        self.burst_enable = bool(burst_enable)
        self.burst_prob = float(burst_prob)
        self.burst_dur_s = float(burst_dur_s)
        self.burst_extra_drop = float(burst_extra_drop)

        self.dist_enable = bool(dist_enable)
        self.dist_ref_m = float(dist_ref_m)
        self.dist_max_m = float(dist_max_m)
        self.dist_loss_exp = float(dist_loss_exp)

    # ------------------------------------------------------------
    # 状态复位
    # ------------------------------------------------------------
    def reset_metrics(self):
        self.state = "G"
        self._bad_time = 0.0
        self._total_time = 0.0
        self._burst_left_s = 0.0
        self._last_dt = 0.0
        self._dist_drop_count = 0
        self._dist_total_count = 0

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
    def pass_now(self, distance_m: float | None = None) -> bool:
        """
        判断当前时刻该链路是否通过。
        ❗ 不推进状态 ❗

        Args:
            distance_m: 发送端与接收端之间的距离（米）。
                        若启用距离衰减且提供此参数，将额外计算距离丢包。

        Returns:
            True 表示该报文通过信道，False 表示被丢弃。
        """
        # 1. 先检查距离衰减（如果启用）
        if self.dist_enable and distance_m is not None:
            self._dist_total_count += 1
            d = float(distance_m)
            if d >= self.dist_max_m:
                # 超出最大范围，100% 丢包
                self._dist_drop_count += 1
                return False
            elif d > self.dist_ref_m:
                # 在参考距离和最大距离之间，按指数衰减丢包
                ratio = (d - self.dist_ref_m) / max(1.0, self.dist_max_m - self.dist_ref_m)
                p_loss = math.pow(ratio, self.dist_loss_exp)
                if self.rng.random() < p_loss:
                    self._dist_drop_count += 1
                    return False

        # 2. GE 信道状态判断
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

    def dist_loss_rate(self) -> float:
        """距离衰减导致的丢包率"""
        if self._dist_total_count <= 0:
            return 0.0
        return float(self._dist_drop_count / self._dist_total_count)

    def dist_stats(self) -> dict:
        """距离衰减统计"""
        return {
            "enable": self.dist_enable,
            "ref_m": self.dist_ref_m,
            "max_m": self.dist_max_m,
            "loss_exp": self.dist_loss_exp,
            "drop_count": self._dist_drop_count,
            "total_count": self._dist_total_count,
            "loss_rate": self.dist_loss_rate(),
        }
