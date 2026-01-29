# ais_comms/scheduler.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class _PerShip:
    next_tx_time: float = 0.0
    last_period_s: float = 0.0
    dbg_tx_prints: int = 0          # ← ADD: 每船调试打印计数
    ais_class: str = "A"            # AIS Class: "A" (fast) or "B" (slow)

class TxScheduler:
    """
    AIS 发报调度器（兼容旧接口，增强如下）：
      - 不规则 Tx：按 SOG 分档确定基础周期 + 每次触发重采样抖动
      - 抖动：period = base * U(1-j, 1+j)，并带最小分数下限避免过密
      - 分档解析更健壮：支持
          1) len(sog_breaks) == len(sog_periods)   （你当前用法）
          2) len(sog_breaks) == len(sog_periods)+1 （区间端点法）
      - 低速/漂泊丢发：兼容原布尔 drop_on_idle，同时可选
          drop_prob_on_idle: float in [0,1]，sog_idle_thr_knots: float
      - 初相位（可选）：默认保持旧行为（t0 即可发）；如需异步相位可传 randomize_phase=True
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng(0)
        # === 旧参数（保持默认）===
        self.base_period: float = 2.0
        self.jitter_frac: float = 0.1
        self.sog_breaks: List[float] = []      # 旧：与 sog_periods 等长；新：也可比其多 1
        self.sog_periods: List[float] = []     # 每档绝对周期（秒）
        self.drop_on_idle: bool = False        # 旧布尔开关

        # === 新增可选参数（不传即不改变旧行为）===
        self.min_frac_interval: float = 0.05   # 抖动后最小间隔= min_frac * period
        self.randomize_phase: bool = False     # True → reset 时加初相位
        self.randomize_phase_span: float = 1.0 # 初相位范围系数，默认 U(0, 1*T)
        self.drop_prob_on_idle: float = 0.0    # 低速时丢发概率，默认 0（等价 "不启用"）
        self.sog_idle_thr_mps: float = 0.0
        self._ships: Dict[int, _PerShip] = {}

        # === AIS Class A/B differentiation ===
        # Class B has longer reporting intervals (IMO: 30s-3min vs Class A: 2-10s)
        # class_b_period_mult: multiplier applied to period for Class B ships
        self.class_b_period_mult: float = 15.0  # Default: Class B = 15x slower
        self._ship_classes: Dict[int, str] = {}  # ship_id -> "A" or "B"

        # ---- ADD: 调试开关（默认 False，不影响原逻辑）----
        self.debug_tx: bool = True
        self.debug_tx_max: int = 12    # 每船最多打印 N 次
        
    # -------- 配置入口（兼容旧调用；新增键可选）--------

    # -------- 新增：返回 sog 所在分档 idx（与 _period_for_sog 同口径） -------
    def _bin_index_for_sog(self, sog: float) -> int:
        if not self.sog_breaks or not self.sog_periods:
            return 0
        brk = self.sog_breaks
        per = self.sog_periods
        k = len(per)
        idx = 0
        while idx < len(brk) and sog > brk[idx]:
            idx += 1
        if idx >= k:
            idx = k - 1
        return idx

    def set_params(
        self, *,
        base_period: float,
        jitter_frac: float,
        sog_breaks: List[float],
        sog_periods: List[float],
        drop_on_idle: bool,
        # 新增可选键（全部有缺省，不传不影响旧行为）
        min_frac_interval: float = 0.05,
        randomize_phase: bool = False,
        randomize_phase_span: float = 1.0,
        drop_prob_on_idle: float = 0.0,
        sog_idle_thr_mps: float = 0.0,
        sog_idle_thr_knots: float = 0.0,
        class_b_period_mult: float = 15.0,
    ):
        self.base_period = float(base_period)
        self.jitter_frac = float(jitter_frac)
        self.sog_breaks  = list(sog_breaks or [])
        self.sog_periods = list(sog_periods or [])
        self.drop_on_idle = bool(drop_on_idle)

        # 新增可选项（安全夹取）
        self.min_frac_interval = float(max(0.0, min(0.5, min_frac_interval)))
        self.randomize_phase   = bool(randomize_phase)
        self.randomize_phase_span = float(max(0.0, randomize_phase_span))
        self.drop_prob_on_idle = float(max(0.0, min(1.0, drop_prob_on_idle)))
        self.sog_idle_thr_mps  = float(max(0.0, sog_idle_thr_mps))
        #self.sog_idle_thr_mps: float = 0.0     # 低速阈值（m/s），默认 0（与旧逻辑等价）
        self.class_b_period_mult = float(max(1.0, class_b_period_mult))

    # -------- 复位每艘船（旧接口不变）--------
    def reset_ship(
        self,
        ship_id: int,
        t0: float,
        initial_sog_knots: Optional[float] = None,
        ais_class: str = "A",
    ):
        """
        旧行为：next_tx_time = t0（立刻可发）
        新增：若 randomize_phase=True，则 next_tx_time = t0 + U(0, span*T0)
        T0 来自 initial_sog_knots 对应档位（若未提供则退化到 base_period）

        Args:
            ship_id: Ship identifier
            t0: Initial time
            initial_sog_knots: Initial SOG (optional)
            ais_class: "A" (Class A, fast reporting) or "B" (Class B, slow reporting)
        """
        ais_class = str(ais_class).upper()
        if ais_class not in ("A", "B"):
            ais_class = "A"
        self._ship_classes[ship_id] = ais_class

        st = _PerShip(next_tx_time=float(t0), ais_class=ais_class)
        # 记录一个初始 period（用于统计/初相位）
        sog0 = float(initial_sog_knots) if (initial_sog_knots is not None) else 0.0
        T0 = self._period_for_sog(sog0, ais_class=ais_class)
        st.last_period_s = float(T0)

        if self.randomize_phase:
            phase_span = max(0.0, self.randomize_phase_span)
            # 缺省：U(0, 1*T0)；也可以通过 randomize_phase_span 放大/缩小
            dt0 = T0 * float(self.rng.uniform(0.0, max(1e-9, phase_span)))
            st.next_tx_time = float(t0 + dt0)

        self._ships[ship_id] = st

    # -------- 主判定：本步是否应发报（旧接口不变）--------
    def should_tx(self, ship_id: int, t: float, sog: float) -> bool:
        st = self._ships.get(ship_id)
        if st is None:
            # 向后兼容：没 reset 也不崩，按 t0=t 初始化（旧行为）
            self.reset_ship(ship_id, t0=float(t), initial_sog_knots=sog)
            st = self._ships[ship_id]

        # 低速/漂泊处理（兼容旧布尔 + 新概率/阈值）
        if (sog is None):
            sog_val = 0.0
        else:
            try:
                sog_val = float(sog)
            except Exception:
                sog_val = 0.0

        # Get ship's AIS class
        ship_class = st.ais_class if hasattr(st, "ais_class") else self._ship_classes.get(ship_id, "A")

        # 仅当启用了"旧 drop_on_idle==True 或 新阈值>0/概率>0"才考虑丢发
        if self._is_idle(sog_val) and (self.drop_on_idle or self.drop_prob_on_idle > 0.0):
            # 若尚未到触发时刻，也直接返回 False（不影响闹钟）
            if float(t) + 1e-9 < st.next_tx_time:
                return False
            # 命中"时间闹钟"但处于 idle：按概率决定是否丢发；无论发不发都滚动下一次
            period = self._period_for_sog(sog_val, ais_class=ship_class)
            st.last_period_s = period
            #st.next_tx_time = float(t + self._sample_period(period))
            sampled = self._sample_period(period)
            st.next_tx_time = float(t + sampled)

            if self.debug_tx and st.dbg_tx_prints < self.debug_tx_max:
                idx = self._bin_index_for_sog(sog_val)
                try:
                    dbg_breaks = ",".join(f"{b:.2f}" for b in (self.sog_breaks or []))
                    dbg_periods = ",".join(f"{p:.2f}" for p in (self.sog_periods or []))
                except Exception:
                    dbg_breaks, dbg_periods = str(self.sog_breaks), str(self.sog_periods)
                #print(f"[TxDBG] ship={ship_id} t={t:.1f}s sog={sog_val:.3f} m/s "f"(IDLE) -> bin#{idx} base={period:.2f}s jitter={self.jitter_frac:.3f} "f"sampled_dt={sampled:.2f}s breaks=[{dbg_breaks}] periods=[{dbg_periods}]")
                st.dbg_tx_prints += 1

            if self.drop_prob_on_idle <= 0.0:
                # 旧行为：只要 idle 就不发
                return False
            else:
                # 新行为：按概率丢；返回 True 代表“进入发报通道”，上游可继续做链路/碰撞等
                drop = (self.rng.random() < self.drop_prob_on_idle)
                return (not drop)

        # 非 idle：常规调度
        if float(t) + 1e-9 < st.next_tx_time:
            return False

        period = self._period_for_sog(sog_val, ais_class=ship_class)
        st.last_period_s = period
        sampled = self._sample_period(period)
        st.next_tx_time = float(t + sampled)
        # ---- 调试打印（限次 & 可关）----
        if self.debug_tx and st.dbg_tx_prints < self.debug_tx_max:
            idx = self._bin_index_for_sog(sog_val)
            try:
                dbg_breaks = ",".join(f"{b:.2f}" for b in (self.sog_breaks or []))
                dbg_periods = ",".join(f"{p:.2f}" for p in (self.sog_periods or []))
            except Exception:
                dbg_breaks, dbg_periods = str(self.sog_breaks), str(self.sog_periods)
#            print(
#                f"[TxDBG] ship={ship_id} t={t:.1f}s sog={sog_val:.3f} m/s "
 #               f"-> bin#{idx} base={period:.2f}s jitter={self.jitter_frac:.3f} "
  #              f"sampled_dt={sampled:.2f}s breaks=[{dbg_breaks}] periods=[{dbg_periods}]"
   #         )
            st.dbg_tx_prints += 1
        return True

    # ===== 内部工具 =====

    def _is_idle(self, sog_mps: float) -> bool:
        """
        旧逻辑：drop_on_idle=True 时，只要 sog 很小就不发（阈值近 0）
        新逻辑：若配置了 sog_idle_thr_knots>0，则使用显式阈值
        """
        thr = self.sog_idle_thr_mps if self.sog_idle_thr_mps > 0.0 else 1e-6
        return sog_mps <= thr

    def _sample_period(self, base: float) -> float:
        j = max(0.0, min(0.99, self.jitter_frac))
        lo, hi = (1.0 - j), (1.0 + j)
        dt = float(base * self.rng.uniform(lo, hi))
        # 安全下限：避免抖动过小导致连发
        dt = max(self.min_frac_interval * float(base), dt)
        return dt

    def _period_for_sog(self, sog: float, ais_class: str = "A") -> float:
        """
        返回该 SOG 档位的绝对周期（秒）。
        兼容两种配置：
          A) len(breaks) == len(periods)   → periods[i] 对应 (-inf,b0], (b0,b1], ..., (>b_{k-1})
          B) len(breaks) == len(periods)+1 → periods[i] 对应 [b_i, b_{i+1})
        若未配置，回退 base_period。

        For AIS Class B ships, the period is multiplied by class_b_period_mult.
        """
        base = float(self.base_period)

        if self.sog_periods:
            breaks = self.sog_breaks or []
            periods = self.sog_periods

            # A) 旧用法：长度相等
            if len(breaks) == len(periods):
                idx = 0
                while idx < len(breaks) and sog > breaks[idx]:
                    idx += 1
                if idx >= len(periods):
                    idx = len(periods) - 1
                base = float(periods[idx])

            # B) 区间端点法：长度多 1
            elif len(breaks) == len(periods) + 1:
                # 在 [b_i, b_{i+1}) 里找
                for i in range(len(periods)):
                    if sog >= breaks[i] and sog < breaks[i + 1]:
                        base = float(periods[i])
                        break
                else:
                    # 右侧溢出 → 用最后一档
                    base = float(periods[-1])

        # Apply Class B multiplier
        if str(ais_class).upper() == "B":
            base *= self.class_b_period_mult

        return base

    def reseed(self, seed: int):
        self.rng = np.random.default_rng(int(seed))

    # ===== AIS Class A/B Methods =====
    def set_ship_class(self, ship_id: int, ais_class: str):
        """Set the AIS class for a specific ship."""
        ais_class = str(ais_class).upper()
        if ais_class not in ("A", "B"):
            ais_class = "A"
        self._ship_classes[ship_id] = ais_class
        if ship_id in self._ships:
            self._ships[ship_id].ais_class = ais_class

    def get_ship_class(self, ship_id: int) -> str:
        """Get the AIS class for a specific ship."""
        return self._ship_classes.get(ship_id, "A")

    def class_distribution(self) -> dict:
        """Get distribution of AIS classes."""
        a_count = sum(1 for c in self._ship_classes.values() if c == "A")
        b_count = sum(1 for c in self._ship_classes.values() if c == "B")
        total = a_count + b_count
        return {
            "class_a_count": a_count,
            "class_b_count": b_count,
            "class_a_ratio": a_count / total if total > 0 else 0.0,
            "class_b_ratio": b_count / total if total > 0 else 0.0,
        }
