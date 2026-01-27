# ais_comms/fraud_wrapper.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Deque, Optional
from collections import deque
import math
import random

from .datatypes import RawTxMsg, ShipId, Ts


@dataclass
class FraudHistoryEntry:
    """Phase 1/2: per-ship AIS 历史缓存（目前只用 ts/x/y，后面可扩展）"""
    ts: Ts
    x: float
    y: float


@dataclass
class FraudWrapperConfig:
    """
    欺诈层配置（Phase 3 含 mode=4）

    enable          : 总开关（False 时完全透传）
    hist_len        : 历史缓存长度（供后续高阶攻击使用）

    attack_mode     : 攻击模式
                      0 = 关闭攻击（仅记录历史）
                      1 = 仅位置偏置（x,y 加固定偏移）
                      2 = 仅航向偏置（cog 加固定偏移）
                      3 = 位置 + 航向同时偏置
                      4 = 间歇静默（按时间片让攻击船 AIS 完全消失）

    pos_bias_max_m  : 位置偏置最大半径（米），从圆盘内均匀采样
    cog_bias_max_deg: 航向偏置最大幅度（度），从 [-max, max] 均匀采样

    silence_min_dur_s : 间歇静默的最小持续时间（秒）
    silence_max_dur_s : 间歇静默的最大持续时间（秒）
                        若 <=0，则静默模式无效（即便 attack_mode=4）
    """

    enable: bool = False
    hist_len: int = 8

    attack_mode: int = 0           # 0(off), 1(pos), 2(cog), 3(pos+cog), 4(silence)
    pos_bias_max_m: float = 0.0
    cog_bias_max_deg: float = 0.0

    silence_min_dur_s: float = 0.0
    silence_max_dur_s: float = 0.0


@dataclass
class ShipBiasState:
    """对每条船采样一次的固定偏置（保证轨迹平滑、自洽）"""
    dx: float      # 位置 x 偏移量（米）
    dy: float      # 位置 y 偏移量（米）
    d_cog: float   # 航向偏移量（弧度）


@dataclass
class ShipSilenceState:
    """
    间歇静默状态：
      phase         : "on" or "off"
                      on  表示当前“可播报”；
                      off 表示当前“静默”（这段时间内所有 AIS 被吞掉）
      next_switch_ts: 下一次切换 phase 的绝对时间（sim t_true）
    """
    phase: str
    next_switch_ts: float


class FraudAgentWrapper:
    """
    AIS 欺诈封装（支持 mode 0~4）

    - 接口保持：apply(ship_id, raw_msg) -> RawTxMsg | None
      * 返回 RawTxMsg：继续走 GE + delay + ArrivalQueue
      * 返回 None    ：本次发报被“静默”，下游看不到这条报文

    - 当前模式：
      0: 关闭攻击（记录历史，透传）
      1: 固定位置偏置
      2: 固定航向偏置
      3: 固定位置+航向偏置
      4: 间歇静默（按 on/off 时间片让 AIS 消失）
    """

    def __init__(self, cfg: FraudWrapperConfig) -> None:
        self.cfg = cfg

        # 每条船的 AIS 历史：ShipId -> deque[FraudHistoryEntry]
        self._hist: Dict[ShipId, Deque[FraudHistoryEntry]] = {}

        # 每条船的偏置状态：ShipId -> ShipBiasState
        self._bias: Dict[ShipId, ShipBiasState] = {}

        # 每条船的静默状态：ShipId -> ShipSilenceState
        self._silence: Dict[ShipId, ShipSilenceState] = {}

        # 独立随机数生成器（不强依赖 AISComms 的 rng，便于独立测试）
        self._rng = random.Random()

    # ---------- Episode 生命周期 ----------

    def reset_episode(self) -> None:
        """在每个 episode 开始时调用，清空历史缓存、偏置与静默状态。"""
        self._hist.clear()
        self._bias.clear()
        self._silence.clear()

    # ---------- 核心接口：apply ----------

    def apply(self, ship_id: ShipId, raw_msg: RawTxMsg) -> Optional[RawTxMsg]:
        """
        根据 attack_mode 对 RawTxMsg 进行修改或静默。
        - 返回 RawTxMsg：正常继续发送（可包含偏置后的假值）
        - 返回 None    ：本次发报被静默（不进入信道、完全消失）
        """
        # 开关关闭：直接透传
        if not self.cfg.enable:
            return raw_msg

        # 更新历史缓存（无论是否攻击，方便后续扩展）
        self._update_history(ship_id, raw_msg)

        # mode=0：当前 episode 启用 wrapper，但不做攻击，仅做日志/历史
        if self.cfg.attack_mode == 0:
            return raw_msg

        # mode=4：间歇静默 —— 只控制“发/不发”，不改字段
        if self.cfg.attack_mode == 4:
            if self._is_silent_now(ship_id, raw_msg.tx_ts_true):
                # 调试输出可以打开
                print(f"[FraudSilence] sid={ship_id} t={raw_msg.tx_ts_true:.1f} SILENT")
                return None
            # 当前处于“on”阶段：不造假，透传
            # 若以后想在 on 段里叠加偏置，可以在此调用偏置逻辑
            print(f"[FraudSilence] sid={ship_id} t={raw_msg.tx_ts_true:.1f} ON")
            return raw_msg

        # 其他模式：固定偏置（1/2/3）
        bias = self._get_or_sample_bias(ship_id)

        x = raw_msg.x
        y = raw_msg.y
        sog = raw_msg.sog
        cog = raw_msg.cog

        # 1) 位置偏置：模式 1 或 3 且配置了 pos_bias_max_m > 0
        if self.cfg.attack_mode in (1, 3) and self.cfg.pos_bias_max_m > 0.0:
            x = float(x + bias.dx)
            y = float(y + bias.dy)

        # 2) 航向偏置：模式 2 或 3 且配置了 cog_bias_max_deg > 0
        if self.cfg.attack_mode in (2, 3) and self.cfg.cog_bias_max_deg > 0.0:
            cog = float(cog + bias.d_cog)
            # 暂不 wrap 到 [0,2π) 或 [-π,π)，保持与 AISComms 行为一致

        forged = RawTxMsg(
            msg_id=raw_msg.msg_id,
            tx_ship=raw_msg.tx_ship,
            mmsi=raw_msg.mmsi,
            tx_ts_true=raw_msg.tx_ts_true,
            x=x,
            y=y,
            sog=sog,
            cog=cog,
        )

        # 调试时可打开：
        # print(f"[FraudApply] sid={ship_id} t={raw_msg.tx_ts_true:.1f} "
        #       f"x:{raw_msg.x:.1f}->{x:.1f} y:{raw_msg.y:.1f}->{y:.1f} "
        #       f"cog:{raw_msg.cog:.3f}->{cog:.3f}")

        return forged

    # ---------- 内部工具：历史与偏置 ----------

    def _update_history(self, ship_id: ShipId, raw_msg: RawTxMsg) -> None:
        """维护每条船最近 hist_len 条 AIS 报文的 (ts, x, y)。"""
        if self.cfg.hist_len <= 0:
            return
        if ship_id not in self._hist:
            self._hist[ship_id] = deque(maxlen=self.cfg.hist_len)
        hist = self._hist[ship_id]
        hist.append(FraudHistoryEntry(
            ts=raw_msg.tx_ts_true,
            x=raw_msg.x,
            y=raw_msg.y,
        ))

    def _get_or_sample_bias(self, ship_id: ShipId) -> ShipBiasState:
        """
        为每条船采样一次静态偏置：
          - 位置偏置：从半径 <= pos_bias_max_m 的圆盘内均匀采样 (dx, dy)；
          - 航向偏置：从 [-cog_bias_max_deg, +cog_bias_max_deg] 采样，再转为弧度；
        若对应 max 为 0，则该项偏置为 0。
        """
        if ship_id in self._bias:
            return self._bias[ship_id]

        # --- 位置偏置采样 ---
        dx, dy = 0.0, 0.0
        if self.cfg.pos_bias_max_m > 0.0 and self.cfg.attack_mode in (1, 3):
            r_max = float(self.cfg.pos_bias_max_m)
            # 在圆盘内均匀采样：r ~ sqrt(U(0,1))*r_max，theta~U[0,2π)
            u = self._rng.random()
            r = (u ** 0.5) * r_max
            theta = 2.0 * math.pi * self._rng.random()
            dx = float(r * math.cos(theta))
            dy = float(r * math.sin(theta))

        # --- 航向偏置采样 ---
        d_cog = 0.0
        if self.cfg.cog_bias_max_deg > 0.0 and self.cfg.attack_mode in (2, 3):
            max_deg = float(self.cfg.cog_bias_max_deg)
            d_deg = self._rng.uniform(-max_deg, max_deg)
            d_cog = math.radians(d_deg)

        bias = ShipBiasState(dx=dx, dy=dy, d_cog=d_cog)
        self._bias[ship_id] = bias
        return bias

    # ---------- 内部工具：间歇静默状态机 ----------

    def _is_silent_now(self, ship_id: ShipId, t_true: float) -> bool:
        """
        间歇静默逻辑：
          - 若 silence_* 无效，则始终返回 False（不静默）；
          - 否则维护一个 per-ship 的 on/off 状态机：
              初始 phase='on'，持续随机 U[min,max] 秒；
              到期后切到 'off'，再持续一段随机 U[min,max]；
              然后 on/off 交替。
        """
        cfg = self.cfg
        if cfg.silence_min_dur_s <= 0.0 or cfg.silence_max_dur_s < cfg.silence_min_dur_s:
            # 参数非法或未启用静默
            return False

        st = self._silence.get(ship_id, None)

        # 初次调用：初始化为 on 阶段
        if st is None:
            dur = self._rng.uniform(cfg.silence_min_dur_s, cfg.silence_max_dur_s)
            st = ShipSilenceState(phase="on", next_switch_ts=t_true + dur)
            self._silence[ship_id] = st
            # 初始就认为在 "on"：本次不静默
            print(f"[FraudSilenceInit] sid={ship_id} t={t_true:.1f} phase=on dur={dur:.1f}")
            return False

        # 检查是否需要切换 phase
        if t_true >= st.next_switch_ts:
            # 交替 on/off
            new_phase = "off" if st.phase == "on" else "on"
            dur = self._rng.uniform(cfg.silence_min_dur_s, cfg.silence_max_dur_s)
            st.phase = new_phase
            st.next_switch_ts = t_true + dur
            print(f"[FraudSilenceSwitch] sid={ship_id} t={t_true:.1f} -> {new_phase} dur={dur:.1f}")

        return (st.phase == "off")
