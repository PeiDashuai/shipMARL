# ais_comms/track_manager.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import math

from .delay_robust_ctrv_ekf import DelayRobustCTRV_EKF, DelayRobustCTRVConfig
from .ais_preproc import AISPreprocessor, SimpleMeas
from .datatypes import TrueState, ShipId, AgentId, Ts, RxMsg


@dataclass
class Track:
    ekf: DelayRobustCTRV_EKF
    # 最近一次“被用来更新 EKF 的量测”的时间戳（arrival_time）
    last_meas_ts: float
    valid: bool = True


class AISTrackManager:
    """
    AIS 轨迹管理（延迟鲁棒 CTRV-EKF）：

    - AISPreprocessor 完成 gating + 低通 + 初始化均值；
    - 这里只做：
        * mmsi ↔ ship_id 映射；
        * EKF init / step_delay_robust；
        * all_estimates(t) 时 CTRV 预测到 t；
        * 按 max_age 丢弃过旧轨迹。
    - 默认完全 AIS-only，但 all_estimates 支持可选 true_states fallback。
    """

class AISTrackManager:
    def __init__(
        self,
        max_age: float = 20.0,
        window_sec: float = 60.0,
        max_buffer: int = 256,
        max_pos_jump: float = 80.0,
        max_cog_jump_deg: float = 90.0,
        init_avg_K: int = 3,
        lowpass_alpha: float = 0.4,
        process_std_a: float = 0.2,
        process_std_yaw: float = math.radians(2.0),
        meas_std_pos: float = 5.0,
        meas_std_sog: float = 0.5,
        meas_std_cog_deg: float = 1.0,
        nis_thresh: Optional[float] = 25.0,
        age_fresh: float = 1.0,
        age_stale: float = 3.0,
        age_very_stale: float = 5.0,
        age_max: float = 8.0,
        scale_fresh: float = 1.0,
        scale_stale: float = 4.0,
        scale_very_stale: float = 10.0,
    ):
        self.max_age = float(max_age)

        # ✅ 这里改成用 base_pos_jump，而不是不存在的 max_pos_jump
        self.preproc = AISPreprocessor(
            window_sec=window_sec,
            max_buffer=max_buffer,
            base_pos_jump=max_pos_jump,
            max_cog_jump_deg=max_cog_jump_deg,
            init_avg_K=init_avg_K,
            lowpass_alpha=lowpass_alpha,
        )

        self.cfg = DelayRobustCTRVConfig(
            process_std_a=float(process_std_a),
            process_std_yaw=float(process_std_yaw),
            meas_std_pos=float(meas_std_pos),
            meas_std_sog=float(meas_std_sog),
            meas_std_cog_deg=float(meas_std_cog_deg),
            nis_thresh=nis_thresh,
            age_fresh=float(age_fresh),
            age_stale=float(age_stale),
            age_very_stale=float(age_very_stale),
            age_max=float(age_max),
            scale_fresh=float(scale_fresh),
            scale_stale=float(scale_stale),
            scale_very_stale=float(scale_very_stale),
        )

        self.tracks: Dict[int, Track] = {}
        self.mmsi_to_sid: Dict[int, ShipId] = {}
        self.t0: float = 0.0
        self.last_t: float = 0.0


    # ------------------------------------------------------------------
    # 映射 & reset
    # ------------------------------------------------------------------
    def set_mmsi_map(self, mmsi_of_ship: Dict[ShipId, int]):
        """从 AISCommsSim 传入 ship_id -> mmsi 映射，内部构造 mmsi -> ship_id。"""
        self.mmsi_to_sid = {int(m): int(sid) for sid, m in mmsi_of_ship.items()}

    def reset(self, ship_ids, t0, init_states=None):
        """
        episode 开始：清空轨迹 & 预处理器。
        这里不使用真值初始化 EKF，EKF 只会从 AIS 量测初始化。
        """
        self.tracks.clear()
        self.preproc.reset()
        self.t0 = float(t0)
        self.last_t = float(t0)

    # ------------------------------------------------------------------
    # ingest: 接收 AISCommsSim.step() 输出的 RxMsg
    # ------------------------------------------------------------------
    def ingest(self, t: Ts, ready: Dict[AgentId, List[RxMsg]]):
        """
        把 RxMsg 丢进 AISPreprocessor，再把“清洗后量测”送给延迟鲁棒 EKF。

        t: 当前 env 时间，用于计算 measurement age:
           age = t_env - arrival_time
        """
        self.last_t = float(t)

        for msgs in ready.values():
            for r in msgs:
                sid = self.mmsi_to_sid.get(int(r.mmsi), None)
                if sid is None:
                    continue

                meas_in = SimpleMeas(
                    ship_id=int(sid),
                    x=float(r.reported_x),
                    y=float(r.reported_y),
                    sog_mps=float(r.reported_sog),
                    cog_rad=float(r.reported_cog),
                    arrival_time=float(r.arrival_time),
                )

                cleaned_list = self.preproc.ingest(meas_in)
                for cm in cleaned_list:
                    age = max(0.0, self.last_t - float(cm.arrival_time))
                    self._update_with_meas(cm, age)

    # ------------------------------------------------------------------
    # 内部：用一条清洗后的量测更新对应 ship 的 EKF
    # ------------------------------------------------------------------
    def _update_with_meas(self, m: SimpleMeas, age: float):
        sid = int(m.ship_id)
        ts = float(m.arrival_time)

        meas_dict = {
            "x": float(m.x),
            "y": float(m.y),
            "sog": float(m.sog_mps),
            "cog": float(m.cog_rad),
        }

        tr = self.tracks.get(sid, None)

        # 首次量测：初始化 EKF
        if tr is None:
            ekf = DelayRobustCTRV_EKF(self.cfg)
            ekf.init_from_meas(ts, meas_dict)
            self.tracks[sid] = Track(ekf=ekf, last_meas_ts=ts, valid=True)
            return

        # 后续量测：调用延迟鲁棒更新
        tr.ekf.step_delay_robust(ts, meas_dict, float(age))
        tr.last_meas_ts = ts
        tr.valid = True

    # ------------------------------------------------------------------
    # 查询接口：给上层环境 / GNN 使用
    # ------------------------------------------------------------------
    def all_estimates(
        self,
        t: Ts,
        true_states: Dict[ShipId, TrueState] | None = None,
    ) -> Dict[ShipId, TrueState]:
        """
        在时间 t 上返回所有 ship 的 EKF 估计 TrueState（CTRV 外推到 t）。

        - 若 track_age > max_age，则认为轨迹失效；
        - 若提供 true_states 且某条 ship 没有有效轨迹，则回退到真值（可选）。
        """
        t = float(t)
        out: Dict[ShipId, TrueState] = {}

        for sid, tr in list(self.tracks.items()):
            if not tr.valid:
                continue

            track_age = t - float(tr.last_meas_ts)
            if track_age > self.max_age:
                tr.valid = False
                continue

            ekf = tr.ekf

            # 把 EKF 状态外推到当前时间 t
            if ekf.last_ts is not None and t > ekf.last_ts:
                dt = t - ekf.last_ts
                if dt > 0.0:
                    ekf.predict(dt)
                    ekf.last_ts = t

            # ✅ 这里改为 4 维解包
            px, py, v, yaw = ekf.x
            vx = v * math.cos(yaw)
            vy = v * math.sin(yaw)

            out[int(sid)] = TrueState(
                ship_id=int(sid),
                x=float(px),
                y=float(py),
                vx=float(vx),
                vy=float(vy),
            )

        # 可选：对没有轨迹的船回退到真值（用于 RL 训练更稳定）
        if true_states is not None:
            for sid, ts in true_states.items():
                if sid not in out:
                    out[sid] = ts

        return out

    # ------------------------------------------------------------------
    # 调试接口（保留）
    # ------------------------------------------------------------------
    def get_latest(self, sid: int):
        sid = int(sid)
        tr = self.tracks.get(sid, None)
        if tr is None or not tr.valid:
            return None
        return tr.ekf.x.copy()

    def get_estimate(self, sid: int, ts: float):
        sid = int(sid)
        tr = self.tracks.get(sid, None)
        if tr is None or not tr.valid:
            return None

        ekf = tr.ekf
        if ekf.last_ts is None:
            return None

        dt = float(ts - ekf.last_ts)
        if dt > 0.0:
            ekf.predict(dt)
            ekf.last_ts = float(ts)

        return ekf.x.copy()
