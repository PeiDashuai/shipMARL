
from __future__ import annotations


"""
================================================================================
AISPreprocessor — TIME/ANGLE SEMANTICS CONTRACT (MUST MATCH track_manager_pf.py)
================================================================================

This module MUST remain semantically consistent with:
  - ais_comms/track_manager_pf.py  (single-point COG conversion; PF uses arrival-time axis)

--------------------------------
0) Time axis conventions (DO NOT MIX)
--------------------------------

We carry two timestamps on each AIS measurement:

(1) reported_ts  [message timestamp / Tx-time]
    - Name: reported_ts
    - Unit: seconds
    - Meaning: the timestamp contained in the AIS message (when it was "reported" by sender)
    - Role in THIS module:
        * used for motion-consistent gating: dt_rep = rep_now - rep_last
        * used for gap reset decisions (primary)
        * NEVER modified in this module

(2) arrival_time [delivery time / Rx-time]
    - Name: arrival_time
    - Unit: seconds
    - Meaning: the simulated time when the message arrives at receiver after channel delay/loss
    - Role in THIS module:
        * used ONLY for buffering/sorting and detecting out-of-order arrivals (dt_arr < 0)
        * NEVER used as the main "physics dt" for gating
        * NEVER modified in this module

Hard rule:
  - This module must NEVER create a new time axis or overwrite reported_ts/arrival_time.
  - It must NEVER "project" x/y to another time (projection happens later in TrackManagerPF).

Angle conventions (MUST MATCH track_manager_pf.py)

(A1) Incoming raw COG direction (RxMsg.reported_cog):
    - Meaning: yaw_east_ccw_rad == atan2(vy, vx)
      0 rad along +X (EAST), positive COUNTER-CLOCKWISE
    - Unit: radians, range (-pi, pi]

(A2) Internal yaw (used by preproc + PF):
    - Same convention as (A1)
    - In this module, SimpleMeas.cog_rad is a LEGACY FIELD NAME,
      but SEMANTICS ARE yaw_sim_rad (ENU, +X=0, CCW+).

Single-point rule:
    - TrackManagerPF performs the ONLY normalization:
          yaw_sim_rad = wrap_pi(raw_cog_rad)
    - This module may wrap angles to (-pi, pi], but must not change conventions.
    - This module must not compute or inject any North/CW nautical COG for PF input.

Time conventions:
    - reported_ts: message timestamp (Tx-time), used for gating dt in this module
    - arrival_time: delivery time (Rx-time), used only for buffering/sorting/out-of-order detection
    - PF time axis is t_env (environment time); projection happens later in TrackManagerPF.


--------------------------------
2) Variable name policy (avoid semantic drift)
--------------------------------

- Field name "cog_rad" is legacy; treat it as yaw_sim_rad everywhere in this file.
- All debug prints must label it as "yaw_sim" (never as "cog" alone).

--------------------------------
3) What this module is allowed to do
--------------------------------

Allowed:
  - buffer/sort by arrival_time for debugging/cleanup
  - stale detection:
      * drop if arrival_time goes backwards (dt_arr < 0)
      * drop if reported_ts is non-increasing (dt_rep <= 0)
  - gating using dt_rep (position/sog/yaw jump checks)
  - lowpass filtering of (sog, yaw) with angle-safe averaging
  - init policy handling: avg / passthrough / warmstart_ok

Not allowed:
  - any COG convention conversion
  - any coordinate transform
  - any forward projection of x/y across time (done in TrackManagerPF)
  - modifying reported_ts / arrival_time values

================================================================================
"""


from dataclasses import dataclass
from typing import Dict, List
import math
import numpy as np
import os
import zlib

@dataclass
class SimpleMeas:
    """
    TrackManager 用的简化量测格式（与 track_manager_pf.py 语义完全对齐）.

    时间语义：
      - reported_ts: AIS 报文自身的时间戳（Tx-time / message timestamp），单位 s
      - arrival_time: 报文到达接收端的仿真时间（network delivery time），单位 s
        *在预处理里 arrival_time 仅作为排序/缓存/乱序检测的元数据，不作为滤波时间轴*

    角度语义（硬规则）：
      - cog_rad: **内部 yaw_sim_rad**（+x=0, CCW+，atan2(vy,vx) 语义），单位 rad，范围建议 (-pi, pi]
        注意：此字段名是历史遗留，语义上等同于 yaw_sim_rad。
      - 本预处理模块不做任何坐标系/角度约定转换。
        AIS NorthCW COG -> yaw_sim_rad 的唯一转换点在 TrackManagerPF 中完成。
    """
    ship_id: int
    x: float
    y: float
    sog_mps: float
    cog_rad: float           # legacy name; semantics: yaw_sim_rad
    arrival_time: float
    reported_ts: float = 0.0
    msg_id: str = ""

class AISPreprocessor:
    """
    AIS 报文预处理：
      - 按 ship_id 分桶缓存；
      - 位置 / SOG / yaw(legacy:cog_rad) 跳变 gating（带 time gap 自适应）；
      - 一阶 IIR 低通滤波；
      - 初始化策略可选：avg / passthrough / warmstart_ok

    重要语义对齐（与 TrackManagerPF）：
      - 本模块不做 AIS COG 角度约定转换，cog_rad 仅作为 yaw_sim_rad 使用；
      - jump gating 的时间间隔使用 reported_ts（更贴近物理运动），arrival_time 只用于排序与乱序检测；
      - 不改变 reported_ts / arrival_time 的数值（只做 pass-through）。
    """

    def __init__(
        self,
        window_sec: float = 60.0,
        max_buffer: int = 256,

        base_pos_jump: float = 20.0,
        dyn_jump_k: float = 3.0,
        v_floor_mps: float = 0.2,

        max_sog_jump_mps: float = 3.0,
        max_cog_jump_deg: float = 90.0,   # legacy name; semantics: yaw jump

        gap_reset_sec: float = 15.0,

        init_avg_K: int = 1,
        lowpass_alpha: float = 1.0,

        max_consec_drop: int = 5,

        init_policy: str = "passthrough",  # "avg" | "passthrough" | "warmstart_ok"
        debug: bool = False,
    ):
        self.window_sec = float(window_sec)
        self.max_buffer = int(max_buffer)

        self.base_pos_jump = float(base_pos_jump)
        self.dyn_jump_k = float(dyn_jump_k)
        self.v_floor_mps = float(v_floor_mps)

        self.max_sog_jump_mps = float(max_sog_jump_mps)
        self.max_cog_jump_rad = math.radians(max_cog_jump_deg)
        self.gap_reset_sec = float(gap_reset_sec)

        self.init_avg_K = int(init_avg_K)
        self.lowpass_alpha = float(lowpass_alpha)

        self.max_consec_drop = int(max_consec_drop)
        self.init_policy = str(init_policy)
        self.debug = bool(debug)

        # ship_id -> List[SimpleMeas]
        self.buffers: Dict[int, List[SimpleMeas]] = {}
        self.init_buffers: Dict[int, List[SimpleMeas]] = {}
        self.inited: Dict[int, bool] = {}
        self.last_acc: Dict[int, SimpleMeas] = {}
        self.last_raw: Dict[int, SimpleMeas] = {}
        self.lp_state: Dict[int, dict] = {}
        self.drop_streak: Dict[int, int] = {}
        self.warm_started: Dict[int, bool] = {}

    # ---------------- debug helpers ----------------
    def _dbg(self, msg: str):
        if self.debug or os.environ.get("PP_DEBUG", "0") == "1":
            print(msg)

    def _dbg_detail(self, msg: str):
        if os.environ.get("PP_DEBUG_DETAIL", "0") == "1":
            print(msg)



    # ---------------- msg_id helpers ----------------
    def _norm_mid(self, mid) -> str:
        """Normalize msg_id to stable, non-null string (or empty)."""
        if mid is None:
            return ""
        s = str(mid).strip()
        if (not s) or (s.lower() in ("none", "null")):
            return ""
        return s

    def _blend_mid(self, mids: list[str]) -> str:
        """
        Stable, order-independent id for a fused output (e.g., avg-init K samples).
        This prevents non-determinism when arrival order changes under delay/replay.
        """
        xs = [self._norm_mid(x) for x in mids]
        xs = [x for x in xs if x]
        if not xs:
            return ""
        xs = sorted(set(xs))
        s = "|".join(xs)
        return f"blend:{zlib.crc32(s.encode('utf-8')):08x}"



    @staticmethod
    def _wrap_pi(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _rad2deg(a: float) -> float:
        return float(a) * 180.0 / math.pi

    # ---------------- drop/accept reason log ----------------
    def _pp_on(self) -> bool:
        return (self.debug or os.environ.get("PP_DEBUG", "0") == "1" or os.environ.get("PP_DEBUG_DETAIL", "0") == "1")

    def _fmt_meas(self, m: SimpleMeas) -> str:
        try:
            # 重要：此处 cog_rad 按 yaw_sim 打印，避免语义漂移
            return (f"sid={int(m.ship_id)} rep={float(m.reported_ts):.2f} arr={float(m.arrival_time):.2f} "
                    f"pos=({float(m.x):.1f},{float(m.y):.1f}) sog={float(m.sog_mps):.2f} "
                    f"yaw_sim(rad)={float(m.cog_rad):+.3f}")
        except Exception:
            return f"sid={getattr(m,'ship_id',None)}"

    def _dbg_drop(self, m: SimpleMeas, reason: str, *, last: SimpleMeas | None = None, extra: str = ""):
        if not self._pp_on():
            return
        msg = f"[PP-DROP] {self._fmt_meas(m)} reason={reason}"
        if extra:
            msg += f" {extra}"
        print(msg)
        if os.environ.get("PP_DEBUG_DETAIL", "0") == "1" and last is not None:
            print(f"  last: {self._fmt_meas(last)}")

    def _dbg_accept(self, m_in: SimpleMeas, m_out: SimpleMeas, stage: str, *, extra: str = ""):
        if os.environ.get("PP_DEBUG_DETAIL", "0") != "1":
            return
        msg = f"[PP-OK:{stage}] in({self._fmt_meas(m_in)}) -> out({self._fmt_meas(m_out)})"
        if extra:
            msg += f" {extra}"
        print(msg)

    # ------------------------------------------------------------
    def reset(self):
        self.buffers.clear()
        self.init_buffers.clear()
        self.inited.clear()
        self.last_acc.clear()
        self.lp_state.clear()
        self.drop_streak.clear()
        self.warm_started.clear()
        self.last_raw.clear()

    def warm_start(self, sid: int, meas: SimpleMeas, *, mark_inited: bool | None = None):
        """
        预处理 warm_start：只做类型与范围归一（wrap），不做角度语义转换。
        meas.cog_rad 必须已经是 yaw_sim_rad（由 TrackManagerPF 单点转换保证）。
        """
        sid = int(sid)

        meas = SimpleMeas(
            ship_id=int(meas.ship_id),
            x=float(meas.x),
            y=float(meas.y),
            sog_mps=float(meas.sog_mps),
            cog_rad=float(self._wrap_pi(float(meas.cog_rad))),   # 仅 wrap，不转换语义
            arrival_time=float(meas.arrival_time),
            reported_ts=float(meas.reported_ts),
            msg_id=self._norm_mid(getattr(meas, "msg_id", "")),
        )

        self.last_acc[sid] = meas
        self.drop_streak[sid] = 0
        self.warm_started[sid] = True
        self.last_raw[sid] = meas

        if mark_inited is None:
            mark_inited = (self.init_policy != "warmstart_ok")

        if mark_inited:
            self.inited[sid] = True
            self.init_buffers[sid] = []

        self._dbg(
            f"[PP-WARM] sid={sid} mark_inited={mark_inited} "
            f"arr={meas.arrival_time:.2f} rep={meas.reported_ts:.2f} "
            f"pos=({meas.x:.1f},{meas.y:.1f}) sog={meas.sog_mps:.2f} "
            f"yaw_sim={float(meas.cog_rad):+.3f}rad({self._rad2deg(float(meas.cog_rad)):+.1f}deg)"
        )

    def _reset_ship_state(self, sid: int):
        self.init_buffers.pop(sid, None)
        self.inited[sid] = False
        self.last_acc.pop(sid, None)
        self.lp_state.pop(sid, None)
        self.last_raw.pop(sid, None)
        self.drop_streak[sid] = 0
        self.warm_started.pop(sid, None)
        self._dbg(f"[PP-RESET] sid={sid}")

    # ------------------------------------------------------------
    def ingest(self, m: SimpleMeas) -> List[SimpleMeas]:
        """
        输入约束（与 TrackManagerPF 对齐）：
          - m.cog_rad 是 yaw_sim_rad（内部 yaw），本模块不做任何角度变换。
          - reported_ts / arrival_time 仅透传，不改值。
        """
        # >>> DEBUG PROBE 2: 预处理入口 <<<
        #if m.ship_id == 1:
            #yaw_wrapped = self._wrap_pi(float(m.cog_rad))
            #print(f"[PROBE-PP] sid=1 rep={float(m.reported_ts):.3f} arr={float(m.arrival_time):.3f} "
                #f"raw_yaw={float(m.cog_rad):+.4f} wrapped={yaw_wrapped:+.4f} alpha={self.lowpass_alpha}")

        sid = int(m.ship_id)
        arr = float(m.arrival_time)
        rep = float(m.reported_ts)
        mid = self._norm_mid(getattr(m, "msg_id", ""))
        # 仅做类型与 wrap 归一（不转换语义）
        yaw_sim_rad = self._wrap_pi(float(m.cog_rad))

        m = SimpleMeas(
            ship_id=int(m.ship_id),
            x=float(m.x),
            y=float(m.y),
            sog_mps=float(m.sog_mps),
            cog_rad=float(yaw_sim_rad),
            arrival_time=float(m.arrival_time),
            reported_ts=float(m.reported_ts),
            msg_id=mid,
        )

        # ---------- 1) buffer（arrival_time 排序用于“到达顺序”观测/调试，不影响主 gating dt） ----------
        buf = self.buffers.setdefault(sid, [])
        buf.append(m)
        buf.sort(key=lambda mm: float(mm.arrival_time))
        while buf and arr - float(buf[0].arrival_time) > self.window_sec:
            buf.pop(0)
        if len(buf) > self.max_buffer:
            buf[:] = buf[-self.max_buffer:]

        # ---------- 2) gating ----------
        last = self.last_acc.get(sid, None)
        if last is not None:
            last_arr = float(last.arrival_time)
            last_rep = float(last.reported_ts)

            dt_arr = arr - last_arr
            dt_rep = rep - last_rep

            # (0) arrival_time 乱序：直接丢弃（不累计 streak）
            if dt_arr < -1e-6:
                self._dbg_drop(
                    m, "stale_arrival_time", last=last,
                    extra=f"dt_arr={dt_arr:.3f} (<0) dt_rep={dt_rep:.3f}"
                )
                return []

            # (A) reported_ts 乱序/旧报文：丢弃（不累计 streak）
            if dt_rep <= 0.0:
                self._dbg_drop(
                    m, "stale_reported_ts", last=last,
                    extra=f"dt_rep={dt_rep:.3f} (<=0) dt_arr={dt_arr:.3f}"
                )
                return []

            # (B) gap reset：以 reported_ts 为主，避免 delay/burst 误触发
            if dt_rep > self.gap_reset_sec:
                self._dbg(f"[PP-GAP_RESET] sid={sid} dt_rep={dt_rep:.2f} > {self.gap_reset_sec:.2f}")
                self._reset_ship_state(sid)
                last = None
            else:
                dt = float(dt_rep)  # 已保证 > 0

                # (C) pos gating：用 dt_rep
                dx = float(m.x - last.x)
                dy = float(m.y - last.y)
                dist = math.hypot(dx, dy)

                v_est = max(float(last.sog_mps), self.v_floor_mps)
                max_jump = self.base_pos_jump + self.dyn_jump_k * v_est * dt

                if dist > max_jump:
                    s = self.drop_streak.get(sid, 0) + 1
                    self.drop_streak[sid] = s
                    self._dbg_drop(
                        m, "pos_jump", last=last,
                        extra=f"dist={dist:.2f} > max_jump={max_jump:.2f} dt_rep={dt_rep:.2f} streak={s}"
                    )
                    if s < self.max_consec_drop:
                        return []
                    self._dbg(f"[PP-FORCE_RESET] sid={sid} reason=POS streak={s} -> reseed_current")
                    self._reset_ship_state(sid)
                    last = None

                last_gate = self.last_raw.get(sid, last)

                # (D) sog gating
                if last_gate is not None:
                    dsog = abs(float(m.sog_mps) - float(last_gate.sog_mps))
                    if dsog > self.max_sog_jump_mps:
                        s = self.drop_streak.get(sid, 0) + 1
                        self.drop_streak[sid] = s
                        self._dbg_drop(
                            m, "sog_jump", last=last,
                            extra=f"dsog={dsog:.2f} > max_sog_jump={self.max_sog_jump_mps:.2f} streak={s}"
                        )
                        if s < self.max_consec_drop:
                            return []
                        self._dbg(f"[PP-FORCE_RESET] sid={sid} reason=SOG streak={s} -> reseed_current")
                        self._reset_ship_state(sid)
                        last = None

                # (E) yaw gating（legacy 字段名 cog_rad，但语义是 yaw_sim）
                if last_gate is not None:
                    d = float(m.cog_rad) - float(last_gate.cog_rad)
                    d_yaw = abs(math.atan2(math.sin(d), math.cos(d)))

                    # 动态放宽：dt_rep 越大允许更大角度变化；上限 π
                    yaw_thr = min(math.pi, self.max_cog_jump_rad * max(1.0, float(dt_rep)))
                    if d_yaw > yaw_thr:
                        s = self.drop_streak.get(sid, 0) + 1
                        self.drop_streak[sid] = s
                        self._dbg_drop(
                            m, "yaw_jump", last=last,
                            extra=f"d_yaw={d_yaw:.3f} > yaw_thr={yaw_thr:.3f} dt_rep={dt_rep:.2f} streak={s}"
                        )
                        if s < self.max_consec_drop:
                            return []
                        self._dbg(f"[PP-FORCE_RESET] sid={sid} reason=YAW streak={s} -> reseed_current")
                        self._reset_ship_state(sid)
                        last = None

                # 仅当未 reseed 且全通过时清 streak
                if last is not None:
                    self.drop_streak[sid] = 0
        else:
            self.drop_streak[sid] = 0

        self.last_raw[sid] = m

        # ---------- 3) lowpass ----------
        mf = self._lowpass(sid, m)

        # ---------- 4) init policy ----------
        already_inited = bool(self.inited.get(sid, False))

        # passthrough: 第一条就吐
        if (not already_inited) and (self.init_policy == "passthrough"):
            self.inited[sid] = True
            self.init_buffers[sid] = []
            self.last_acc[sid] = mf
            self._dbg(f"[PP-INIT_PASS] sid={sid} rep={mf.reported_ts:.2f} arr={mf.arrival_time:.2f}")
            self._dbg_accept(m, mf, "INIT_PASS")
            return [mf]

        # warmstart_ok: warm-start 过就吐，否则按 avg
        if (not already_inited) and (self.init_policy == "warmstart_ok") and self.warm_started.get(sid, False):
            self.inited[sid] = True
            self.init_buffers[sid] = []
            self.last_acc[sid] = mf
            self._dbg_accept(m, mf, "WARMSTART_OK")
            return [mf]

        # avg init（旧逻辑）
        if not already_inited:
            init_buf = self.init_buffers.setdefault(sid, [])
            init_buf.append(mf)
            self.last_acc[sid] = mf

            K = max(1, self.init_avg_K)
            if len(init_buf) < K:
                self._dbg_drop(
                    mf, "init_hold", last=self.last_acc.get(sid, None),
                    extra=f"len={len(init_buf)}/{K} policy=avg"
                )
                self._dbg(f"[PP-INIT_HOLD] sid={sid} len={len(init_buf)}/{K} rep={mf.reported_ts:.2f} arr={mf.arrival_time:.2f}")
                return []

            xs = np.array([mm.x for mm in init_buf], dtype=np.float64)
            ys = np.array([mm.y for mm in init_buf], dtype=np.float64)
            sogs = np.array([mm.sog_mps for mm in init_buf], dtype=np.float64)
            yaws = np.array([mm.cog_rad for mm in init_buf], dtype=np.float64)  # semantics: yaw_sim

            avg_x = float(xs.mean())
            avg_y = float(ys.mean())
            avg_sog = float(sogs.mean())
            avg_yaw = float(math.atan2(np.sin(yaws).mean(), np.cos(yaws).mean()))

            # stable fused msg_id (order-independent)
            blend_id = self._blend_mid([mm.msg_id for mm in init_buf])
            if not blend_id:
                blend_id = f"blend:t{int(round(float(init_buf[-1].reported_ts)*1000.0))}"


            meas_out = SimpleMeas(
                ship_id=sid,
                x=avg_x,
                y=avg_y,
                sog_mps=avg_sog,
                cog_rad=float(self._wrap_pi(avg_yaw)),
                arrival_time=float(init_buf[-1].arrival_time),
                reported_ts=float(init_buf[-1].reported_ts),
                msg_id=blend_id,
            )

            self.inited[sid] = True
            self.init_buffers[sid] = []
            self.last_acc[sid] = meas_out
            self._dbg_accept(mf, meas_out, "INIT_OK", extra=f"K={K}")
            self._dbg(f"[PP-INIT_OK] sid={sid} K={K} out_rep={meas_out.reported_ts:.2f} out_arr={meas_out.arrival_time:.2f}")
            return [meas_out]

        # ---------- 5) normal ----------
        self.last_acc[sid] = mf

        # 保证 reported_ts/arrival_time 不被污染（严格对齐语义）
        if self.debug:
            self._dbg_accept(m, mf, "PASS")
            assert abs(float(mf.reported_ts) - float(m.reported_ts)) < 1e-6, f"reported_ts changed! sid={sid}"
            assert abs(float(mf.arrival_time) - float(m.arrival_time)) < 1e-6, f"arrival_time changed! sid={sid}"

        return [mf]

    # ------------------------------------------------------------
    def _lowpass(self, sid: int, m: SimpleMeas) -> SimpleMeas:
        if self.lowpass_alpha <= 0.0:
            return m

        alpha = float(self.lowpass_alpha)
        prev = self.lp_state.get(sid, None)

        x_f = float(m.x)
        y_f = float(m.y)

        if prev is None:
            sog_f = float(m.sog_mps)
            yaw_f = float(m.cog_rad)
        else:
            sog_f = alpha * float(m.sog_mps) + (1.0 - alpha) * float(prev["sog"])
            sin_c = math.sin(float(m.cog_rad))
            cos_c = math.cos(float(m.cog_rad))
            sin_p = float(prev["sin_yaw"])
            cos_p = float(prev["cos_yaw"])
            sin_f = alpha * sin_c + (1.0 - alpha) * sin_p
            cos_f = alpha * cos_c + (1.0 - alpha) * cos_p
            yaw_f = math.atan2(sin_f, cos_f)

        yaw_f = float(self._wrap_pi(yaw_f))

        self.lp_state[sid] = dict(
            x=x_f, y=y_f, sog=sog_f,
            sin_yaw=math.sin(yaw_f),
            cos_yaw=math.cos(yaw_f),
        )

        return SimpleMeas(
            ship_id=int(m.ship_id),
            x=x_f,
            y=y_f,
            sog_mps=float(sog_f),
            cog_rad=float(yaw_f),                 # semantics: yaw_sim
            arrival_time=float(m.arrival_time),   # pass-through
            reported_ts=float(m.reported_ts),     # pass-through
            msg_id=self._norm_mid(getattr(m, "msg_id", "")),
        )
