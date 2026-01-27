# ais_comms/pf_ctrv.py
from __future__ import annotations

"""
================================================================================
ParticleCTRVFilter (CTRV-PF) — TIME/ANGLE SEMANTICS CONTRACT
(MUST MATCH track_manager_pf.py AND ais_preproc.py)
================================================================================

This PF module must be end-to-end consistent with:
  - ais_comms/track_manager_pf.py  : single-point AIS COG conversion + PF arrival-time axis
  - ais_comms/ais_preproc.py       : preproc never converts angles; cog_rad is legacy yaw_sim_rad

0) Time axis conventions
PF internal time cursor: self.last_ts
- Unit: seconds
- Meaning: PF propagation/update cursor on ENV TIME axis.
- Enforced by TrackManagerPF: every predict/update is performed at t_pf = t_env.

reported_ts / arrival_time exist upstream only:
- reported_ts (Tx-time) is used by TrackManagerPF to compute age = t_env - ts_rep
  and to forward-project (x_rep,y_rep) to (x_use,y_use) at t_env.
- arrival_time is metadata only (comms debug); PF never uses it as time axis.

"age" parameter:
- In current system, PF.step_delay_robust(..., age) receives
    age := t_env - ts_rep
  (message staleness), used only for noise inflation / gating.

1) Angle conventions
Incoming raw COG direction (RxMsg.reported_cog):
- Meaning: raw_cog_rad == yaw_sim_rad == atan2(vy,vx), ENU +X=0, CCW+, rad, (-pi,pi]

Hard rules:
- PF consumes yaw_sim_rad only; no North/CW conversion anywhere in PF.
- meas["yaw"] is canonical; meas["cog"] is legacy alias and MUST equal yaw_sim_rad.

About "age" parameter used by AGE gating:
  - In this project (per TrackManagerPF), PF.step_delay_robust(..., age=age_axis)
    receives age_axis := t_env_now - meas_ts_on_pf_axis (typically now - arrival_time).
  - This "age" therefore measures *how stale the PF-axis measurement is relative to now*,
    not the message Tx timestamp.

--------------------------------
1) Angle conventions (SINGLE-POINT conversion)
--------------------------------

We use TWO angle conventions in the project:

(A) AIS COG from RxMsg.reported_cog (north=0, CW+).  [AIS convention]
(B) Internal yaw (yaw_sim_rad): +x=0, CCW+ (atan2(vy, vx)). [PF convention]

Single-point conversion rule (ENFORCED upstream):
  - AIS COG -> yaw_sim_rad conversion happens ONLY in track_manager_pf.py:
        yaw_sim_rad = wrap_pi(pi/2 - cog_north_cw_rad)

Hard rules in THIS PF:
  - Any measurement angle consumed by PF MUST already be yaw_sim_rad.
  - PF never performs NorthCW<->ENU conversion.
  - Field name "cog" may exist in meas dict, but it is LEGACY and MUST carry yaw_sim_rad.

Measurement dict conventions (from TrackManagerPF):
  - meas["yaw"] : canonical key, semantics = yaw_sim_rad (rad)
  - meas["cog"] : legacy alias, semantics = yaw_sim_rad (rad)
  - PF must prefer "yaw" and fallback to "cog".

--------------------------------
2) State definition
--------------------------------

State vector for each particle:
  x = [px, py, v, yaw, yawd]
    px, py : position in world/local plane (meters)
    v      : speed magnitude (m/s)
    yaw    : yaw_sim_rad, internal yaw (+x=0, CCW+), wrapped to (-pi, pi]
    yawd   : yaw rate (rad/s)

--------------------------------
3) Allowed / NOT allowed
--------------------------------

Allowed:
  - predict(dt) on PF axis
  - update using (x,y,sog,yaw) measurements already aligned to PF axis
  - AGE gating based on "age" (as defined above) to inflate measurement noise

Not allowed:
  - use of reported_ts as PF axis
  - any angle convention conversion
  - time rollback

================================================================================
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import math
import numpy as np
import os

# ------------------------- helpers -------------------------

def _wrap_pi(a: float) -> float:
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi

def _wrap_pi_np(a: np.ndarray) -> np.ndarray:
    """Vectorized wrap to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _angle_diff(a: float, b: float) -> float:
    """Return wrap_pi(a - b)."""
    return _wrap_pi(float(a) - float(b))

def _meas_yaw(meas: Dict[str, Any]) -> float:
    """Canonical measurement yaw accessor: prefer 'yaw', fallback to legacy 'cog'."""
    return _wrap_pi(float(meas.get("yaw", meas.get("cog", 0.0))))

# ------------------------- configs -------------------------

@dataclass
class PFAgeGatingConfig:
    # --------- AGE gating parameters (seconds) ----------
    age_fresh: float = 1.0
    age_stale: float = 3.0
    age_very_stale: float = 5.0
    age_max: float = 8.0  # age >= age_max => reject update

    # noise inflation multipliers (applied on measurement std)
    scale_fresh: float = 1.0
    scale_stale: float = 1.5
    scale_very_stale: float = 2.0

    # per-channel extra scale
    pos_scale_factor: float = 1.0
    vel_scale_factor: float = 1.0

    # IMPORTANT (track_manager_pf.py expects this name):
    # legacy: "cog" field name, but semantics is yaw_sim_rad scaling.
    cog_scale_factor: float = 1.0
    # >>> NEW: continuous staleness inflation (Scheme A)
    # >>> (原来 Scheme A 的连续膨胀参数：现在主要用于 Q 扩散，不再用于 pos/sog 的 R)
    pos_age_v_gain: float = 0.6      # unitless, extra Q pos std ~ gain * v * age
    sog_age_a_gain: float = 0.25     # m/s^2, extra Q sog std ~ gain * age
    yaw_age_rate_std: float = math.radians(3.0)  # rad/s, yaw 的 R 退火项：std ~ rate_std * age

    # >>> 原有 R caps
    pos_std_max: float = 300.0
    sog_std_max: float = 5.0
    yaw_std_max: float = math.radians(120.0)


    # =========================
    # STAGE-4: R->Q shift switches
    # =========================
    use_q_gap_inflation: bool = True          # age 主要作用转移到 Q（update 前扩散）
    use_r_continuous_inflation: bool = False  # 若要回退旧行为，可设 True（不建议）

    # STAGE-4: Q gap 扩散 caps（建议远小于 R caps，避免一次扩散过猛）
    q_pos_std_max: float = 60.0               # meters, extra Q pos std cap
    q_sog_std_max: float = 2.0                # m/s, extra Q sog std cap
    q_yaw_std_max: float = math.radians(45.0) # rad, extra Q yaw std cap
    q_yawd_std_max: float = math.radians(10.0)# rad/s, extra Q yawd std cap

    # STAGE-4: yaw gating / annealing (R-side)
    yaw_gate_enable: bool = True
    yaw_gate_age: float = 2.5                # age >= 该值才允许触发 yaw 门控
    yaw_gate_innov_deg: float = 35.0         # |wrap(z_yaw - yaw_pred)| 超过该阈值触发
    yaw_gate_sigma_mult: float = 6.0         # 触发后 sigma_yaw *= mult（再 clip 到 yaw_std_max）

    # STAGE-4: 可选的 Q yaw/yawd 扩散强度（默认先关：0.0）
    q_yaw_age_rate_std: float = 0.0          # rad/s, extra Q yaw std ~ rate_std * age
    q_yawd_age_rate_std: float = 0.0         # rad/s^2, extra Q yawd std ~ rate_std * age


    # =========================
    # NEW: R->Q shift switches
    # =========================
    use_q_gap_inflation: bool = True          # age 主要作用转移到 Q（update 前扩散）
    use_r_continuous_inflation: bool = False  # 若要回退旧行为，可设 True（不建议）

    # NEW: Q gap 扩散 caps（建议远小于 R caps，避免一次扩散过猛）
    q_pos_std_max: float = 60.0               # meters, extra Q pos std cap
    q_sog_std_max: float = 2.0                # m/s, extra Q sog std cap
    q_yaw_std_max: float = math.radians(45.0) # rad, extra Q yaw std cap (可先不用)
    q_yawd_std_max: float = math.radians(10.0)# rad/s, extra Q yawd std cap

    # NEW: yaw gating / annealing
    yaw_gate_enable: bool = True
    yaw_gate_age: float = 2.5                # age >= 该值才允许触发 yaw 门控
    yaw_gate_innov_deg: float = 35.0         # |wrap(z_yaw - yaw_pred)| 超过该阈值触发
    yaw_gate_sigma_mult: float = 6.0         # 触发后 sigma_yaw *= mult（再 clip 到 yaw_std_max）

    # NEW: 可选的 Q yaw/yawd 扩散强度（默认先关：0.0）
    q_yaw_age_rate_std: float = 0.0          # rad/s, extra Q yaw std ~ rate_std * age
    q_yawd_age_rate_std: float = 0.0         # rad/s^2, extra Q yawd std ~ rate_std * age



@dataclass
class PFNoiseConfig:
    # --------- process noise (CTRV simplified) ----------
    process_std_a: float = 0.2       # [m/s^2] accel noise used to perturb v
    process_std_yaw: float = 0.02    # [rad/s^2] yaw-accel noise used to perturb yawd

    # CTRV numerical stability
    ctrv_omega_eps: float = 0.02     # [rad/s] |yawd| < eps => CV-like
    yawd_clip: float = 1.0           # [rad/s] yaw rate clip

    # --------- base measurement noise (before AGE scaling) ----------
    meas_std_pos: float = 15.0        # [m] NOTE: piecewise R uses this as the "stale" level
    meas_std_sog: float = 0.5        # [m/s]
    meas_std_cog_deg: float = 1.0    # [deg] LEGACY FIELD NAME; semantics: yaw_sim std in degrees

    # =========================
    # STAGE-4: robust update switches (yaw/yawd decoupled from weights)
    # =========================
    # If True: yaw enters weight likelihood (can sharpen likelihood and collapse NEFF).
    # Recommended default: False (yaw handled via conditional soft update).
    use_yaw_in_weight: bool = False

    # yaw soft update (state correction, does NOT enter weights)
    yaw_soft_enable: bool = True
    yaw_soft_gain: float = 0.18              # base gain (before sigma-based attenuation)
    yaw_soft_min_speed: float = 0.3          # m/s
    yaw_soft_max_abs_yawd: float = 0.45      # rad/s  (maneuver gate)
    yaw_soft_max_age: float = 2.5            # s      (stale yaw => skip)
    yaw_soft_jitter_deg: float = 0.4         # deg    (small diversity to avoid particle lock)

    # innovation-aware taper/skip (independent of age)
    yaw_soft_taper_start_deg: float = 25.0   # deg
    yaw_soft_taper_end_deg: float = 90.0     # deg
    yaw_soft_outlier_skip_deg: float = 150.0 # deg

    # yawd soft update (turn-rate correction, does NOT enter weights)
    yawd_soft_enable: bool = True
    yawd_soft_gain: float = 0.10           # 建议 0.05~0.15，小一点更稳
    yawd_soft_min_dt: float = 0.10         # s，dt 太小会导致 dpsi/dt 爆
    yawd_soft_max_innov_deg: float = 25.0  # deg，创新太大通常是 outlier/错配，别用来估 yawd

    # ---------------- robust update switches ----------------
    # If True: yaw enters weight likelihood (can sharpen likelihood and collapse NEFF).
    # Recommended default: False (yaw handled via conditional soft update).
    use_yaw_in_weight: bool = False

    # yaw soft update (state correction, does NOT enter weights)
    yaw_soft_enable: bool = True
    yaw_soft_gain: float = 0.18              # base gain (before sigma-based attenuation)
    yaw_soft_min_speed: float = 0.3          # m/s
    yaw_soft_max_abs_yawd: float = 0.45      # rad/s  (maneuver gate)
    yaw_soft_max_age: float = 2.5            # s      (stale yaw => skip)
    yaw_soft_jitter_deg: float = 0.4         # deg    (small diversity to avoid particle lock)


    # NEW: innovation-aware taper/skip (independent of age)
    # - when yaw innovation is moderately large, taper gain down
    # - when yaw innovation is extremely large, skip yaw correction entirely
    yaw_soft_taper_start_deg: float = 25.0   # deg
    yaw_soft_taper_end_deg: float = 90.0     # deg
    yaw_soft_outlier_skip_deg: float = 150.0 # deg

    # >>> NEW: yawd soft update (turn-rate correction, does NOT enter weights)
    yawd_soft_enable: bool = True
    yawd_soft_gain: float = 0.10           # 建议 0.05~0.15，小一点更稳
    yawd_soft_min_dt: float = 0.10         # s，dt 太小会导致 dpsi/dt 爆
    yawd_soft_max_innov_deg: float = 25.0  # deg，创新太大通常是 outlier/错配，别用来估 yawd


    # --------- AGE gating ----------
    age: PFAgeGatingConfig = field(default_factory=PFAgeGatingConfig)


@dataclass
class PFConfig:
    """CTRV-PF configuration."""
    noise: PFNoiseConfig = field(default_factory=PFNoiseConfig)
    num_particles: int = 256
    resample_threshold_ratio: float = 0.5  # resample if N_eff < ratio * N
    seed: int = 0


# ------------------------- PF implementation -------------------------

class ParticleCTRVFilter:
    """
    Simplified CTRV Particle Filter:
      - state: [px, py, v, yaw, yawd]
      - bootstrap PF with Gaussian measurement likelihood on [x, y, sog, yaw]
      - AGE gating inflates measurement std based on provided 'age' (per contract above)
    """

    def __init__(self, cfg: PFConfig):
        self.cfg = cfg
        self.N = int(cfg.num_particles)
        self.rng = np.random.default_rng(int(cfg.seed))

        self.x_particles: Optional[np.ndarray] = None  # (N, 5)
        self.w: Optional[np.ndarray] = None            # (N,)
        self.last_ts: Optional[float] = None           # PF-axis cursor (ENV time axis)


        # public point estimate (5,)
        self.x = np.zeros(5, dtype=float)

        self._deg2rad = math.pi / 180.0
        self.resample_threshold = float(self.N) * float(self.cfg.resample_threshold_ratio)

        # =========================
        # STAGE-4: anti-double-counting / integrity guards
        # =========================
        self.last_meas_update_ts = -1.0
        self.last_update_fingerprint = None  # (meas_source_ts, x, y, yaw)
        self.processed_msg_ids = set()
        self.last_update_stats = {}
    # ------------------------------------------------------------------
    # dynamics: CTRV step (deterministic)
    # ------------------------------------------------------------------

    def _ctrv_step(self, x: np.ndarray, dt: float) -> np.ndarray:
        if dt <= 0.0:
            return x

        px, py, v, yaw, yawd = x.T

        omega_eps = float(self.cfg.noise.ctrv_omega_eps)
        small_omega = np.abs(yawd) < omega_eps

        px_new = np.copy(px)
        py_new = np.copy(py)
        v_new = np.copy(v)
        yaw_new = np.copy(yaw)
        yawd_new = np.copy(yawd)

        # CV-like
        idx_cv = small_omega
        px_new[idx_cv] += v[idx_cv] * np.cos(yaw[idx_cv]) * dt
        py_new[idx_cv] += v[idx_cv] * np.sin(yaw[idx_cv]) * dt
        yaw_new[idx_cv] += yawd[idx_cv] * dt

        # CTRV
        idx_ctrv = ~small_omega
        omega = yawd[idx_ctrv]
        omega_clip = float(self.cfg.noise.yawd_clip)
        omega = np.clip(omega, -omega_clip, omega_clip)

        v_c = v[idx_ctrv]
        yaw_c = yaw[idx_ctrv]

        px_new[idx_ctrv] += v_c / omega * (np.sin(yaw_c + omega * dt) - np.sin(yaw_c))
        py_new[idx_ctrv] += v_c / omega * (-np.cos(yaw_c + omega * dt) + np.cos(yaw_c))
        yaw_new[idx_ctrv] += omega * dt

        # fast vectorized wrap (avoid np.vectorize)
        yaw_new = _wrap_pi_np(yaw_new)
        return np.stack([px_new, py_new, v_new, yaw_new, yawd_new], axis=1)

    # ------------------------------------------------------------------
    # process noise
    # ------------------------------------------------------------------

    def _sample_process_noise(self, N: int, dt: float) -> np.ndarray:
        c = self.cfg.noise
        dt = float(max(0.0, dt))

        a_std = float(c.process_std_a)
        yawdd_std = float(c.process_std_yaw)  # yaw acceleration std

        dv_std = max(1e-6, a_std * dt)
        dyawd_std = max(1e-6, yawdd_std * dt)

        dv = self.rng.normal(0.0, dv_std, size=N)
        dyawd = self.rng.normal(0.0, dyawd_std, size=N)

        # simple position diffusion
        pos_std = max(0.2, 0.5 * a_std * (dt ** 2))
        dpx = self.rng.normal(0.0, pos_std, size=N)
        dpy = self.rng.normal(0.0, pos_std, size=N)

        return np.stack([dpx, dpy, dv, np.zeros(N), dyawd], axis=1)

    # ------------------------------------------------------------------
    # STAGE-4: R->Q shift (gap diffusion before update)
    # ------------------------------------------------------------------
    def _apply_gap_process_noise(self, gap: float, meas: Dict[str, Any] | None = None):
        """
        Move staleness effect from R -> Q:
        Before applying measurement likelihood, diffuse particles according to gap(age).
        This helps keep R tight (so measurement can pull back), while representing uncertainty as process noise.
        """
        if self.x_particles is None:
            return

        ag = self.cfg.noise.age
        if not bool(getattr(ag, "use_q_gap_inflation", True)):
            return

        gap = float(max(0.0, gap))
        if gap <= 0.0:
            return

        # reference speed for diffusion scale
        v_ref = 0.0
        try:
            if isinstance(meas, dict) and ("sog" in meas):
                v_ref = float(meas["sog"])
            else:
                v_ref = float(self.x[2]) if hasattr(self, "x") else 0.0
        except Exception:
            v_ref = 0.0
        v_ref = max(0.0, v_ref)

        # --- main diffusion (pos/sog) ---
        sigma_pos = float(ag.pos_age_v_gain) * v_ref * gap
        sigma_sog = float(ag.sog_age_a_gain) * gap

        sigma_pos = min(sigma_pos, float(getattr(ag, "q_pos_std_max", 60.0)))
        sigma_sog = min(sigma_sog, float(getattr(ag, "q_sog_std_max", 2.0)))

        if sigma_pos > 0.0:
            self.x_particles[:, 0] += self.rng.normal(0.0, sigma_pos, size=self.N)
            self.x_particles[:, 1] += self.rng.normal(0.0, sigma_pos, size=self.N)

        if sigma_sog > 0.0:
            self.x_particles[:, 2] += self.rng.normal(0.0, sigma_sog, size=self.N)

        # --- optional yaw/yawd diffusion (default 0) ---
        q_yaw_rate = float(getattr(ag, "q_yaw_age_rate_std", 0.0))
        q_yawd_rate = float(getattr(ag, "q_yawd_age_rate_std", 0.0))

        sigma_yaw = q_yaw_rate * gap
        sigma_yawd = q_yawd_rate * gap

        sigma_yaw = min(sigma_yaw, float(getattr(ag, "q_yaw_std_max", math.radians(45.0))))
        sigma_yawd = min(sigma_yawd, float(getattr(ag, "q_yawd_std_max", math.radians(10.0))))

        if sigma_yaw > 0.0:
            self.x_particles[:, 3] = _wrap_pi_np(
                self.x_particles[:, 3] + self.rng.normal(0.0, sigma_yaw, size=self.N)
            )
        if sigma_yawd > 0.0:
            self.x_particles[:, 4] += self.rng.normal(0.0, sigma_yawd, size=self.N)

        # keep bounds
        yclip = float(self.cfg.noise.yawd_clip)
        self.x_particles[:, 4] = np.clip(self.x_particles[:, 4], -yclip, yclip)
        self.x_particles[:, 3] = _wrap_pi_np(self.x_particles[:, 3])


    def _apply_gap_process_noise(self, gap: float, meas: Dict[str, Any] | None = None):
        """
        Move staleness effect from R -> Q:
        Before applying measurement likelihood, diffuse particles according to gap(age).
        This helps keep R tight (so measurement can pull back), while representing uncertainty as process noise.
        """
        if self.x_particles is None:
            return

        ag = self.cfg.noise.age
        if not bool(getattr(ag, "use_q_gap_inflation", True)):
            return

        gap = float(max(0.0, gap))
        if gap <= 0.0:
            return

        # reference speed for diffusion scale
        v_ref = 0.0
        try:
            if isinstance(meas, dict) and ("sog" in meas):
                v_ref = float(meas["sog"])
            else:
                v_ref = float(self.x[2]) if hasattr(self, "x") else 0.0
        except Exception:
            v_ref = 0.0
        v_ref = max(0.0, v_ref)

        # --- main diffusion (pos/sog) ---
        sigma_pos = float(ag.pos_age_v_gain) * v_ref * gap
        sigma_sog = float(ag.sog_age_a_gain) * gap

        sigma_pos = min(sigma_pos, float(getattr(ag, "q_pos_std_max", 60.0)))
        sigma_sog = min(sigma_sog, float(getattr(ag, "q_sog_std_max", 2.0)))

        if sigma_pos > 0.0:
            self.x_particles[:, 0] += self.rng.normal(0.0, sigma_pos, size=self.N)
            self.x_particles[:, 1] += self.rng.normal(0.0, sigma_pos, size=self.N)

        if sigma_sog > 0.0:
            self.x_particles[:, 2] += self.rng.normal(0.0, sigma_sog, size=self.N)

        # --- optional yaw/yawd diffusion (default 0) ---
        q_yaw_rate = float(getattr(ag, "q_yaw_age_rate_std", 0.0))
        q_yawd_rate = float(getattr(ag, "q_yawd_age_rate_std", 0.0))

        sigma_yaw = q_yaw_rate * gap
        sigma_yawd = q_yawd_rate * gap

        sigma_yaw = min(sigma_yaw, float(getattr(ag, "q_yaw_std_max", math.radians(45.0))))
        sigma_yawd = min(sigma_yawd, float(getattr(ag, "q_yawd_std_max", math.radians(10.0))))

        if sigma_yaw > 0.0:
            self.x_particles[:, 3] = _wrap_pi_np(
                self.x_particles[:, 3] + self.rng.normal(0.0, sigma_yaw, size=self.N)
            )
        if sigma_yawd > 0.0:
            self.x_particles[:, 4] += self.rng.normal(0.0, sigma_yawd, size=self.N)

        # keep bounds
        yclip = float(self.cfg.noise.yawd_clip)
        self.x_particles[:, 4] = np.clip(self.x_particles[:, 4], -yclip, yclip)
        self.x_particles[:, 3] = _wrap_pi_np(self.x_particles[:, 3])


    # ------------------------------------------------------------------
    # AGE gating: age -> noise scale
    # ------------------------------------------------------------------

    def _age_to_scale(self, age: float, kind: str) -> float:
        g = self.cfg.noise.age
        age = float(max(0.0, age))

        if age <= g.age_fresh:
            base = g.scale_fresh
        elif age <= g.age_stale:
            u = (age - g.age_fresh) / max(1e-9, (g.age_stale - g.age_fresh))
            base = (1.0 - u) * g.scale_fresh + u * g.scale_stale
        elif age <= g.age_very_stale:
            u = (age - g.age_stale) / max(1e-9, (g.age_very_stale - g.age_stale))
            base = (1.0 - u) * g.scale_stale + u * g.scale_very_stale
        else:
            base = g.scale_very_stale

        if kind == "pos":
            return float(base) * float(g.pos_scale_factor)
        if kind == "vel":
            return float(base) * float(g.vel_scale_factor)

        # accept both names to be extra safe
        if kind in ("cog", "yaw"):
            return float(base) * float(g.cog_scale_factor)

        return float(base)


    # 2) 让 _build_meas_std 接受 meas，从 meas["sog"] 提取 v_ref
    # ------------------------------------------------------------------
    # STAGE-4: piecewise R (pos/sog stable) + yaw anneal/gate; optional old continuous R inflation
    # ------------------------------------------------------------------
    def _build_meas_std(self, age: float, meas: dict | None = None, *, yaw_innov_abs: float | None = None):
        age = float(max(0.0, age))
        ag = self.cfg.noise.age

        # (1) POS: freshness piecewise (R stays interpretable)
        sigma_pos_stale = float(self.cfg.noise.meas_std_pos) * float(ag.pos_scale_factor)
        sigma_pos_fresh = min(4.0, max(2.0, 0.25 * sigma_pos_stale))
        if sigma_pos_stale <= 4.0:
            sigma_pos_fresh = sigma_pos_stale

        if age <= float(ag.age_fresh):
            sigma_pos = sigma_pos_fresh
        elif age <= float(ag.age_stale):
            u = (age - float(ag.age_fresh)) / max(1e-9, (float(ag.age_stale) - float(ag.age_fresh)))
            sigma_pos = (1.0 - u) * sigma_pos_fresh + u * sigma_pos_stale
        else:
            sigma_pos = sigma_pos_stale

        # (2) SOG: keep R stable
        sigma_sog = float(self.cfg.noise.meas_std_sog) * float(ag.vel_scale_factor)

        # (3) yaw: age-based annealing (R) + optional gating
        if age <= float(ag.age_fresh):
            yaw_scale = float(ag.scale_fresh)
        elif age <= float(ag.age_stale):
            yaw_scale = float(ag.scale_stale)
        else:
            yaw_scale = float(ag.scale_very_stale)

        sigma_yaw0 = math.radians(float(self.cfg.noise.meas_std_cog_deg)) * yaw_scale * float(ag.cog_scale_factor)
        sigma_yaw = math.sqrt(sigma_yaw0 * sigma_yaw0 + (float(ag.yaw_age_rate_std) * age) ** 2)

        # optional: backward compatible (old Scheme A in R)
        if bool(getattr(ag, "use_r_continuous_inflation", False)):
            v_ref = 0.0
            try:
                if isinstance(meas, dict) and ("sog" in meas):
                    v_ref = float(meas["sog"])
                else:
                    v_ref = float(self.x[2]) if hasattr(self, "x") else 0.0
            except Exception:
                v_ref = 0.0
            v_ref = max(0.0, v_ref)

            sigma_pos = math.sqrt(sigma_pos * sigma_pos + (float(ag.pos_age_v_gain) * v_ref * age) ** 2)
            sigma_sog = math.sqrt(sigma_sog * sigma_sog + (float(ag.sog_age_a_gain) * age) ** 2)

        # caps (R)
        sigma_pos = min(sigma_pos, float(ag.pos_std_max))
        sigma_sog = min(sigma_sog, float(ag.sog_std_max))
        sigma_yaw = min(sigma_yaw, float(ag.yaw_std_max))

        # yaw gating
        if bool(getattr(ag, "yaw_gate_enable", True)) and (yaw_innov_abs is not None):
            if (age >= float(getattr(ag, "yaw_gate_age", 2.5))) and (
                yaw_innov_abs >= math.radians(float(getattr(ag, "yaw_gate_innov_deg", 35.0)))
            ):
                sigma_yaw = min(
                    sigma_yaw * float(getattr(ag, "yaw_gate_sigma_mult", 6.0)),
                    float(ag.yaw_std_max),
                )

        return sigma_pos, sigma_sog, sigma_yaw

    # ------------------------------------------------------------------
    # STAGE-4: yaw/yawd soft update (decoupled from weights)
    # ------------------------------------------------------------------
    def _yaw_soft_update(
        self,
        meas: Dict[str, Any],
        *,
        age: float,
        sigma_yaw: float,
        yaw_innov_abs: float | None,
        dt: float | None = None,
    ):
        c = self.cfg.noise
        ag = self.cfg.noise.age
        if not bool(getattr(c, "yaw_soft_enable", True)):
            return
        if self.x_particles is None or self.w is None:
            return

        age = float(max(0.0, age))
        if age > float(getattr(c, "yaw_soft_max_age", 2.5)):
            return

        # speed gate
        try:
            v_ref = float(meas.get("sog", 0.0))
        except Exception:
            v_ref = 0.0
        if v_ref < float(getattr(c, "yaw_soft_min_speed", 0.3)):
            return

        # maneuver gate
        yawd_est = float(np.average(self.x_particles[:, 4], weights=self.w))
        if abs(yawd_est) > float(getattr(c, "yaw_soft_max_abs_yawd", 0.45)):
            return

        z_yaw = _meas_yaw(meas)
        yaw = self.x_particles[:, 3]
        dpsi = _wrap_pi_np(z_yaw - yaw)

        base_sigma = max(1e-6, math.radians(float(c.meas_std_cog_deg)))
        sigma_yaw = float(max(1e-6, sigma_yaw))
        gain0 = float(getattr(c, "yaw_soft_gain", 0.18))
        gain = gain0 * min(1.0, (base_sigma / sigma_yaw) ** 2)

        # innovation-aware taper/skip
        if yaw_innov_abs is not None:
            out_skip = math.radians(float(getattr(c, "yaw_soft_outlier_skip_deg", 150.0)))
            if yaw_innov_abs >= out_skip:
                return
            taper_start = math.radians(float(getattr(c, "yaw_soft_taper_start_deg", 25.0)))
            taper_end = math.radians(float(getattr(c, "yaw_soft_taper_end_deg", 90.0)))
            if yaw_innov_abs > taper_start and taper_end > taper_start:
                fac = (taper_end - yaw_innov_abs) / max(1e-9, (taper_end - taper_start))
                fac = float(min(1.0, max(0.0, fac)))
                gain *= fac

        # reuse yaw-gate idea to attenuate gain under stale+large innovation
        if yaw_innov_abs is not None:
            if bool(getattr(ag, "yaw_gate_enable", True)) and (
                age >= float(getattr(ag, "yaw_gate_age", 2.5))
            ) and (
                yaw_innov_abs >= math.radians(float(getattr(ag, "yaw_gate_innov_deg", 35.0)))
            ):
                mult = float(getattr(ag, "yaw_gate_sigma_mult", 6.0))
                gain = gain / max(1.0, mult)

        if gain <= 0.0:
            return

        # (A) yaw correction + tiny jitter
        jitter = math.radians(float(getattr(c, "yaw_soft_jitter_deg", 0.4)))
        self.x_particles[:, 3] = _wrap_pi_np(
            yaw + gain * dpsi + self.rng.normal(0.0, jitter, size=self.N)
        )

        # (B) yawd soft update (optional; needs dt)
        if bool(getattr(c, "yawd_soft_enable", True)) and (dt is not None):
            dt = float(dt)
            if dt >= float(getattr(c, "yawd_soft_min_dt", 0.10)):
                max_innov = math.radians(float(getattr(c, "yawd_soft_max_innov_deg", 25.0)))
                dpsi_clip = np.clip(dpsi, -max_innov, max_innov)

                yawd = self.x_particles[:, 4]
                yclip = float(self.cfg.noise.yawd_clip)
                yawd_target = np.clip(dpsi_clip / dt, -yclip, yclip)

                g0 = float(getattr(c, "yawd_soft_gain", 0.10))
                g = g0 * min(1.0, (base_sigma / sigma_yaw) ** 2)

                self.x_particles[:, 4] = (1.0 - g) * yawd + g * yawd_target
                self.x_particles[:, 4] = np.clip(self.x_particles[:, 4], -yclip, yclip)


    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------

    def init_from_meas(self, ts: float, meas: Dict[str, Any]):
        """
        Initialize PF at PF-axis time ts (arrival-time axis).
        meas must carry internal yaw_sim_rad:
          - prefer meas["yaw"], fallback meas["cog"] (legacy alias).
        Required keys: x, y, sog, yaw(or cog)
        """
        ts = float(ts)
        self.last_ts = ts

        x_m = float(meas["x"])
        y_m = float(meas["y"])
        sog_m = float(meas["sog"])
        yaw_m = _meas_yaw(meas)

        c = self.cfg.noise
        sigma_pos0 = max(0.5, float(c.meas_std_pos))
        sigma_sog0 = max(0.05, float(c.meas_std_sog))
        sigma_yaw0 = max(0.5 * self._deg2rad, float(c.meas_std_cog_deg) * self._deg2rad)

        px0 = self.rng.normal(x_m, sigma_pos0, size=self.N)
        py0 = self.rng.normal(y_m, sigma_pos0, size=self.N)
        v0 = self.rng.normal(sog_m, sigma_sog0, size=self.N)
        yaw0 = self.rng.normal(yaw_m, sigma_yaw0, size=self.N)
        yaw0 = _wrap_pi_np(yaw0)

        # initialize yawd around 0 (rad/s)
        yawd0 = self.rng.normal(0.0, max(1e-4, float(c.process_std_yaw)), size=self.N)
        yawd0 = np.clip(yawd0, -float(c.yawd_clip), float(c.yawd_clip))

        self.x_particles = np.stack([px0, py0, v0, yaw0, yawd0], axis=1)
        self.w = np.ones(self.N, dtype=float) / float(self.N)
        self._update_estimate()

    # ------------------------------------------------------------------
    # time propagation on PF axis
    # ------------------------------------------------------------------

    def predict(self, dt: float):
        """Propagate particles forward on PF axis by dt (seconds)."""
        if self.x_particles is None:
            return
        dt = float(dt)
        if dt <= 0.0:
            return

        self.x_particles = self._ctrv_step(self.x_particles, dt)
        self.x_particles += self._sample_process_noise(self.N, dt)

        yclip = float(self.cfg.noise.yawd_clip)
        self.x_particles[:, 4] = np.clip(self.x_particles[:, 4], -yclip, yclip)
        self.x_particles[:, 3] = _wrap_pi_np(self.x_particles[:, 3])

        if self.last_ts is not None:
            self.last_ts = float(self.last_ts) + dt

        self._update_estimate()

    def predict_to(self, ts: float, eps: float = 1e-6):
        """Predict to absolute PF-axis time ts (monotonic; no rollback)."""
        ts = float(ts)
        if self.last_ts is None:
            self.last_ts = ts
            return
        dt = ts - float(self.last_ts)
        if abs(dt) <= eps:
            self.last_ts = ts
            return
        if dt > 0.0:
            self.predict(dt)
            # snap for exactness
            self.last_ts = ts
        # dt < 0 => ignore (no rollback)

    # ------------------------------------------------------------------
    # measurement update
    # ------------------------------------------------------------------

    def _meas_likelihood_update(
        self,
        meas: Dict[str, Any],
        sigma_pos: float,
        sigma_sog: float,
        sigma_yaw: float,
        *,
        age: float = 0.0,
        yaw_innov_abs: float | None = None,
        dt: float | None = None, 
    ):
        """
        Measurement update at current PF axis time.
        Default behavior:
          - weights: strong update using (x, y, sog)
          - yaw: conditional SOFT update (state correction), does NOT enter weights
        """
        assert self.x_particles is not None and self.w is not None

        z_x = float(meas["x"])
        z_y = float(meas["y"])
        z_sog = float(meas["sog"])
        z_yaw = _meas_yaw(meas)

        px = self.x_particles[:, 0]
        py = self.x_particles[:, 1]
        v  = self.x_particles[:, 2]
        yaw = self.x_particles[:, 3]

        dx = z_x - px
        dy = z_y - py
        dv = z_sog - v
        dyaw = _wrap_pi_np(z_yaw - yaw)

        # -------------------------
        # (A) log-domain weight update (POS + SOG are strong channels)
        # -------------------------
        inv2_pos_var = 0.5 / (sigma_pos ** 2 + 1e-12)
        inv2_sog_var = 0.5 / (sigma_sog ** 2 + 1e-12)
        log_like = - (dx * dx + dy * dy) * inv2_pos_var - (dv * dv) * inv2_sog_var

        # Optional: include yaw in weights (NOT recommended for AIS under maneuver)
        if bool(getattr(self.cfg.noise, "use_yaw_in_weight", False)):
            inv2_yaw_var = 0.5 / (sigma_yaw ** 2 + 1e-12)
            log_like = log_like - (dyaw * dyaw) * inv2_yaw_var

        # Combine with prior weights in log domain, normalize with log-sum-exp
        eps = 1e-300
        logw = np.log(self.w + eps) + log_like
        logw_max = float(np.max(logw))
        w_unn = np.exp(logw - logw_max)
        w_sum = float(np.sum(w_unn))

        collapsed = False
        resampled = False
        if (w_sum <= 0.0) or (not np.isfinite(w_sum)):
            collapsed = True
            self.w[:] = 1.0 / float(self.N)
        else:
            self.w = w_unn / w_sum

        # resample decision (based on normalized weights)
        Neff = 1.0 / float(np.sum(self.w ** 2) + 1e-12)
        neff = float(Neff)
        if Neff < float(self.resample_threshold):
            resampled = True
            self._resample_systematic()
            Neff2 = 1.0 / float(np.sum(self.w ** 2) + 1e-12)
            neff = float(Neff2)

        # -------------------------
        # (B) yaw soft update (decoupled from weights)
        # -------------------------
        if not bool(getattr(self.cfg.noise, "use_yaw_in_weight", False)):
            self._yaw_soft_update(meas, age=float(age), sigma_yaw=float(sigma_yaw), yaw_innov_abs=yaw_innov_abs, dt=dt,)

        self._update_estimate()


        # stats cache (for TrackManager logging)
        age_val = None
        if isinstance(meas, dict):
            age_val = meas.get("age", None)

        # lightweight likelihood range only when probe enabled
        like_min = None
        like_max = None
        if os.environ.get("PF_WEIGHT_PROBE", "0") == "1":
            ll = log_like - float(np.max(log_like))
            like = np.exp(ll)
            like_min = float(np.min(like))
            like_max = float(np.max(like))

        self.last_update_stats = {
                "t_pf": float(getattr(self, "last_ts", -1.0)),
                "sigma_pos": float(sigma_pos),
                "sigma_sog": float(sigma_sog),
                "sigma_yaw": float(sigma_yaw),
                "age": None if age_val is None else float(age_val),
                "neff": float(neff),
                "resampled": bool(resampled),
                "collapsed": bool(collapsed),
                "like_min": like_min,
                "like_max": like_max,
            }

        if os.environ.get("PF_WEIGHT_PROBE", "0") == "1":
            print(
                f"[PF-WEIGHT-PROBE] t={getattr(self,'last_ts',-1.0):.2f} "
                f"sigma_pos={sigma_pos:.2f} sigma_sog={sigma_sog:.2f} sigma_yaw={sigma_yaw:.3f} "
                f"neff={neff:.1f}/{self.N} resampled={int(resampled)} collapsed={int(collapsed)} "
                f"LIKE=[{like_min if like_min is not None else float('nan'):.1e},"
                f"{like_max if like_max is not None else float('nan'):.1e}]",
                flush=True
            )

    # 4) update() 里把 meas 传给 _build_meas_std，确保 v_ref=meas["sog"] 生效
    def update(self, meas: dict, *, age: float = 0.0):
        """
        STAGE-4: update() keeps API but uses:
          - Q gap diffusion first (R->Q shift)
          - piecewise R for pos/sog + yaw anneal/gate
        """
        if age >= float(self.cfg.noise.age.age_max):
            return
        if self.x_particles is None or self.w is None:
            return

        # Q inflation first (R->Q shift)
        self._apply_gap_process_noise(age, meas)

        yaw_innov_abs = None
        try:
            yaw_innov_abs = abs(_angle_diff(_meas_yaw(meas), float(self.x[3])))
        except Exception:
            yaw_innov_abs = None

        sigma_pos, sigma_sog, sigma_yaw = self._build_meas_std(age, meas=meas, yaw_innov_abs=yaw_innov_abs)
        try:
            meas["age"] = float(age)
        except Exception:
            pass

        self._meas_likelihood_update(
            meas, sigma_pos, sigma_sog, sigma_yaw,
            age=float(age),
            yaw_innov_abs=yaw_innov_abs,
        )

        try:
            meas["age"] = float(age)
        except Exception:
            pass
        self._meas_likelihood_update(
            meas, sigma_pos, sigma_sog, sigma_yaw,
            age=float(age),
            yaw_innov_abs=yaw_innov_abs,
        )

    def step_delay_robust(self, env_time: float, meas: Dict[str, Any], age: float):
        """
        STAGE-4: Delay-robust update on PF axis (ENV time).

        Args:
            env_time: Current Environment Time (t_now). PF predicts to this time.
            meas:     Measurement dict containing x, y, sog, yaw (yaw_sim_rad).
            age:      Measurement staleness on PF axis (t_now - t_meas_source).
        """
        msg_id = meas.get("msg_id", None)
        if os.environ.get("PF_ID_CHECK", "0") == "1":
            print(f"[PF-ID-CHECK] t={float(env_time):.2f} msg_id={msg_id}", flush=True)

        # (1) Age gating first (avoid polluting dedup sets with too-old msgs)
        if float(age) >= float(self.cfg.noise.age.age_max):
            return {"accepted": False, "reason": "too_old"}

        env_time = float(env_time)
        age = float(max(0.0, age))

        # (2) msg_id dedup (strict)
        if msg_id is not None:
            if msg_id in self.processed_msg_ids:
                print(f"[PF-INTERCEPT] Blocked duplicate msg_id={msg_id} at t={env_time}")
                return {"accepted": False, "reason": "duplicate_id_intercepted"}
            self.processed_msg_ids.add(msg_id)
            if len(self.processed_msg_ids) > 200:
                self.processed_msg_ids.clear()
                self.processed_msg_ids.add(msg_id)

        # (3) fingerprint/ts dedup (double counting protection)
        meas_source_ts = env_time - age
        current_fingerprint = (
            round(meas_source_ts, 6),
            round(float(meas["x"]), 3),
            round(float(meas["y"]), 3),
            round(float(_meas_yaw(meas)), 4),
        )
        if self.last_update_fingerprint == current_fingerprint:
            return {"accepted": False, "reason": "duplicate_fingerprint"}
        if abs(meas_source_ts - float(self.last_meas_update_ts)) < 1e-6:
            return {"accepted": False, "reason": "duplicate_ts"}
        self.last_meas_update_ts = meas_source_ts

        # (4) init
        if self.x_particles is None or self.w is None:
            self.init_from_meas(env_time, meas)
            self.last_update_fingerprint = current_fingerprint
            return {"accepted": True, "type": "init"}

        # (5) enforce monotonic PF axis (no rollback)
        if self.last_ts is not None and env_time < float(self.last_ts) - 1e-9:
            return {"accepted": False, "reason": "time_rollback"}

        # (6) predict to now; dt_pred computed from pre-predict last_ts
        prev_ts = None if (self.last_ts is None) else float(self.last_ts)
        self.predict_to(env_time)
        dt_pred = None if (prev_ts is None) else max(0.0, env_time - prev_ts)

        # (7) R->Q shift: gap diffusion before update
        self._apply_gap_process_noise(age, meas)

        # yaw innovation for gating
        yaw_innov_abs = None
        try:
            yaw_innov_abs = abs(_angle_diff(_meas_yaw(meas), float(self.x[3])))
        except Exception:
            yaw_innov_abs = None

        # build R
        sigma_pos, sigma_sog, sigma_yaw = self._build_meas_std(age, meas=meas, yaw_innov_abs=yaw_innov_abs)
        try:
            meas["age"] = float(age)
        except Exception:
            pass

        # update
        self._meas_likelihood_update(
            meas, sigma_pos, sigma_sog, sigma_yaw,
            age=float(age),
            yaw_innov_abs=yaw_innov_abs,
            dt=dt_pred,
        )

        self.last_ts = env_time
        self.last_update_fingerprint = current_fingerprint
        return {"accepted": True, "type": "update"}

    # ------------------------------------------------------------------
    # resampling & estimate
    # ------------------------------------------------------------------

    def _resample_systematic(self):
        N = self.N
        positions = (self.rng.random() + np.arange(N)) / N
        cumsum_w = np.cumsum(self.w)
        idx = np.zeros(N, dtype=int)

        i = 0
        j = 0
        while i < N:
            if positions[i] < cumsum_w[j]:
                idx[i] = j
                i += 1
            else:
                j += 1

        self.x_particles = self.x_particles[idx]
        self.w[:] = 1.0 / float(N)

    def _update_estimate(self):
        """Compute public estimate self.x using weighted mean; yaw uses circular mean."""
        if self.x_particles is None or self.w is None:
            return

        w = self.w
        xp = self.x_particles

        px = float(np.average(xp[:, 0], weights=w))
        py = float(np.average(xp[:, 1], weights=w))
        v  = float(np.average(xp[:, 2], weights=w))

        s = float(np.sum(w * np.sin(xp[:, 3])))
        c = float(np.sum(w * np.cos(xp[:, 3])))
        yaw = float(math.atan2(s, c))
        yaw = _wrap_pi(yaw)

        yawd = float(np.average(xp[:, 4], weights=w))
        self.x = np.array([px, py, v, yaw, yawd], dtype=float)

    def refresh_estimate(self):
        self._update_estimate()

    # ------------------------------------------------------------------
    # soft relock (expects yaw_sim_rad; PF-axis time)
    # ------------------------------------------------------------------

    def soft_relock(
        self,
        meas_ts: float,
        meas: Dict[str, Any],
        age: float,
        *,
        pos_thr_m: float = 25.0,
        beta_pos: float = 0.35,
        beta_vel: float = 0.25,
        beta_yaw: float = 0.25,
        jitter_pos_m: float = 1.0,
        jitter_vel_mps: float = 0.05,
        jitter_yaw_rad: float = math.radians(0.5),
        reset_weights: bool = True,
    ) -> bool:
        if self.x_particles is None or self.w is None or self.last_ts is None:
            return False
        if age >= float(self.cfg.noise.age.age_max):
            return False

        meas_ts = float(meas_ts)

        z_x = float(meas["x"])
        z_y = float(meas["y"])
        z_sog = float(meas["sog"])
        z_yaw = float(meas.get("yaw", meas.get("cog", 0.0)))
        z_yaw = _wrap_pi(z_yaw)

        est_px, est_py, est_v, est_yaw, est_yawd = self.x.tolist()
        dist = math.hypot(z_x - est_px, z_y - est_py)
        if dist < float(pos_thr_m):
            return False

        px = self.x_particles[:, 0]
        py = self.x_particles[:, 1]
        v  = self.x_particles[:, 2]
        yaw = self.x_particles[:, 3]
        yawd = self.x_particles[:, 4]

        px += beta_pos * (z_x - px)
        py += beta_pos * (z_y - py)
        v  += beta_vel * (z_sog - v)

        dpsi = _wrap_pi_np(z_yaw - yaw)
        yaw += beta_yaw * dpsi
        yaw = _wrap_pi_np(yaw)

        # yawd injection (weak)
        dt_eff = max(0.5, meas_ts - float(self.last_ts))
        dpsi_rate = dpsi / float(dt_eff)
        yawd_target = np.clip(dpsi_rate, -float(self.cfg.noise.yawd_clip), float(self.cfg.noise.yawd_clip))

        beta_yawd = 0.15
        yawd = (1.0 - beta_yawd) * yawd + beta_yawd * yawd_target
        yawd *= 0.98
        yawd += self.rng.normal(0.0, 0.01, size=self.N)
        yawd = np.clip(yawd, -float(self.cfg.noise.yawd_clip), float(self.cfg.noise.yawd_clip))

        # jitter
        if jitter_pos_m > 0.0:
            px += self.rng.normal(0.0, jitter_pos_m, size=self.N)
            py += self.rng.normal(0.0, jitter_pos_m, size=self.N)
        if jitter_vel_mps > 0.0:
            v += self.rng.normal(0.0, jitter_vel_mps, size=self.N)
        if jitter_yaw_rad > 0.0:
            yaw += self.rng.normal(0.0, jitter_yaw_rad, size=self.N)
            yaw = _wrap_pi_np(yaw)

        self.x_particles[:, 0] = px
        self.x_particles[:, 1] = py
        self.x_particles[:, 2] = v
        self.x_particles[:, 3] = yaw
        self.x_particles[:, 4] = yawd

        if reset_weights:
            self.w[:] = 1.0 / float(self.N)

        # keep PF time monotonic: snap to meas_ts (caller ensures monotonic)
        if self.last_ts is None or meas_ts >= float(self.last_ts) - 1e-9:
            self.last_ts = meas_ts

        self._update_estimate()
        return True
