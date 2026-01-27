# miniship/reward/lagrangian.py

# === reward/lagrangian.py (DROP-IN REPLACEMENT) ===
from dataclasses import dataclass
import numpy as np

@dataclass
class DualState:
    # ---- lambdas ----
    lam_near: float = 3.0
    lam_rule: float = 2.0
    lam_coll: float = 0.0    # ← 关闭碰撞对偶
    lam_time: float = 1.0

    # ---- EMA states for constraints ----
    ema_near: float = 0.33
    ema_rule: float = 0.33
    ema_coll: float = 0.0    # 保留字段，仅不参与更新
    ema_time: float = 0.0

    # ---- targets for constraints ----
    ctarget_near: float = 0.06
    ctarget_rule: float = 0.02
    ctarget_coll: float = 0.0
    ctarget_time: float = 0.10

    # ---- base step sizes ----
    beta: float = 0.95       # EMA 衰减（约 episode 级）
    eta_near: float = 0.02
    eta_rule: float = 0.02
    eta_coll: float = 0.00   # ← 保留，但无效
    eta_time: float = 0.01

    # ---- pacing & safety knobs ----
    eta_scale: float = 0.33
    band_near: float = 0.015
    band_rule: float = 0.02
    band_coll: float = 0.01
    band_time: float = 0.02
    decay_in_band: float = 0.99
    rel_cap: float = 0.015

    # ---- hard caps ----
    lam_near_max: float = 3.06
    lam_rule_max: float = 2.30
    lam_coll_max: float = 0.0   # ← 不允许上升
    lam_time_max: float = 1.80

    # ---- performance EMA (episode-level) ----
    perf_beta: float = 0.90
    succ_ma: float = 0.0
    coll_ma: float = 1.0
    tout_ma: float = 1.0

    # performance targets (train distribution)
    succ_target: float = 0.85
    coll_target: float = 0.06
    tout_target: float = 0.12
    perf_tol: float = 0.02

    # governor gains
    gov_down: float = 0.98
    gov_up: float   = 1.02
    gov_eta_decay: float = 0.95
    eta_scale_min: float = 0.15
    eta_scale_max: float = 0.50

    # hysteresis for (disabled) collision lambda
    coll_hyst_hi: float = 0.10
    coll_hyst_lo: float = 0.06

    # floors + hysteresis
    lam_near_floor: float = 1.50
    lam_rule_floor: float = 0.30
    lam_coll_floor: float = 0.00
    lam_time_floor: float = 0.40

    floor_band_near: float = 0.05
    floor_band_rule: float = 0.05
    floor_band_coll: float = 0.40
    floor_band_time: float = 0.05


def combine_reward(r_task, c_near, c_rule, c_coll, c_time, dual: DualState, clip=None):
    """碰撞不参与对偶；仅作为硬约束/终端项存在。"""
    r = (
        r_task
        - dual.lam_near * c_near
        - dual.lam_rule * c_rule
        # - dual.lam_coll * c_coll   # ← 禁用
        - dual.lam_time * c_time
    )
    return np.clip(r, -clip, clip) if clip is not None else r


def _update_one_lambda(lam, ema, target, eta, band,
                       dual_min, dual_max, rel_cap,
                       eta_scale, decay_in_band, floor=0.0, floor_band=0.0):
    """
    目标跟踪 + 死区带 + 相对步幅限幅 + 地板带 的单λ更新。
    """
    err = ema - target

    # 目标带内：仅衰减（防过保守）
    if abs(err) <= band:
        lam = lam * decay_in_band
        if lam <= floor + floor_band:  # 钩住地板带
            lam = max(lam, floor)
        return float(np.clip(lam, dual_min, dual_max))

    # 目标带外：按误差更新（带全局节流）
    delta = eta_scale * eta * err

    # 相对步幅限幅：|Δλ| ≤ rel_cap * max(lam, ε)
    base = max(lam, 1e-6)
    max_abs_delta = rel_cap * base
    delta = float(np.clip(delta, -max_abs_delta, max_abs_delta))

    lam = lam + delta
    lam = float(np.clip(lam, dual_min, dual_max))
    return lam


def _update_coll_with_hysteresis(lam, ema_coll=None, dual: DualState=None, **_ignored):
    """
    保留兼容接口；碰撞λ实际被禁用。
    """
    if dual is None:
        return lam
    # 强制夹到 [floor, max] 且 max=0 => 恒为 0
    return float(np.clip(max(lam, dual.lam_coll_floor), dual.lam_coll_floor, dual.lam_coll_max))


def dual_update_end_of_episode(c_near_mean, c_rule_mean, c_coll_max, c_time_max, dual: DualState):
    # ---- 更新约束 EMA（不含碰撞）----
    b = dual.beta
    dual.ema_near = b * dual.ema_near + (1 - b) * float(c_near_mean)
    dual.ema_rule = b * dual.ema_rule + (1 - b) * float(c_rule_mean)
    # dual.ema_coll = b * dual.ema_coll + (1 - b) * float(c_coll_max)  # 禁用
    dual.ema_time = b * dual.ema_time + (1 - b) * float(c_time_max)

    # ---- 目标归一化对偶更新 ----
    dual.lam_near = _update_one_lambda(
        lam=dual.lam_near, ema=dual.ema_near, target=dual.ctarget_near,
        eta=dual.eta_near, band=dual.band_near,
        dual_min=0.0, dual_max=dual.lam_near_max, rel_cap=dual.rel_cap,
        eta_scale=dual.eta_scale, decay_in_band=dual.decay_in_band,
        floor=dual.lam_near_floor, floor_band=dual.floor_band_near,
    )
    dual.lam_rule = _update_one_lambda(
        lam=dual.lam_rule, ema=dual.ema_rule, target=dual.ctarget_rule,
        eta=dual.eta_rule, band=dual.band_rule,
        dual_min=0.0, dual_max=dual.lam_rule_max, rel_cap=dual.rel_cap,
        eta_scale=dual.eta_scale, decay_in_band=dual.decay_in_band,
        floor=dual.lam_rule_floor, floor_band=dual.floor_band_rule,
    )
    # dual.lam_coll = _update_coll_with_hysteresis(lam=dual.lam_coll, ema_coll=dual.ema_coll, dual=dual)

    dual.lam_time = _update_one_lambda(
        lam=dual.lam_time, ema=dual.ema_time, target=dual.ctarget_time,
        eta=dual.eta_time, band=dual.band_time,
        dual_min=0.0, dual_max=dual.lam_time_max, rel_cap=dual.rel_cap,
        eta_scale=dual.eta_scale, decay_in_band=dual.decay_in_band,
        floor=dual.lam_time_floor, floor_band=dual.floor_band_time,
    )


def dual_govern_by_performance(dual: DualState, succ_ep: float, coll_ep: float, tout_ep: float):
    """
    轻量治理：仅调节 eta_scale & 少量楼层钩住。
    在整体性能好时退火 eta_scale，让后期对偶更新更平滑。
    """
    b = dual.perf_beta  # 使用 DualState 的参数
    if not hasattr(dual, "ema_succ"):
        dual.ema_succ = float(succ_ep)
        dual.ema_coll = float(coll_ep)
        dual.ema_tout = float(tout_ep)
        dual.eta_scale = float(getattr(dual, "eta_scale", 1.0))
    else:
        dual.ema_succ = b * dual.ema_succ + (1 - b) * float(succ_ep)
        dual.ema_coll = b * dual.ema_coll + (1 - b) * float(coll_ep)
        dual.ema_tout = b * dual.ema_tout + (1 - b) * float(tout_ep)

    # 使用 DualState 中的 target，而不是硬编码 0.9/0.05/0.10
    succ_ok = (dual.ema_succ >= dual.succ_target)
    coll_ok = (dual.ema_coll <= dual.coll_target)
    tout_ok = (dual.ema_tout <= dual.tout_target)

    freeze = (succ_ok and coll_ok and tout_ok)

    if freeze:
        # 性能已经很好：逐步减小 eta_scale（退火）
        dual.eta_scale = max(dual.eta_scale_min, dual.eta_scale * dual.gov_eta_decay)
    else:
        # 性能还不够好：略微恢复一点灵敏度
        dual.eta_scale = min(dual.eta_scale_max, dual.eta_scale * 1.02)

    # 钩住楼层，避免继续削弱安全性
    dual.lam_near = max(dual.lam_near, dual.lam_near_floor)
    dual.lam_rule = max(dual.lam_rule, dual.lam_rule_floor)
    dual.lam_time = max(dual.lam_time, dual.lam_time_floor)
    dual.lam_coll = 0.0  # 硬置零，更明确
