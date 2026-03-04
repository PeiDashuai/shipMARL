import math, numpy as np
from ..utils.math import wrap_to_pi

def progress_phi(ships):
    """势函数：-distance_to_goal。"""
    return -np.array([np.linalg.norm(s.pos - s.goal) for s in ships], dtype=np.float64)

def task_reward(ships, dpsi_rl, last_phi, risk, v_max, step_cost, arrival_bonus):
    """核心任务奖励（进度差分 + 对齐/直行 + 步长代价 + 到达 + 轻微停滞罚）。"""
    N = len(ships)
    prog = progress_phi(ships)
    if last_phi is None or getattr(last_phi, "shape", (0,))[0] != N:
        last_phi = prog.copy()
    dphi = np.clip(prog - last_phi, -3.0, 3.0)
    new_last_phi = prog.copy()

    # --- goal bearing / distance ---
    goal_vec = np.array([s.goal - s.pos for s in ships], dtype=np.float64)      # (N,2)
    to_goal_norm = np.linalg.norm(goal_vec, axis=1)                             # (N,)
    dir_goal = np.arctan2(goal_vec[:, 1], goal_vec[:, 0])                       # (N,)
    psi = np.array([s.psi for s in ships], dtype=np.float64)
    head_err = np.abs(np.vectorize(wrap_to_pi)(dir_goal - psi))                 # 朝向误差

    risk_low = np.maximum(0.0, 0.25 - risk) / 0.25  # 抑制在高风险时的鼓励项：
    align_ok = np.maximum(0.0, 1.0 - head_err / math.radians(25.0))                 # 对齐度

    v_self = np.array([s.v for s in ships], dtype=np.float64)

    # ------------------------------------------------------------
    # Strong “toward-goal” shaping (dense + directional)
    # ------------------------------------------------------------
    # Velocity projection onto goal direction: v_toward = v * cos(heading_error)
    goal_hat = goal_vec / np.maximum(to_goal_norm[:, None], 1e-6)
    v_vec = np.stack([v_self * np.cos(psi), v_self * np.sin(psi)], axis=1)
    v_toward = np.sum(v_vec * goal_hat, axis=1)
    v_toward_norm = np.clip(v_toward / max(v_max, 1e-6), -1.0, 1.0)

    # Distance-aware shaping:
    #   - Far from goal: encourage speed + closing velocity (learning signal is strong).
    #   - Near goal: discourage high speed (reduce overshoot / orbiting around goal_tol).
    near_radius = 80.0
    near_w = np.clip((near_radius - to_goal_norm) / max(near_radius, 1e-6), 0.0, 1.0)  # 1 near, 0 far
    far_w  = 1.0 - near_w

    speed_ratio = np.clip(v_self / max(v_max, 1e-6), 0.0, 1.0)

    # Closing-velocity reward (positive term attenuated near goal to avoid "barrel into goal" overshoot).
    v_toward_gate = 0.25 + 0.75 * far_w   # 0.25 near, 1.0 far
    w_toward_pos = 1.6
    w_toward_neg = 2.0
    r_toward = risk_low * (
        w_toward_pos * np.maximum(0.0, v_toward_norm) * v_toward_gate +
        w_toward_neg * np.minimum(0.0, v_toward_norm)
    )

    # Heading alignment: slightly stronger near goal, but keep it modest (progress term should dominate).
    w_heading = 0.18
    r_heading = w_heading * risk_low * (0.6 + 0.4 * near_w) * np.clip(np.cos(head_err), 0.0, 1.0) * speed_ratio

    # Penalize high speed near goal to increase capture probability within goal_tol.
    w_slow_near = 0.8
    r_slow_near = -w_slow_near * risk_low * near_w * (speed_ratio ** 2)

    dpsi_scale = math.radians(20.0)  # 或从 env 传入 dpsi_max
    straight_term = 1.0 - (np.abs(dpsi_rl) / max(dpsi_scale, 1e-6))

    # Gate "straight/boost" mainly to far regime; near goal we don't want "go fast" bonuses.
    r_straight = 0.12 * risk_low * align_ok * far_w * np.clip(straight_term, 0.0, 1.0)
    r_boost = 0.08 * risk_low * align_ok * far_w * np.maximum(0.0, (v_self / v_max - 0.60))


    step_cost_vec = step_cost * np.array([not s.reached for s in ships], dtype=np.float64) # 未到达才收取固定步长代价：抑制无休止拖延；促使尽快完成任务。

    # Make progress dominate (potential-based, hard to game).
    w_prog = 4.0
    r_task = w_prog * dphi - step_cost_vec + arrival_bonus + r_straight + r_boost + r_toward + r_heading + r_slow_near
    # Penalize stagnation mostly when far; near goal small dphi is normal during fine alignment.
    r_task = r_task - 0.05 * far_w * (np.abs(dphi) < 0.02)
    return r_task, new_last_phi
