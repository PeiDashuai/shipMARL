import numpy as np

def finalize_episode_rewards(r_vec, success_all, timeout, t, T_max, spawn_len,
                             success_bonus=350., timeout_base=120., collision_penalty=120.):
    """统一收尾奖励/惩罚与 term_code。"""
    term_code = 0
    if success_all:
        time_bonus = 150.0 * (1.0 - min(t / T_max, 1.0))
        r_vec = r_vec + success_bonus + time_bonus
        term_code = 2
    elif timeout:
        # 与未到达距离成比例的惩罚：这里外部应当已把 dist 影响体现在 r_task 中
        pen_tout = timeout_base + 0.6 * (160.0 / max(spawn_len, 1.0)) * 160.0  # 保持与原版近似
        r_vec = r_vec - pen_tout
        term_code = 3
    else:
        r_vec = r_vec - collision_penalty
        term_code = 1
    return r_vec, term_code
