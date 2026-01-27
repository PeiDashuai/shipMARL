# miniship/reward/cost.py

import math, numpy as np
from ..utils.math import wrap_to_pi

def cost_near(risk, vj_max, v_self, dmin_i, collide_thr, tau=4.0):
    """
    精简、鲁棒版 near cost：
      - 用一个适度的安全距离 D_safe
      - 线性 risk 门控
      - 归一化违约 m^2
    """
    # 1) risk 门控系数：低风险基本关断，高风险放大
    #    risk <= 0.3 => k_r ~ 0.3；risk -> 1 => k_r -> 0.8
    gate_risk = np.clip((risk - 0.3) / (1.0 - 0.3), 0.0, 1.0)
    k_r = 0.3 + (0.8 - 0.3) * gate_risk   # [0.3, 0.8]

    # 2) 近似相对速度：这里不用 v_rel 的几何，先用 vj_max 做保守上界
    v_rel_eff = np.minimum(vj_max + 0.5 * v_self, 4.0)  # 简单限幅，防 D_safe 爆炸

    # 3) 安全距离：硬碰撞半径 + 速度项
    D_safe = collide_thr + k_r * v_rel_eff * tau

    # 4) 归一化违约
    m = np.maximum(0.0, (D_safe - dmin_i) / np.maximum(D_safe, 1e-9))
    return m * m


def cost_rule(ships, tc, dc, risk_T_thr, risk_D_thr, thHeadOn, thCross, dpsi_rl, dv_act, dv_max):
    N = len(ships)
    c_rule = np.zeros(N, dtype=np.float64)
    active = [i for i,s in enumerate(ships) if not s.reached]
    for i in active:
        bx, bj, bbrg = -1.0, -1, 0.0
        others = [j for j in active if j != i]
        for j in others:
            tij, dij = tc[i, j], dc[i, j]
            tn = max(0.0, min(1.0, (risk_T_thr - max(0.0, tij))/risk_T_thr))
            dn = max(0.0, min(1.0, (risk_D_thr - dij)/risk_D_thr))
            xij = max(tn, dn)
            if xij > bx:
                bx = xij; bj = j
                rel = ships[j].pos - ships[i].pos
                bbrg = wrap_to_pi(math.atan2(rel[1], rel[0]) - ships[i].psi)
        if bj < 0: continue
        g_conf = max(0.0, (bx - 0.35) / max(1.0 - 0.35, 1e-6))
        if g_conf <= 0.0: continue

        isHeadOn = (abs(bbrg) <= thHeadOn)
        inHalf   = (abs(bbrg) <= thCross)
        isStarX  = (bbrg > 0.0) and inHalf
        isPortX  = (bbrg < 0.0) and inHalf

        dpsi_scale = math.radians(20.0)  # 或从 env 传入
        lt  = max(0.0,  dpsi_rl[i] / max(dpsi_scale, 1e-6))
        rt  = max(0.0, -dpsi_rl[i] / max(dpsi_scale, 1e-6))

        acc = max(0.0,  dv_act[i] / max(dv_max, 1e-6))
        dec = max(0.0, -dv_act[i] / max(dv_max, 1e-6))

        viol = 0.0
        if isHeadOn or isStarX:
            viol = 1.2 * lt + 0.9 * acc
        elif isPortX:
            viol = 0.6 * (lt + rt) + 0.4 * (acc + dec)

        c_rule[i] = g_conf * viol
    return c_rule

def cost_coll_time(ships, collide_thr, t, T_max):
    N = len(ships)
    coll = False; pair=(-1,-1)
    for i in range(N-1):
        for j in range(i+1, N):
            if np.linalg.norm(ships[i].pos - ships[j].pos) < collide_thr:
                coll=True; pair=(i,j); break
        if coll: break
    timeout = (t >= T_max)
    c_coll = np.zeros(N, np.float64)
    if coll:
        c_coll[pair[0]] = 1.0; c_coll[pair[1]] = 1.0
    c_time = np.zeros(N, np.float64)
    if timeout: c_time[:] = 1.0
    return coll, pair, c_coll, c_time, timeout
