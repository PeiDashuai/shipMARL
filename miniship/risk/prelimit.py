import numpy as np

def speed_soft_cap_before_step(
    ships, tc0, dc0, v_max, v_min, risk_T_thr, risk_D_thr,
    thrRisk_gate, vCruiseK, alphaOpen, capPow, capGain,
    K_release, M_boost, dvBoost,
    guard_safeCount, guard_boostCount,
    v_cmd
):
    """
    基于推进前风险（bestX0）计算速度软限 v_cap，并生成最终 v_target。
    同时维护连续安全释放(guard_safeCount)与加速回血(guard_boostCount)。
    """
    N = len(ships)
    # 1) 先尊重策略速度
    v_target = np.clip(v_cmd.astype(np.float64), v_min, v_max)

    # 2) 仍然可以保留 v_cruise 作为“风险上界基准”或直接用 v_max（看你设计）
    v_cruise = vCruiseK * v_max

    bestX0 = np.zeros(N, dtype=np.float64)
    close_n0 = np.zeros(N, dtype=np.float64)

    active = [i for i,s in enumerate(ships) if not s.reached]
    v_list = [np.array([np.cos(s.psi), np.sin(s.psi)]) * s.v for s in ships]

    for ii in active:
        others = [j for j in active if j != ii]
        bx = -1.0; bj = -1
        for j in others:
            tij = max(0.0, tc0[ii, j])
            dij = dc0[ii, j]
            tn = max(0.0, min(1.0, (risk_T_thr - tij)/risk_T_thr))
            dn = max(0.0, min(1.0, (risk_D_thr - dij)/risk_D_thr))
            xij = max(tn, dn)
            if xij > bx:
                bx = xij; bj = j
        bestX0[ii] = max(0.0, (bx - thrRisk_gate) / max(1.0 - thrRisk_gate, 1e-6))
        if bj >= 0:
            vi = v_list[ii]; vj = v_list[bj]
            pij = ships[bj].pos - ships[ii].pos
            vij = vj - vi
            cls = max(0.0, - float(np.dot(pij, vij)) / max(np.linalg.norm(pij), 1e-6))
            close_n0[ii] = min(1.0, cls / max(v_max, 1e-6))

    g = np.power(bestX0 * (close_n0 > 1e-6), capPow)
    v_cap = v_cruise * (1.0 - capGain * g)
    v_cap = np.clip(v_cap, v_min, v_max)

    for i in active:
        in_conflict = (bestX0[i] > 0.0)
        if in_conflict:
            guard_safeCount[i] = 0
            # 关键：cap 只是上界，不再把速度拉回巡航
            v_target[i] = min(v_target[i], v_cap[i])
        else:
            guard_safeCount[i] += 1
            if guard_safeCount[i] >= K_release:
                guard_boostCount[i] = max(guard_boostCount[i], M_boost)

    return v_target, v_cap, bestX0
