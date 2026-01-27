import numpy as np

def post_step_risk(ships, tc, dc, risk_T_thr, risk_D_thr):
    """推进后：每船 risk ∈ [0,1]、dmin_i、对手最大速 vj_max。"""
    N = len(ships)
    risk = np.zeros(N, np.float64)
    dmin_i = np.full(N, np.inf, np.float64)
    vj_mx = np.zeros(N, np.float64)
    active = [i for i,s in enumerate(ships) if not s.reached]
    for ii in active:
        others = [j for j in active if j != ii]
        if not others: continue
        tmin = np.min(tc[ii, others]) if np.isfinite(tc[ii, others]).any() else np.inf
        dmin = np.min(dc[ii, others]) if np.isfinite(dc[ii, others]).any() else np.inf
        dmin_i[ii] = dmin
        tn = max(0.0, min(1.0, (risk_T_thr - tmin)/risk_T_thr))
        dn = max(0.0, min(1.0, (risk_D_thr - dmin)/risk_D_thr))
        risk[ii] = max(tn, dn)
        vmax = 0.0
        for j in others:
            vmax = max(vmax, ships[j].v)
        vj_mx[ii] = vmax
    return risk, dmin_i, vj_mx
