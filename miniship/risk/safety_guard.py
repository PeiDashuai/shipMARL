# ===== risk/safety_guard.py (UNIFIED GUARD: hard / ghost / hybrid) =====
import numpy as np
from ..risk.tcpa_dcpa import tcpa_dcpa_matrix

def _wrap_pi(x): 
    return (x + np.pi) % (2*np.pi) - np.pi

def _guard_core(
    ships, dpsi, v_cmd, dt,
    dpsi_max, v_min, v_max,
    risk_T_thr, risk_D_thr, collide_thr,
    # 预计算态（可选，减少重复开销；若为 None 则内部计算）
    pos=None, psi=None, v_all=None, tc=None, dc=None,
    # 行为参数
    yaw_rate_max=None, brake_gain=0.4, steer_gain=0.4,
    # 触发规则参数
    dist_soft_gate=0.40, time_soft_gate=0.40, dist_hard_gamma=0.90,
    # 近邻裁剪与算子
    K_guard: int = 6,
    use_float32: bool = True,
    # 是否实际应用修正（True=Hard；False=Ghost）
    apply_correction: bool = True,
):
    """
    返回：
      dpsi_out: (N,) 修正后的艏向增量（Hard模式）或原样（Ghost模式）
      v_out   : (N,) 修正后的速度（Hard模式）或原样（Ghost模式）
      trig    : (N,) 每条船是否触发护栏（0/1）
      cand_trig: (N,K) 候选对触发矩阵（便于统计/可选）
    """
    _dtype = np.float32 if use_float32 else np.float64
    dpsi  = np.asarray(dpsi,  dtype=_dtype).copy()
    v_cmd = np.asarray(v_cmd, dtype=_dtype).copy()
    N = len(ships)

    # (0) yaw-rate -> 本步角度限幅
    if yaw_rate_max is not None and yaw_rate_max > 0:
        dpsi_cap = _dtype(yaw_rate_max * dt)
        dpsi[:] = np.clip(dpsi, -dpsi_cap, dpsi_cap)

    # (1) 取态（尽量外部传入，避免重复构造）
    if pos is None:
        pos = np.stack([s.pos for s in ships], 0).astype(_dtype)            # [N,2]
    else:
        pos = pos.astype(_dtype)
    if psi is None:
        psi = np.array([s.psi for s in ships], dtype=_dtype)                # [N]
    else:
        psi = psi.astype(_dtype)
    if v_all is None:
        v_all = np.array([s.v for s in ships], dtype=_dtype)                # [N]
    else:
        v_all = v_all.astype(_dtype)

    vel = np.stack([
        [np.cos(psi[i])*v_all[i], np.sin(psi[i])*v_all[i]] for i in range(N)
    ], 0).astype(_dtype)                                                    # [N,2]

    # (2) 风险矩阵（若外部未给，内部计算一次）
    if tc is None or dc is None:
        tc, dc, _ = tcpa_dcpa_matrix(ships)  # 一次 O(N^2)
    # 统一到 dtype（避免隐式 upcast）
    tc = tc.astype(_dtype); dc = dc.astype(_dtype)

    # (3) vj_max 向量化
    v_mat = np.broadcast_to(v_all, (N, N)).astype(_dtype)
    np.fill_diagonal(v_mat, -np.inf)
    vj_max = np.max(v_mat, axis=1)   # [N]

    # (4) D_safe
    tau, alpha = _dtype(6.0), _dtype(0.35)
    D_safe = np.maximum(_dtype(collide_thr),
                        _dtype(0.5)*vj_max*tau + alpha*np.minimum(v_all, _dtype(v_max))*tau)

    # (5) 仅取每船 K 个最近邻（按 DCPA）
    dc_masked = dc.copy()
    np.fill_diagonal(dc_masked, np.inf)
    K = min(int(K_guard), max(0, N-1))
    if K == 0:
        # 无邻居：返回夹持后的原动作与 0 触发
        dpsi_out = np.clip(dpsi, -dpsi_max, dpsi_max).astype(np.float64)
        v_out    = np.clip(v_cmd, v_min, v_max).astype(np.float64)
        trig     = np.zeros(N, dtype=np.float64)
        cand_trig= np.zeros((N,0), dtype=bool)
        return dpsi_out, v_out, trig, cand_trig

    knn_idx = np.argpartition(dc_masked, K, axis=1)[:, :K]   # [N,K]
    idx = np.arange(N)

    # (6) 风险归一
    tn = np.clip((risk_T_thr - np.clip(tc, 0.0, None)) / max(risk_T_thr, 1e-6), 0.0, 1.0)
    dn = np.clip((risk_D_thr - dc) / max(risk_D_thr, 1e-6), 0.0, 1.0)

    # (7) 相对几何（只在 [N,K] 里算）
    i_rep = np.repeat(idx[:, None], K, axis=1)          # [N,K]
    j_rep = knn_idx                                     # [N,K]
    pij = pos[j_rep] - pos[i_rep]                       # [N,K,2]
    vij = vel[j_rep] - vel[i_rep]                       # [N,K,2]
    rel_norm = np.linalg.norm(pij, axis=-1) + _dtype(1e-6)   # [N,K]
    closing = - np.sum(pij * vij, axis=-1) / rel_norm         # [N,K]

    time_ok    = (tc[i_rep, j_rep] >= 0.0) & (tc[i_rep, j_rep] <= _dtype(risk_T_thr))
    closing_ok = (closing > 0.0)
    hard_dist  = (dc[i_rep, j_rep] < (D_safe[:, None] * _dtype(dist_hard_gamma)))
    soft_both  = (tn[i_rep, j_rep] > _dtype(time_soft_gate)) & (dn[i_rep, j_rep] > _dtype(dist_soft_gate))

    cand_trig  = (closing_ok & time_ok) & (hard_dist | soft_both)   # [N,K]
    trig_bool  = np.any(cand_trig, axis=1)                          # [N] bool
    trig       = trig_bool.astype(np.float64)

    # (8) Hard：真正改写；Ghost：仅返回触发，不改动作
    dpsi_out = dpsi.copy()
    v_out    = v_cmd.copy()

    if apply_correction and np.any(trig_bool):
        for i in range(N):
            if not trig_bool[i]:
                continue
            # 候选中选评分最高的威胁 j*
            Js = j_rep[i][cand_trig[i]]
            if Js.size == 0:
                continue
            sc_d = (risk_D_thr - dc[i, Js]) / max(risk_D_thr, 1e-6)
            sc_t = (risk_T_thr - np.maximum(0.0, tc[i, Js])) / max(risk_T_thr, 1e-6)
            sc = 0.5 * sc_d + 0.5 * sc_t
            j_star = Js[int(np.argmax(sc))]

            rel = (pos[j_star] - pos[i]).astype(_dtype)
            bearing = _wrap_pi(np.arctan2(rel[1], rel[0]) - psi[i])

            # 紧急降速 + 反侧转向（威胁在右→左转）
            v_out[i]   = max(v_min, _dtype(brake_gain) * v_out[i])
            dpsi_out[i]= np.clip(dpsi_out[i] - np.sign(bearing) * _dtype(steer_gain) * dpsi_max,
                                 -dpsi_max, dpsi_max)

    # 基本夹持
    dpsi_out = np.clip(dpsi_out, -dpsi_max, dpsi_max).astype(np.float64)
    v_out    = np.clip(v_out,    v_min,     v_max).astype(np.float64)
    return dpsi_out, v_out, trig, cand_trig


# --------- 对外包装：三种模式 ---------
def safety_guard_hard(*args, **kwargs):
    kwargs = dict(kwargs)
    kwargs["apply_correction"] = True
    return _guard_core(*args, **kwargs)

def safety_guard_ghost(*args, **kwargs):
    kwargs = dict(kwargs)
    kwargs["apply_correction"] = False
    return _guard_core(*args, **kwargs)

def safety_guard_hybrid(
    *args,
    mode_prob: float = 0.5,    # 在 hybrid 下以此概率选择 hard，否则 ghost
    rng: np.random.Generator | None = None,
    **kwargs
):
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < float(mode_prob):
        kwargs["apply_correction"] = True
    else:
        kwargs["apply_correction"] = False
    return _guard_core(*args, **kwargs)
