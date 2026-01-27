# risk/safety_guard_strict.py
import numpy as np
from ..risk.tcpa_dcpa import tcpa_dcpa_matrix

def _wrap_pi(x): 
    return (x + np.pi) % (2*np.pi) - np.pi

def _clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def _dcpa_tcpa(p_i, v_i, p_j, v_j):
    """标量 DCPA/TCPA（适合在小候选集上频繁调用）"""
    pij = p_j - p_i
    vij = v_j - v_i
    den = float(np.dot(vij, vij))
    if den < 1e-12:
        return float(np.linalg.norm(pij)), 1e9
    tc = - float(np.dot(pij, vij)) / den
    if tc < 0.0:
        return float(np.linalg.norm(pij)), 0.0
    dmin = pij + tc * vij
    return float(np.linalg.norm(dmin)), tc

def _advance_once(pos, psi, vel, dpsi, v_cmd, dt, dpsi_max, v_min, v_max):
    """向前推进一步（就地更新 copy 的数组）"""
    N = pos.shape[0]
    dpsi = _clip(dpsi, -dpsi_max, dpsi_max)
    psi  = _wrap_pi(psi + dpsi)
    v    = _clip(v_cmd, v_min, v_max)
    vel[:, 0] = np.cos(psi) * v
    vel[:, 1] = np.sin(psi) * v
    pos[:, 0] += vel[:, 0] * dt
    pos[:, 1] += vel[:, 1] * dt
    return pos, psi, vel, v

def strict_guard_multi(
    ships,
    dpsi, v_cmd, dt,
    dpsi_max, v_min, v_max,
    collide_thr,
    # —— 复用推进“前”的 tc/dc，减少一次 O(N^2) —— 强烈推荐传入
    tc: np.ndarray | None = None,
    dc: np.ndarray | None = None,
    # —— 轻量多步预览 & 候选集控制
    preview_M: int = 3,
    inflate: float = 1.15,        # 安全距离略放大，抵消数值误差
    K_guard: int = 6,             # 每船最多 K 个候选对
    steer_gain: float = 1.0,      # =1 → 最大允许转角；<1 更柔和
    brake_to_min: bool = True,    # 触发时是否直接刹到 v_min
    use_float32: bool = True,     # 降带宽
):
    """
    通用 N 船“严格护栏”：多步预览，任何预览步若 DCPA < inflate*collide_thr，
    即对相关船舶强制：减速 + 远离彼此的最大允许转角。
    返回 (dpsi_safe[N], v_safe[N])，均 float64。
    """
    N = len(ships)
    if N <= 1:
        return np.asarray(dpsi, np.float64), np.asarray(v_cmd, np.float64)

    # ---- 数据打包 ----
    _dtype = np.float32 if use_float32 else np.float64
    dpsi = np.asarray(dpsi, _dtype).copy()
    v_cmd = np.asarray(v_cmd, _dtype).copy()
    pos0 = np.stack([s.pos for s in ships], 0).astype(np.float64)     # 预览用 double 更稳
    psi0 = np.array([s.psi for s in ships], dtype=np.float64)
    v0   = np.array([s.v   for s in ships], dtype=np.float64)
    vel0 = np.stack([np.cos(psi0)*v0, np.sin(psi0)*v0], 1).astype(np.float64)

    # —— 复用/计算 tc, dc（推进“前”的矩阵）——
    if tc is None or dc is None:
        tc, dc, _ = tcpa_dcpa_matrix(ships)   # 注意：这是唯一的 O(N^2)

    # ---- 候选邻居集（每船 K 个）----
    thr = float(inflate * collide_thr)
    dc_masked = dc.astype(np.float64).copy()
    np.fill_diagonal(dc_masked, np.inf)
    K = min(max(1, K_guard), max(1, N-1))
    # 优先：所有 dc < thr 的对；不足 K 个再补最近的
    knn_idx = np.argpartition(dc_masked, K, axis=1)[:, :K]  # [N,K]
    # 额外把强风险对加入候选（并集）
    risk_mask = (dc_masked < thr)
    cand_lists = []
    for i in range(N):
        cand = set(knn_idx[i].tolist())
        # 把强风险对补进来
        risk_js = np.nonzero(risk_mask[i])[0]
        for j in risk_js:
            if j != i:
                cand.add(int(j))
        cand_lists.append(np.fromiter(cand, dtype=np.int64))

    # ---- 预览缓冲（拷贝一份用于模拟）----
    pos = pos0.copy()
    psi = psi0.copy()
    vel = vel0.copy()
    dpsi_safe = dpsi.copy()
    v_safe    = v_cmd.copy()

    # 记录是否在任一预览步触发了强制避让
    any_forced = np.zeros(N, dtype=bool)

    # ---- 预览循环 ----
    for _ in range(int(max(1, preview_M))):
        # 以“当前修正后动作”推进一步
        pos, psi, vel, v = _advance_once(pos, psi, vel, dpsi_safe, v_safe,
                                          dt, dpsi_max, v_min, v_max)

        # 检查候选对的 DCPA，若 < thr 则对成对船舶都强制修正
        to_fix = [ [] for _ in range(N) ]  # 每船触发对方的列表
        for i in range(N):
            Js = cand_lists[i]
            if Js.size == 0:
                continue
            p_i = pos[i]; v_i = vel[i]
            worst_j, worst_d = -1, 1e9
            for j in Js:
                if j == i:
                    continue
                dmin, _ = _dcpa_tcpa(p_i, v_i, pos[j], vel[j])
                if dmin < thr and dmin < worst_d:
                    worst_d, worst_j = dmin, int(j)
            if worst_j >= 0:
                to_fix[i].append(worst_j)

        # 应用强制修正
        changed = False
        for i in range(N):
            if not to_fix[i]:
                continue
            # 取最近威胁的一个 j*
            j_star = to_fix[i][0]
            # 远离对方的方向
            rel = pos[j_star] - pos[i]
            bearing = _wrap_pi(np.arctan2(rel[1], rel[0]) - psi[i])
            # 最大允许转角的 steer_gain 比例
            dpsi_safe[i] = _clip(dpsi_safe[i] - np.sign(bearing) * (steer_gain * dpsi_max),
                                 -dpsi_max, dpsi_max)
            if brake_to_min:
                v_safe[i] = v_min
            any_forced[i] = True
            changed = True

        # 如果这一预览步没人需要修正，说明未来 M 步都安全，可提前结束
        if not changed:
            break

    # 统一夹持 & 输出 float64
    dpsi_out = _clip(dpsi_safe, -dpsi_max, dpsi_max).astype(np.float64)
    v_out    = _clip(v_safe,  v_min,     v_max   ).astype(np.float64)
    return dpsi_out, v_out, any_forced
