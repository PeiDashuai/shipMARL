# miniship/observe/neighbors.py
from __future__ import annotations

import math

from typing import Sequence, Optional, List
import numpy as np


def pick_neighbors(
    ships: List[object],
    risk_T_thr: float,
    risk_D_thr: float,
    K: int,
    Rn: Optional[float] = None,
    Vm: Optional[float] = None,
) -> np.ndarray:
    """
    选择每条船的 K 个邻居（用于 GNN），返回 shape=(N, K) 的邻接索引矩阵。

    - K 是“最大邻居容量”，固定由 env_cfg["numNeighbors"] 决定
    - 实际邻居数 < K 时，用本船索引 i 进行 padding，保证输出 shape 始终为 (N, K)
    - Rn 作为归一化半径/距离阈值，如果为 None，则不做半径截断
    - risk_T_thr, risk_D_thr, Vm 目前保留在接口中，便于以后扩展，但本函数核心只用 Rn
    """
    N = len(ships)
    assert K >= 0, "K must be non-negative"

    if N == 0:
        # 空场景：返回 (0, K)
        return np.zeros((0, K), dtype=np.int64)

    # 位置矩阵 (N, 2)
    pos = np.asarray([s.pos for s in ships], dtype=float)

    # pairwise squared distances d2[i, j]
    d2 = np.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=-1)

    # 自身不能作为“真实邻居”，先标成 inf，后面会用 i 自己去 padding
    np.fill_diagonal(d2, np.inf)

    # 半径阈值
    if Rn is None:
        Rn2 = np.inf
    else:
        Rn2 = float(Rn) * float(Rn)

    neighbors_idx = np.empty((N, K), dtype=np.int64)

    for i in range(N):
        # 根据距离排序，越近越前
        order = np.argsort(d2[i])  # 长度 N

        cnt = 0
        for j in order:
            if np.isinf(d2[i, j]):
                # 自己或不存在的点
                continue

            # 如果设置了半径阈值，超过就不再选（后面的只会更远）
            if d2[i, j] > Rn2:
                break

            neighbors_idx[i, cnt] = j
            cnt += 1
            if cnt >= K:
                break

        # padding：真实邻居不足 K，就用本船索引 i 填满
        while cnt < K:
            neighbors_idx[i, cnt] = i
            cnt += 1

    return neighbors_idx



def neighbor_features(
    si,
    sj,
    Rn: Optional[float] = None,
    Vm: Optional[float] = None,
    risk_T_thr: float = 130.0,
    risk_D_thr: float = 50.0,
    v_max: Optional[float] = None,
    **kwargs,
) -> np.ndarray:
    """
    8 维邻居几何/风险特征（F_nei_base = 8）:
      [r_n, dv_n, cos(dpsi_i), sin(dpsi_i),
       cos(dpsi_j), sin(dpsi_j), d_n, t_n]

    说明：
      - Rn   : 空间尺度，用于距离归一
      - Vm   : 速度尺度，用于相对速度归一；若 Vm=None 则回退到 v_max 或自身 max(v_i, v_j)
      - risk_*: 仅用于 d_n/t_n 归一
    """
    # 兼容老接口：有的地方只传 v_max
    if Vm is None and v_max is not None:
        Vm = float(v_max)

    # 1) 相对位移
    dx = float(sj.pos[0] - si.pos[0])
    dy = float(sj.pos[1] - si.pos[1])
    r = math.hypot(dx, dy)

    def _get_v_psi(s):
        # 优先用 v/psi
        if hasattr(s, "v") and hasattr(s, "psi"):
            vv = float(getattr(s, "v", 0.0))
            pp = float(getattr(s, "psi", 0.0))
            return vv, pp, vv * math.cos(pp), vv * math.sin(pp)
        # 否则尝试 vx/vy（兼容 PF 输出/其他状态对象）
        vx = float(getattr(s, "vx", getattr(s, "vx_mps", 0.0)))
        vy = float(getattr(s, "vy", getattr(s, "vy_mps", 0.0)))
        vv = math.hypot(vx, vy)
        pp = math.atan2(vy, vx) if vv > 1e-9 else float(getattr(s, "psi", 0.0))
        return vv, pp, vx, vy

    v_i, psi_i, vx_i, vy_i = _get_v_psi(si)
    v_j, psi_j, vx_j, vy_j = _get_v_psi(sj)

    dvx = vx_j - vx_i
    dvy = vy_j - vy_i
    dv = math.hypot(dvx, dvy)

    # 3) 归一化尺度
    Vm = Vm or max(v_i, v_j, 1.0)
    Rn = Rn or 1.0

    r_n = r / max(Rn, 1e-6)
    dv_n = dv / max(Vm, 1e-6)

    # 4) 方位相关航向
    psi_rel = math.atan2(dy, dx)
    dpsi_i = wrap_to_pi(psi_i - psi_rel)
    dpsi_j = wrap_to_pi(psi_j - psi_rel)

    # 5) 简单 D/T 风险归一，用于 d_n / t_n
    D_thr = risk_D_thr
    T_thr = risk_T_thr

    closing_speed = -(dx * dvx + dy * dvy) / max(r, 1e-6)
    if closing_speed > 1e-3:
        tcpa = r / closing_speed
    else:
        tcpa = float("inf")

    d_n = r / max(D_thr, 1e-6)
    t_n = tcpa / max(T_thr, 1e-6) if math.isfinite(tcpa) else 1.0

    # 裁剪防爆
    d_n = min(d_n, 3.0)
    t_n = min(t_n, 3.0)

    return np.array(
        [
            r_n,
            dv_n,
            math.cos(dpsi_i),
            math.sin(dpsi_i),
            math.cos(dpsi_j),
            math.sin(dpsi_j),
            d_n,
            t_n,
        ],
        dtype=np.float32,
    )


def edge_features(
    si,
    sj,
    Rn: float,
    Vm: float,
    risk_T_thr: float,
    risk_D_thr: float,
) -> np.ndarray:
    """
    8 维边特征（F_edge=8），目前直接复用 neighbor_features。
    """
    base = neighbor_features(
        si,
        sj,
        Rn=Rn,
        Vm=Vm,
        risk_T_thr=risk_T_thr,
        risk_D_thr=risk_D_thr,
    )
    return base.astype(np.float32)


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi
