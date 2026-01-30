# miniship/observe/builder.py

from __future__ import annotations

import math
from typing import Dict, List, Any, Optional, Tuple
import os
import numpy as np

from .neighbors import pick_neighbors, neighbor_features, edge_features

class _ShipView:
    """Adapter to unify core Ship and PF/TrueState-like objects for obs features."""
    __slots__ = (
        "pos", "v", "psi",
        "goal",
        "ship_id", "sid", "id",
        "risk_T_thr", "risk_D_thr",
        "ais_u_stale", "ais_u_silence", "ais_valid",
    )


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _get_ship_id(ship: Any, fallback: int) -> int:
    for key in ("ship_id", "sid", "id"):
        if hasattr(ship, key):
            try:
                return int(getattr(ship, key))
            except Exception:
                pass
    # accept string ids like "ship_1"
    try:
        s = str(getattr(ship, "agent_id", "")).strip()
        if s.startswith("ship_"):
            return int(s.split("_")[-1])
    except Exception:
        pass
    return int(fallback)


def _get_xy(ship: Any) -> Tuple[float, float]:
    # Prefer pos[0],pos[1] if present
    if hasattr(ship, "pos"):
        try:
            p = getattr(ship, "pos")
            return _to_float(p[0]), _to_float(p[1])
        except Exception:
            pass
    # Fallback to x,y
    return _to_float(getattr(ship, "x", 0.0)), _to_float(getattr(ship, "y", 0.0))


def _get_v_psi(ship: Any) -> Tuple[float, float]:
    # Prefer v,psi if present
    if hasattr(ship, "v") and hasattr(ship, "psi"):
        v = _to_float(getattr(ship, "v", 0.0))
        psi = _to_float(getattr(ship, "psi", 0.0))
        return v, psi

    # Fallback to vx,vy -> (v,psi)
    vx = getattr(ship, "vx", None)
    vy = getattr(ship, "vy", None)
    if vx is not None and vy is not None:
        vx = _to_float(vx, 0.0)
        vy = _to_float(vy, 0.0)
        v = float(math.hypot(vx, vy))
        psi = float(math.atan2(vy, vx)) if v > 1e-9 else 0.0
        # wrap psi to (-pi, pi]
        psi = (psi +math.pi) % (2.0 * math.pi) - math.pi
        return v, psi

    return 0.0, 0.0

def _get_goal_xy(ship: Any, default_xy: Tuple[float, float]) -> Tuple[float, float]:
    # ship.goal is expected to be array-like (2,)
    if hasattr(ship, "goal"):
        try:
            g = getattr(ship, "goal")
            return _to_float(g[0], default_xy[0]), _to_float(g[1], default_xy[1])
        except Exception:
            pass
    return default_xy[0], default_xy[1]

def _as_view(ship: Any) -> _ShipView:
    sv = _ShipView()
    x, y = _get_xy(ship)
    v, psi = _get_v_psi(ship)
    sv.pos = np.array([x, y], dtype=np.float32)
    sv.v = float(v)
    sv.psi = float(psi)

    # IMPORTANT: preserve goal from the original ship object.
    # Without this, ego goal features will degenerate to zeros (gx,gy fallback to x,y).
    gx, gy = _get_goal_xy(ship, default_xy=(x, y))
    sv.goal = np.array([gx, gy], dtype=np.float32)

    sid = _get_ship_id(ship, fallback=1)
    sv.ship_id = sid
    sv.sid = sid
    sv.id = sid

    sv.risk_T_thr = _to_float(getattr(ship, "risk_T_thr", 130.0), 130.0)
    sv.risk_D_thr = _to_float(getattr(ship, "risk_D_thr", 50.0), 50.0)

    sv.ais_u_stale = float(np.clip(_to_float(getattr(ship, "ais_u_stale", 0.0), 0.0), 0.0, 1.0))
    sv.ais_u_silence = float(np.clip(_to_float(getattr(ship, "ais_u_silence", 0.0), 0.0), 0.0, 1.0))
    sv.ais_valid = bool(getattr(ship, "ais_valid", True))
    return sv

 

def build_observations(
    ships: List,
    K: int,
    spawn_mode: str,
    spawn_area: float,
    spawn_len: float,
    v_max: float,
    obs_debug: bool = False,
    debug_prefix: str = "",
    ego_only: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    统一的观测 builder：
      - 对 core_env（真值）和 PF AIS env（估计 +AIS 不确定性）都生效
      - 输出结构：6 +K * 11 +K * 8 +1 维
    """
    ships = list(ships)
    N = len(ships)
    if N == 0:
        return {}

    K = int(K)

    # Normalize ship representations (core Ship or TrueState-like) to a unified view.
    ships_v: List[_ShipView] = [_as_view(s) for s in ships]

    # 1) 挑选邻居：纯几何最近邻
    # 重要修订：
    #   - 对 PF/AIS 模式：必须尊重 nei.ais_valid（不可见目标不应占据邻居槽位）
    #   - 对真值模式：ais_valid 默认为 True，不改变行为
    pos = np.asarray([getattr(s, "pos") for s in ships], dtype=np.float64)  # (N,2)
    valid_mask = np.asarray([bool(getattr(s, "ais_valid", True)) for s in ships], dtype=bool)

    neighbors_idx = -np.ones((N, K), dtype=np.int64)
    for i in range(N):
        # 候选：排除自船 +仅保留 ais_valid=True
        cand = []
        xi, yi = float(pos[i, 0]), float(pos[i, 1])
        for j in range(N):
            if j == i:
                continue
            if not valid_mask[j]:
                continue
            dx = float(pos[j, 0]) - xi
            dy = float(pos[j, 1]) - yi
            d = math.hypot(dx, dy)
            if d > float(spawn_area):
                continue
            cand.append((d, j))
        cand.sort(key=lambda x: x[0])
        for kk, (_, j) in enumerate(cand[:K]):
            neighbors_idx[i, kk] = int(j)

    if not isinstance(neighbors_idx, np.ndarray):
        neighbors_idx = np.asarray(neighbors_idx, dtype=np.int64)

    assert neighbors_idx.shape == (N, K), (
        f"[ObsBuilder] neighbors_idx shape mismatch: {neighbors_idx.shape}, "
        f"N={N}, K={K}"
    )

    # ------------------------------------------------------------
    # Repair neighbor indices:
    #   - never include self
    #   - avoid duplicates
    #   - optionally exclude ais_valid==False candidates (PF stale tracks)
    # This prevents subtle "self as neighbor" or "duplicate neighbor" bugs
    # and reduces information leakage from invalid PF tracks.
    # ------------------------------------------------------------
    pos = np.stack([sv.pos for sv in ships_v], axis=0).astype(np.float32)  # (N,2)
    valid_mask = np.array([bool(getattr(sv, "ais_valid", True)) for sv in ships_v], dtype=bool)

    for i in range(N):
        chosen = set()
        # first pass: sanitize existing indices
        for k in range(K):
            j = int(neighbors_idx[i, k])
            if j < 0 or j >= N or j == i:
                neighbors_idx[i, k] = -1
                continue
            # exclude invalid PF tracks from neighbor list
            if not bool(valid_mask[j]):
                neighbors_idx[i, k] = -1
                continue
            if j in chosen:
                neighbors_idx[i, k] = -1
                continue
            chosen.add(j)

        # second pass: fill empty slots with nearest remaining valid candidates
        empties = [k for k in range(K) if int(neighbors_idx[i, k]) < 0]
        if len(empties) > 0:
            cand = [j for j in range(N) if (j != i and valid_mask[j] and (j not in chosen))]
            if len(cand) > 0:
                d = np.linalg.norm(pos[np.asarray(cand)] - pos[i], axis=1)
                order = np.argsort(d)
                for kk, slot in enumerate(empties):
                    if kk >= len(order):
                        break
                    j2 = int(cand[int(order[kk])])
                    neighbors_idx[i, slot] = j2
                    chosen.add(j2)

    # 2) 维度定义：确保和 GNN/LSTM 一致
    F_self = 8
    F_nei = 11   # 8 几何 +3 AIS scalar
    F_edge = 8
    obs_dim = F_self +K * F_nei +K * F_edge +1

    obs_dict: Dict[str, np.ndarray] = {}

    # --------------------------
    # OBS layout debug (optional)
    # --------------------------
    # Enable via:
    #   - env_config['obs_debug']=True (propagates into obs_debug), OR
    #   - env var OBSDBG=1
    # Optional filters:
    #   OBSDBG_SID=<int>   (only print for this ego ship id; empty => all)
    #   OBSDBG_KDIM=<int>  (print first KDIM dims with field names; default 20)
    #   OBSDBG_MAX=<int>   (max prints per process; default 5)
    dbg_enable = bool(obs_debug) or (os.environ.get("OBSDBG", "0") == "1")
    _dbg_sid_s = os.environ.get("OBSDBG_SID", "").strip()
    dbg_sid = int(_dbg_sid_s) if _dbg_sid_s else None
    dbg_kdim = int(os.environ.get("OBSDBG_KDIM", "20"))
    dbg_max = int(os.environ.get("OBSDBG_MAX", "5"))

    # function-level cache/counters (persist across calls within a worker process)
    if not hasattr(build_observations, "_dbg_prints"):
        build_observations._dbg_prints = 0
        build_observations._dbg_name_cache = {}

    def _field_names_for_K(_K: int) -> List[str]:
        cache = getattr(build_observations, "_dbg_name_cache", {})
        if _K in cache:
            return cache[_K]
        names: List[str] = []
        # ego (F_self=8): [x, y, vx, vy, psi, v_norm]
        names += [
            "ego.x_m",
            "ego.y_m",
            "ego.vx_mps",
            "ego.vy_mps",
            "ego.psi_rad",
            "ego.v_norm",
            "ego.goal_fwd_norm",   # NEW
            "ego.goal_lat_norm",   # NEW
        ]

        # tracks: neighbors (F_nei=11) = n8(8) +u_stale +u_silence +valid(mask)
        for i_nei in range(_K):
            names += [f"nei{i_nei}.n8[{j}]" for j in range(8)]
            names += [f"nei{i_nei}.u_stale", f"nei{i_nei}.u_silence", f"nei{i_nei}.valid"]

        # tracks: edges (F_edge=8) = e8(8)
        for i_nei in range(_K):
            names += [f"edge{i_nei}.e8[{j}]" for j in range(8)]

        # extra scalar (placeholder)
        names += ["extra[0]"]

        cache[_K] = names
        build_observations._dbg_name_cache = cache
        return names

    def _fmt_idx_list(idxs: List[int], max_show: int = 12) -> str:
        if len(idxs) <= max_show:
            return str(idxs)
        return f"{idxs[:max_show]} ... {idxs[-3:]}"


    # 3) 对每条船构造观测
    for ei, ego in enumerate(ships_v):
        # Use _get_ship_id which handles 'sid', 'ship_id', and 'id' attributes
        ego_id = _get_ship_id(ego, ei + 1)
        ego_id_str = str(int(ego_id))


        if ego_only is not None and int(ego_id) != int(ego_only):
            continue

        # 3.2 自船 8 维特征
        x_self = _extract_self_state(ego, v_max=v_max, spawn_len=spawn_len, spawn_area=spawn_area)  # (8,)

        nei_slots: List[np.ndarray] = []
        edge_slots: List[np.ndarray] = []

        # 3.3 K 个邻居 slot
        for k in range(K):
            j = int(neighbors_idx[ei, k])

            if j < 0 or j >= N or j == ei:
                # 空 slot，全部 0
                nei_feat = np.zeros((11,), dtype=np.float32)
                edge_feat = np.zeros((8,), dtype=np.float32)
            else:
                nei = ships_v[j]

                # (2) AIS 不确定性（PF 模式下由 AIS env 写入；真值模式下默认）
                u_stale = float(np.clip(getattr(nei, "ais_u_stale", 0.0), 0.0, 1.0))
                u_silence = float(np.clip(getattr(nei, "ais_u_silence", 0.0), 0.0, 1.0))
                ais_valid = getattr(nei, "ais_valid", True)
                valid = 1.0 if ais_valid else 0.0

                if not ais_valid:
                    # IMPORTANT:
                    # If PF track is invalid (too stale/silent), do NOT leak geometry/risk features.
                    # Keep only the uncertainty scalars so the policy can learn to ignore.
                    n8 = np.zeros((8,), dtype=np.float32)
                    nei_feat = np.concatenate(
                        [n8, np.array([u_stale, u_silence, valid], dtype=np.float32)],
                        axis=0,
                    )
                    edge_feat = np.zeros((8,), dtype=np.float32)
                else:
                    # (1) 8 维几何/相对运动特征
                    n8 = neighbor_features(
                        ego,
                        nei,
                        Rn=spawn_area,
                        Vm=v_max,
                    )

                    # 拼成 11 维邻居特征
                    nei_feat = np.concatenate(
                        [
                            n8.astype(np.float32),
                            np.array([u_stale, u_silence, valid], dtype=np.float32),
                        ],
                        axis=0,
                    )

                    # (3) 8 维 edge 特征：带风险归一
                    edge_feat = edge_features(
                        ego,
                        nei,
                        Rn=spawn_area,
                        Vm=v_max,
                        risk_T_thr=getattr(ego, "risk_T_thr", 130.0),
                        risk_D_thr=getattr(ego, "risk_D_thr", 50.0),
                    ).astype(np.float32)

            nei_slots.append(nei_feat)
            edge_slots.append(edge_feat)

        nei_vec = np.concatenate(nei_slots, axis=0)   # (K * 11,)
        edge_vec = np.concatenate(edge_slots, axis=0) # (K * 8,)

        # 3.4 额外 scalar（占位）
        extra_scalar = np.array([0.0], dtype=np.float32)

        full_vec = np.concatenate(
            [x_self, nei_vec, edge_vec, extra_scalar], axis=0
        )

        if full_vec.shape[0] != obs_dim:
            raise RuntimeError(
                f"[ObsBuilder] obs_dim mismatch for agent={ego_id_str}: "
                f"got {full_vec.shape[0]}, expected {obs_dim}"
            )

        obs_dict[ego_id_str] = full_vec

        # Detailed layout debug: print first KDIM dims +field name mapping +segment boundaries.
        if dbg_enable:
            try:
                ego_sid_dbg = int(ego_id_str)
            except Exception:
                ego_sid_dbg = None

            if (dbg_sid is None) or (ego_sid_dbg == dbg_sid):
                if int(getattr(build_observations, "_dbg_prints", 0)) < dbg_max:
                    build_observations._dbg_prints += 1

                    ego_end = F_self
                    nei_end = F_self +K * F_nei
                    edge_end = nei_end +K * F_edge
                    mask_idxs = [F_self +i_nei * F_nei +(F_nei - 1) for i_nei in range(K)]

                    print(
                        f"[ObsBuilder]{debug_prefix} ego_sid={ego_sid_dbg} N={N} K={K} obs_dim={obs_dim} | "
                        f"seg: ego[0:{ego_end}) nei[{ego_end}:{nei_end}) edge[{nei_end}:{edge_end}) extra[{edge_end}:{obs_dim}) | "
                        f"mask(valid) idx={_fmt_idx_list(mask_idxs)}"
                    )
                    print("  units: ego=(norm,norm,norm,norm,norm,unitless) u_*=(0..1) valid∈{0,1}")
                    names = _field_names_for_K(K)
                    k_show = max(0, min(dbg_kdim, obs_dim))
                    for j in range(k_show):
                        nm = names[j] if j < len(names) else f"dim{j}"
                        print(f"  [{j:03d}] {nm:>18s} = {float(full_vec[j]):+.6f}")
 
    # Optional per-ego neighbor debug (compact) - print once per build call (not N times)
    if os.environ.get("OBS_NEI_DBG", "0") == "1":
        for ei, ego in enumerate(ships_v):
            ego_id = _get_ship_id(ego, ei + 1)
            if ego_only is not None and int(ego_id) != int(ego_only):
                continue
            js = [int(neighbors_idx[ei, k]) for k in range(K)]
            ids = []
            metas = []
            for j in js:
                if j < 0:
                    ids.append(-1)
                    metas.append("(empty)")
                else:
                    nv = ships_v[j]
                    ids.append(_get_ship_id(nv, j + 1))
                    metas.append(
                        f"(valid={int(bool(getattr(nv,'ais_valid',True)))}"
                        f",u_stale={float(getattr(nv,'ais_u_stale',0.0)):.2f}"
                        f",u_sil={float(getattr(nv,'ais_u_silence',0.0)):.2f})"
                    )
            print(f"[ObsBuilder]{debug_prefix} ego={ego_id} nei={ids} meta={metas}")


    # ---- finite check (obs_dict is dict[str, np.ndarray]) ----
    if len(obs_dict) > 0:
        agent_ids = list(obs_dict.keys())
        mat = np.stack([np.asarray(obs_dict[aid], dtype=np.float32) for aid in agent_ids], axis=0)  # (n_agent, obs_dim)

        if not np.all(np.isfinite(mat)):
            bad = np.argwhere(~np.isfinite(mat))  # rows: agent index, cols: dim index
            # 取前 20 个坏点
            head = bad[:20].tolist()
            # 可选：打印对应 agent 和 dim
            head_named = [(agent_ids[i], int(j), float(mat[i, j])) for i, j in bad[:20]]
            raise RuntimeError(
                f"[OBS NONFINITE] n_bad={bad.shape[0]} head_idx={head} head_named={head_named}"
            )

    return obs_dict


def _extract_self_state(ship, v_max: float, spawn_len: float, spawn_area: float) -> np.ndarray:
    """自船 8 维特征 (normalized):
    [x_norm, y_norm, vx_norm, vy_norm, psi_norm, v_norm, g_fwd_norm, g_lat_norm]
    """
    x = float(ship.pos[0]) if hasattr(ship, "pos") else _to_float(getattr(ship, "x", 0.0))
    y = float(ship.pos[1]) if hasattr(ship, "pos") else _to_float(getattr(ship, "y", 0.0))

    v = _to_float(getattr(ship, "v", 0.0), 0.0)
    psi = _to_float(getattr(ship, "psi", 0.0), 0.0)

    # wrap psi to (-pi, pi]
    psi = (psi + math.pi) % (2.0 * math.pi) - math.pi

    vx = v * math.cos(psi)
    vy = v * math.sin(psi)

    v_ref = max(float(v_max), 1e-6)
    pos_ref = max(float(spawn_area), float(spawn_len), 1e-6)

    # --- normalize ego absolute states ---
    x_norm = float(np.clip(x / pos_ref, -1.0, 1.0))
    y_norm = float(np.clip(y / pos_ref, -1.0, 1.0))

    vx_norm = float(np.clip(vx / v_ref, -1.0, 1.0))
    vy_norm = float(np.clip(vy / v_ref, -1.0, 1.0))

    psi_norm = float(psi / math.pi)  # already in [-1, 1]

    v_norm = float(np.clip(v / v_ref, 0.0, 1.0))

    # goal relative (world)
    if hasattr(ship, "goal"):
        try:
            gx, gy = float(ship.goal[0]), float(ship.goal[1])
        except Exception:
            gx, gy = x, y
    else:
        gx, gy = x, y

    dx = gx - x
    dy = gy - y

    # world -> body: [fwd; lat] = R(-psi)[dx;dy]
    c = math.cos(psi)
    s = math.sin(psi)
    d_fwd = c * dx + s * dy
    d_lat = -s * dx + c * dy

    # normalize goal projection by spawn_len (already good)
    d_ref = max(float(spawn_len), 1e-6)
    g_fwd_norm = float(np.clip(d_fwd / d_ref, -1.0, 1.0))
    g_lat_norm = float(np.clip(d_lat / d_ref, -1.0, 1.0))

    return np.array(
        [x_norm, y_norm, vx_norm, vy_norm, psi_norm, v_norm, g_fwd_norm, g_lat_norm],
        dtype=np.float32
    )



def zero_observation(obs_space):
    """用于 rllib wrapper 的零观测"""
    return np.zeros(obs_space.shape, dtype=np.float32)
