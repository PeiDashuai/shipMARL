# miniship/controllers/rule_baseline.py
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List
from miniship.risk.tcpa_dcpa import tcpa_dcpa_matrix
from miniship.utils.math import wrap_to_pi

@dataclass
class RuleParams:
    # -------- Gate 阈值（来自你的设计图）--------
    R_turn: float = 130.0       # 半径门
    ahead_cos: float = 0.2      # 朝向目标门：cosθ > ahead_cos（θ越小越“正前”）
    T_gate: float = 90.0        # TCPA 窗
    D_gate: float = 32.0        # DCPA 窗
    R_turn: float = 180.0
    T_gate: float = 110.0
    D_gate: float = 40.0

    # -------- 角色判定（简化 COLREGs）--------
    th_headon: float = math.radians(18.0)   # 对遇方位阈
    th_heading_diff: float = math.radians(150.0)  # 对遇航向差 >150°
    th_cross: float = math.radians(112.5)   # ← 新增：前半空间上界，用于|β|归一化

    # -------- 趋目标（基础路径跟踪器）--------
    heading_scale: float = math.radians(22.5)  # tanh归一化尺度（小=更激进）
    lp_alpha: float = 0.6                      # 航向指令一阶低通
    v_cruise_k: float = 0.90                   # v_max 比例
    v_min_k: float = 0.25                      # 基础最小行驶
    align_pow: float = 2.0                     # 速度随对齐度幂次
    slow_radius_mul: float = 2.0               # 进入慢行圈半径倍数*goal_tol
    stop_k: float = 0.05                       # 接近停船速度

    # -------- 避碰基础参数（给路/直航）--------
    turn_right_frac: float = 0.95              # 右转占 dpsi_max 的最大幅度
    standon_eps: float = 0.12                  # 直航船微右偏
    v_slow_frac: float = 0.55                  # 避碰时速度

    # -------- 混合与回正 --------
    # 风险权重从 Gate 计算：sev∈[0,1]，再平滑
    sev_smooth: float = 0.6                    # sev EMA
    decay_when_safe: float = 0.85              # 解除风险后避碰权重指数衰减（每步）
    min_mix_when_unknown: float = 0.3          # 角色不明时的混合上限（避免长期右偏）

class RuleBaselineController:
    """
    Baseline: “基础路径跟踪器 + 风险Gate触发的避碰控制器 + 连续混合 + 回正衰减”

    Gate（对 i 评 j）四条件：
      1) d = ||pj-pi|| < R_turn
      2) cosθ = e_i · (pj-pi)/||pj-pi|| > ahead_cos   （e_i 为我船朝向单位向量）
      3) closing： (pj-pi)·(vj-vi) < 0
      4) 0 <= TCPA <= T_gate 且 DCPA <= D_gate
    通过 Gate 的邻居集合里取最大 severiy 作为“本步对手”。

    severity（归一化到 0~1）：
       sev = clip( (R_turn-d)/R_turn, 0,1 ) *
             clip( 1 - TCPA/T_gate, 0,1 ) *
             clip( 1 - DCPA/D_gate, 0,1 )
    用 EMA 平滑，再做混合权重。

    角色（简化 COLREGs）：
      - 对遇（|θ|<th_headon 且 航向差>th_heading_diff）：双方右转
      - 交叉：对方在我右舷 -> 我右转（give-way）；对方在我左舷 -> 我 stand-on（微右偏）
      - 超越未实现：按普通交叉处理

    混合：
      dpsi = (1-w)*dpsi_goal + w*dpsi_avoid
      v    = (1-w)*v_goal    + w*v_avoid
    其中 w 来自 sev 平滑；角色不明时 w<=min_mix_when_unknown。
    解除 Gate 后 w 以 decay_when_safe 指数衰减，保证快速回正。
    """
    def __init__(self, env, params: RuleParams = RuleParams()):
        self.env = env
        self.P = params
        self._prev_dpsi_norm: Dict[int, float] = {}
        self._sev_ema: Dict[int, float] = {}  # sid -> 平滑后的 sev

    # ---- 公共API ----
    def reset(self):
        self._prev_dpsi_norm.clear()
        self._sev_ema.clear()

    def act(self) -> Dict[str, np.ndarray]:
        ships = self.env.state.ships
        N = len(ships)
        tc, dc, _ = tcpa_dcpa_matrix(ships)  # 推进前风险几何

        actions: Dict[str, np.ndarray] = {}

        for i, aid in enumerate(self.env.agents):
            si = ships[i]
            if si.reached:
                actions[aid] = np.array([0.0, -1.0], np.float32)
                continue

            # 1) 基础路径跟踪器：航向 + 速度
            dpsi_goal, err_abs = self._goal_heading_cmd(si)
            dist_to_goal = float(np.linalg.norm(si.pos - si.goal))
            v_goal = self._goal_speed_cmd(err_abs, dist_to_goal)

            # 2) Gate & severity & 最危险邻居
            sev, role, dpsi_avoid, v_avoid, has_gate = self._avoid_cmd_for_ship(i, ships, tc, dc)

            # 3) sev 平滑 + 安全衰减
            sev_s = self._sev_ema.get(si.sid, 0.0)
            sev_s = self.P.sev_smooth * sev_s + (1.0 - self.P.sev_smooth) * sev
            if not has_gate:
                sev_s *= self.P.decay_when_safe  # 快速回正
            self._sev_ema[si.sid] = sev_s

            # 4) 混合（角色不明时限制；右舷/对遇时给保底）
            w = sev_s
            if role == "unknown":
                w = min(w, self.P.min_mix_when_unknown)
            elif role in ("headon", "star"):
                w = max(w, 0.5)  # 保底：对遇/右舷交叉至少 0.5 的避碰权重

            dpsi = (1.0 - w) * dpsi_goal + w * dpsi_avoid
            v    = (1.0 - w) * v_goal    + w * v_avoid

            actions[aid] = np.array([float(np.clip(dpsi, -1.0, 1.0)),
                                     float(np.clip(v,    -1.0, 1.0))], dtype=np.float32)
        return actions

    # ---- 细节函数 ----
    def _goal_heading_cmd(self, si) -> Tuple[float, float]:
        goal_dir = math.atan2(si.goal[1]-si.pos[1], si.goal[0]-si.pos[0])
        e = wrap_to_pi(goal_dir - si.psi)
        raw = math.tanh(e / max(1e-6, self.P.heading_scale))
        # 低通
        prev = self._prev_dpsi_norm.get(si.sid, 0.0)
        u = (1.0 - self.P.lp_alpha) * prev + self.P.lp_alpha * raw
        u = float(np.clip(u, -1.0, 1.0))
        self._prev_dpsi_norm[si.sid] = u
        return u, abs(e)

    def _goal_speed_cmd(self, align_err_abs: float, dist_to_goal: float) -> float:
        align = max(0.0, 1.0 - align_err_abs / math.pi) ** self.P.align_pow
        v_frac = self.P.v_min_k + (self.P.v_cruise_k - self.P.v_min_k) * align
        if dist_to_goal < self.P.slow_radius_mul * self.env.goal_tol:
            w = max(0.0, (dist_to_goal - self.env.goal_tol) / max(1e-6, self.env.goal_tol))
            v_frac = max(self.P.stop_k, v_frac * w)
        return 2.0 * float(np.clip(v_frac, 0.0, 1.0)) - 1.0

    def _avoid_cmd_for_ship(self, i: int, ships, tc, dc) -> Tuple[float, str, float, float, bool]:
        """返回 (sev∈[0,1], role∈{'headon','star','port','unknown'}, dpsi_avoid_norm, v_avoid_norm, has_gate)"""
        si = ships[i]
        ei = np.array([math.cos(si.psi), math.sin(si.psi)], float)
        vi = ei * si.v

        best_sev, best_role, best_dpsi, best_v, has_gate = 0.0, "unknown", 0.0, (2.0*self.P.v_slow_frac - 1.0), False

        for j, sj in enumerate(ships):
            if j == i or sj.reached: 
                continue

            # 相对量
            rel = sj.pos - si.pos
            d = float(np.linalg.norm(rel))
            if d <= 1e-9: 
                continue
            r_hat = rel / d
            vj = np.array([math.cos(sj.psi), math.sin(sj.psi)], float) * sj.v
            vij = vj - vi

            # ---- Gate 四条件 ----
            cond_radius = (d < self.P.R_turn)
            cond_ahead  = (np.dot(ei, r_hat) > self.P.ahead_cos)
            cond_closing= (np.dot(rel, vij) < 0.0)

            tij = float(tc[i, j]); dij = float(dc[i, j])
            cond_tcpa = (tij >= 0.0) and (tij <= self.P.T_gate)
            cond_dcpa = (dij <= self.P.D_gate)

            if not (cond_radius and cond_ahead and cond_closing and cond_tcpa and cond_dcpa):
                continue

            has_gate = True
            # ---- Severity ----
            s_d = np.clip((self.P.R_turn - d) / self.P.R_turn, 0.0, 1.0)
            s_t = np.clip(1.0 - (tij / self.P.T_gate), 0.0, 1.0)
            s_c = np.clip(1.0 - (dij / self.P.D_gate), 0.0, 1.0)
            s_tc = max(s_t, s_c)
            sev  = 1.0 - (1.0 - s_d) * (1.0 - s_tc)   # 并集：有一个高就高

            # ---- 角色（简化COLREGs）----
            role = self._role_simple(si, sj)

            # ---- 避碰指令（连续强度，随 sev 与方位 β 渐变）----
            beta = wrap_to_pi(math.atan2(rel[1], rel[0]) - si.psi)  # j 在我右(+)/左(-)
            if role in ("headon", "star"):
                # 强右转，强度随 sev 与 |beta| 增大
                turn = self.P.turn_right_frac * np.clip((abs(beta) / self.P.th_cross), 0.2, 1.0)
                dpsi_avoid = - max(0.5 * self.P.standon_eps, turn * sev)  # 加一个最小右转
                v_avoid = 2.0 * self.P.v_slow_frac - 1.0
            elif role == "port":
                # 直航船：小幅右偏，不强制减速
                dpsi_avoid = - self.P.standon_eps * np.clip(sev, 0.25, 1.0)
                v_avoid    = None  # 用基础速度
            else:  # unknown
                dpsi_avoid = - 0.5 * self.P.standon_eps * sev
                v_avoid    = 2.0 * min(self.P.v_slow_frac, 0.7*self.P.v_cruise_k) - 1.0

            # 取最危险者
            if sev > best_sev:
                best_sev = sev
                best_role = role
                best_dpsi = dpsi_avoid
                best_v = (2.0*self.P.v_slow_frac - 1.0) if v_avoid is None else v_avoid

            # —— 紧急带（hard override）——
            coll_thr = self.env.collide_thr
            emerg = False
            if has_gate:
                # 距离很近 或 DCPA 很小 或 TCPA 很短 -> 直接硬右转 + 大减速
                if (d < 2.0 * coll_thr) or (dij < 1.5 * coll_thr) or (0.0 <= tij < 12.0):
                    emerg = True
                    best_sev  = 1.0
                    best_role = "star" if best_role == "unknown" else best_role
                    best_dpsi = - self.P.turn_right_frac
                    best_v    = 2.0 * self.P.v_slow_frac - 1.0            

        return best_sev, best_role, float(np.clip(best_dpsi, -1.0, 1.0)), float(np.clip(best_v, -1.0, 1.0)), has_gate

    def _role_simple(self, si, sj) -> str:
        """返回 'headon' / 'star' / 'port' / 'unknown' """
        # 方位
        beta = wrap_to_pi(math.atan2((sj.pos-si.pos)[1], (sj.pos-si.pos)[0]) - si.psi)
        # 航向相对
        hdg_diff = abs(wrap_to_pi(sj.psi - si.psi))
        if abs(beta) < self.P.th_headon and hdg_diff > self.P.th_heading_diff:
            return "headon"
        in_half = (abs(beta) <= math.radians(112.5))
        if in_half and beta > 0:
            return "star"   # 对方在右舷 -> 我 give-way
        if in_half and beta < 0:
            return "port"   # 对方在左舷 -> 我 stand-on
        return "unknown"
