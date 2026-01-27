# -*- coding: utf-8 -*-
"""
rule_baseline_ais_demo.py

一个自包含的小型 2 船对遇避碰场景：
  - TinyShipEnv: 简化避碰环境
  - AISCommsSim: 来自 ais_comms 的 AIS 通信仿真
  - AISTrackManager: 从 AIS 报文重建 noisy + delayed 轨迹
  - RuleBaselineController: 原始规则基线
  - RuleBaselineAISController: 使用 AIS 观测做避碰决策

运行:
    (shipRL) $ python rule_baseline_ais_demo.py
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np

# ====== 依赖你现有工程中的模块 ======
from ais_comms.ais_comms import AISCommsSim
from ais_comms.datatypes import TrueState, ShipId, AgentId
from miniship.risk.tcpa_dcpa import tcpa_dcpa_matrix
from miniship.utils.math import wrap_to_pi


# ======================================================================
# 一、小型 2 船对遇环境 TinyShipEnv
# ======================================================================

@dataclass
class TinyShip:
    sid: int
    pos: np.ndarray   # shape (2,)
    goal: np.ndarray  # shape (2,)
    psi: float        # heading [rad]
    v: float          # speed [m/s]
    reached: bool = False

class TinyShipState:
    def __init__(self, ships: List[TinyShip]):
        self.ships = ships

class TinyShipEnv:
    """
    极简双船对遇环境：
      - 两条船从左右对向驶来，目标是对面点
      - 简单积分：x_{t+1} = x + v*dt*[cos(psi), sin(psi)]
      - 控制输入: [dpsi_norm, v_norm] ∈ [-1,1]^2
    """
    def __init__(self,
                 dt: float = 1.0,
                 T_max: float = 200.0,
                 v_min: float = 2.0,
                 v_max: float = 6.0,
                 dpsi_max: float = math.radians(15.0),
                 goal_dist: float = 500.0,
                 collide_thr: float = 20.0):
        self.dt = float(dt)
        self.T_max = float(T_max)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.dpsi_max = float(dpsi_max)
        self.goal_dist = float(goal_dist)
        self.collide_thr = float(collide_thr)
        self.goal_tol = 15.0  # 到达容差

        # 两个 agent: 'A' / 'B'
        self.agents: List[str] = ["A", "B"]
        self.state: TinyShipState = None  # type: ignore
        self.t: float = 0.0

    def reset(self, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        # 船 A: 左 -> 右
        pos_A = np.array([-self.goal_dist, 0.0], float)
        goal_A = np.array([ self.goal_dist, 0.0], float)
        psi_A = 0.0 + rng.normal(0.0, math.radians(2.0))   # 微小扰动
        v_A = 4.0 + rng.normal(0.0, 0.3)

        # 船 B: 右 -> 左
        pos_B = np.array([ self.goal_dist, 0.0], float)
        goal_B = np.array([-self.goal_dist, 0.0], float)
        psi_B = math.pi + rng.normal(0.0, math.radians(2.0))
        v_B = 4.0 + rng.normal(0.0, 0.3)

        ships = [
            TinyShip(sid=100, pos=pos_A, goal=goal_A, psi=psi_A, v=v_A),
            TinyShip(sid=101, pos=pos_B, goal=goal_B, psi=psi_B, v=v_B),
        ]
        self.state = TinyShipState(ships)
        self.t = 0.0
        return None, {}  # 这里只是 demo，不返回 obs

    def step(self, actions: Dict[str, np.ndarray]):
        """
        actions: dict[agent_id -> np.array([dpsi_norm, v_norm])]
        返回：(done, info)，其中 info 包含 succ/coll/tout 标记
        """
        ships = self.state.ships
        id2idx = {100: 0, 101: 1}
        aid2sid = {"A": 100, "B": 101}

        # 应用控制
        for aid, act in actions.items():
            sid = aid2sid[aid]
            si = ships[id2idx[sid]]
            if si.reached:
                continue
            dpsi_norm = float(np.clip(act[0], -1.0, 1.0))
            v_norm = float(np.clip(act[1], -1.0, 1.0))

            dpsi = dpsi_norm * self.dpsi_max
            v_cmd = 0.5 * (v_norm + 1.0) * (self.v_max - self.v_min) + self.v_min

            si.psi = float(wrap_to_pi(si.psi + dpsi * self.dt))
            si.v = float(np.clip(v_cmd, self.v_min, self.v_max))

        # 积分位置
        for si in ships:
            if si.reached:
                continue
            dx = si.v * math.cos(si.psi) * self.dt
            dy = si.v * math.sin(si.psi) * self.dt
            si.pos = si.pos + np.array([dx, dy], float)
            # 到达检测
            if np.linalg.norm(si.pos - si.goal) <= self.goal_tol:
                si.reached = True

        self.t += self.dt

        # 碰撞检测（任意两船距离小于 collide_thr）
        coll = False
        for i in range(len(ships)):
            for j in range(i+1, len(ships)):
                d = np.linalg.norm(ships[i].pos - ships[j].pos)
                if d < self.collide_thr:
                    coll = True
                    break

        all_reached = all(s.reached for s in ships)
        tout = (self.t >= self.T_max)

        done = coll or all_reached or tout
        info = dict(coll=coll, succ=all_reached, tout=tout)
        return done, info


# ======================================================================
# 二、原始 RuleBaseline 控制器（轻微整理自你提供的代码）
# ======================================================================

@dataclass
class RuleParams:
    # -------- Gate 阈值 --------
    R_turn: float = 180.0
    ahead_cos: float = 0.2      # 朝向目标门：cosθ > ahead_cos（θ越小越“正前”）
    T_gate: float = 110.0       # TCPA 窗
    D_gate: float = 40.0        # DCPA 窗

    # -------- 角色判定（简化 COLREGs）--------
    th_headon: float = math.radians(18.0)   # 对遇方位阈
    th_heading_diff: float = math.radians(150.0)  # 对遇航向差 >150°
    th_cross: float = math.radians(112.5)   # 前半空间上界

    # -------- 趋目标（基础路径跟踪器）--------
    heading_scale: float = math.radians(22.5)  # tanh归一化尺度（小=更激进）
    lp_alpha: float = 0.6                      # 航向指令一阶低通
    v_cruise_k: float = 0.90                   # v_max 比例
    v_min_k: float = 0.25                      # 基础最小行驶
    align_pow: float = 2.0                     # 速度随对齐度幂次
    slow_radius_mul: float = 2.0               # 慢行圈半径倍数*goal_tol
    stop_k: float = 0.05                       # 接近停船速度

    # -------- 避碰基础参数（给路/直航）--------
    turn_right_frac: float = 0.95              # 右转占 dpsi_max 的最大幅度
    standon_eps: float = 0.12                  # 直航船微右偏
    v_slow_frac: float = 0.55                  # 避碰时速度比例

    # -------- 混合与回正 --------
    sev_smooth: float = 0.6                    # sev EMA
    decay_when_safe: float = 0.85              # 解除风险后 sev 衰减
    min_mix_when_unknown: float = 0.3          # 角色不明时的混合上限

class RuleBaselineController:
    """
    Baseline: “基础路径跟踪器 + 风险Gate触发的避碰控制器 + 连续混合 + 回正衰减”
    这里默认使用 env.state.ships（真值）做风险判断。
    """
    def __init__(self, env, params: RuleParams = RuleParams()):
        self.env = env
        self.P = params
        self._prev_dpsi_norm: Dict[int, float] = {}
        self._sev_ema: Dict[int, float] = {}  # sid -> 平滑 sev

    def reset(self):
        self._prev_dpsi_norm.clear()
        self._sev_ema.clear()

    # -------- 原始 act：使用真值 ships --------
    def act(self) -> Dict[str, np.ndarray]:
        ships = self.env.state.ships
        tc, dc, _ = tcpa_dcpa_matrix(ships)  # 推进前风险几何

        actions: Dict[str, np.ndarray] = {}

        for i, aid in enumerate(self.env.agents):
            si = ships[i]
            if si.reached:
                actions[aid] = np.array([0.0, -1.0], np.float32)
                continue

            # 1) 基础路径跟踪
            dpsi_goal, err_abs = self._goal_heading_cmd(si)
            dist_to_goal = float(np.linalg.norm(si.pos - si.goal))
            v_goal = self._goal_speed_cmd(err_abs, dist_to_goal)

            # 2) Gate & severity
            sev, role, dpsi_avoid, v_avoid, has_gate = self._avoid_cmd_for_ship(i, ships, tc, dc)

            # 3) sev 平滑
            sev_s = self._sev_ema.get(si.sid, 0.0)
            sev_s = self.P.sev_smooth * sev_s + (1.0 - self.P.sev_smooth) * sev
            if not has_gate:
                sev_s *= self.P.decay_when_safe
            self._sev_ema[si.sid] = sev_s

            # 4) 混合
            w = sev_s
            if role == "unknown":
                w = min(w, self.P.min_mix_when_unknown)
            elif role in ("headon", "star"):
                w = max(w, 0.5)  # 对遇/右舷给保底权重

            dpsi = (1.0 - w) * dpsi_goal + w * dpsi_avoid
            v    = (1.0 - w) * v_goal    + w * v_avoid

            actions[aid] = np.array([
                float(np.clip(dpsi, -1.0, 1.0)),
                float(np.clip(v,    -1.0, 1.0))
            ], dtype=np.float32)
        return actions

    # -------- 细节函数保持不变 --------
    def _goal_heading_cmd(self, si) -> Tuple[float, float]:
        goal_dir = math.atan2(si.goal[1]-si.pos[1], si.goal[0]-si.pos[0])
        e = wrap_to_pi(goal_dir - si.psi)
        raw = math.tanh(e / max(1e-6, self.P.heading_scale))
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
        si = ships[i]
        ei = np.array([math.cos(si.psi), math.sin(si.psi)], float)
        vi = ei * si.v

        best_sev, best_role, best_dpsi, best_v, has_gate = 0.0, "unknown", 0.0, (2.0*self.P.v_slow_frac - 1.0), False

        for j, sj in enumerate(ships):
            if j == i or sj.reached:
                continue

            rel = sj.pos - si.pos
            d = float(np.linalg.norm(rel))
            if d <= 1e-9:
                continue
            r_hat = rel / d
            vj = np.array([math.cos(sj.psi), math.sin(sj.psi)], float) * sj.v
            vij = vj - vi

            # Gate 条件
            cond_radius = (d < self.P.R_turn)
            cond_ahead  = (np.dot(ei, r_hat) > self.P.ahead_cos)
            cond_closing= (np.dot(rel, vij) < 0.0)

            tij = float(tc[i, j]); dij = float(dc[i, j])
            cond_tcpa = (tij >= 0.0) and (tij <= self.P.T_gate)
            cond_dcpa = (dij <= self.P.D_gate)

            if not (cond_radius and cond_ahead and cond_closing and cond_tcpa and cond_dcpa):
                continue

            has_gate = True
            # Severity
            s_d = np.clip((self.P.R_turn - d) / self.P.R_turn, 0.0, 1.0)
            s_t = np.clip(1.0 - (tij / self.P.T_gate), 0.0, 1.0)
            s_c = np.clip(1.0 - (dij / self.P.D_gate), 0.0, 1.0)
            s_tc = max(s_t, s_c)
            sev  = 1.0 - (1.0 - s_d) * (1.0 - s_tc)

            role = self._role_simple(si, sj)
            beta = wrap_to_pi(math.atan2(rel[1], rel[0]) - si.psi)

            if role in ("headon", "star"):
                turn = self.P.turn_right_frac * np.clip((abs(beta) / self.P.th_cross), 0.2, 1.0)
                dpsi_avoid = - max(0.5 * self.P.standon_eps, turn * sev)
                v_avoid = 2.0 * self.P.v_slow_frac - 1.0
            elif role == "port":
                dpsi_avoid = - self.P.standon_eps * np.clip(sev, 0.25, 1.0)
                v_avoid    = None
            else:
                dpsi_avoid = - 0.5 * self.P.standon_eps * sev
                v_avoid    = 2.0 * min(self.P.v_slow_frac, 0.7*self.P.v_cruise_k) - 1.0

            if sev > best_sev:
                best_sev = sev
                best_role = role
                best_dpsi = dpsi_avoid
                best_v = (2.0*self.P.v_slow_frac - 1.0) if v_avoid is None else v_avoid

            coll_thr = self.env.collide_thr
            if has_gate:
                if (d < 2.0 * coll_thr) or (dij < 1.5 * coll_thr) or (0.0 <= tij < 12.0):
                    best_sev  = 1.0
                    best_role = "star" if best_role == "unknown" else best_role
                    best_dpsi = - self.P.turn_right_frac
                    best_v    = 2.0 * self.P.v_slow_frac - 1.0

        return best_sev, best_role, float(np.clip(best_dpsi, -1.0, 1.0)), float(np.clip(best_v, -1.0, 1.0)), has_gate

    def _role_simple(self, si, sj) -> str:
        beta = wrap_to_pi(math.atan2((sj.pos-si.pos)[1], (sj.pos-si.pos)[0]) - si.psi)
        hdg_diff = abs(wrap_to_pi(sj.psi - si.psi))
        if abs(beta) < self.P.th_headon and hdg_diff > self.P.th_heading_diff:
            return "headon"
        in_half = (abs(beta) <= math.radians(112.5))
        if in_half and beta > 0:
            return "star"
        if in_half and beta < 0:
            return "port"
        return "unknown"


# ======================================================================
# 三、AIS 轨迹管理器 & AIS-aware 控制器
# ======================================================================

class AISTrackManager:
    """
    全局 AIS 轨迹管理器：
      - 不区分接收方，任何 RxMsg 到达都可以用来更新该船的 AIS 估计；
      - 主要用于构造 “noisy + delayed” 的他船状态。
    """
    def __init__(self, ais_sim: AISCommsSim, max_age: float = 60.0):
        self.ais = ais_sim
        self.max_age = float(max_age)
        # sid -> dict(x, y, sog, cog, age, arrival_time)
        self._tracks: Dict[int, dict] = {}

    def reset(self):
        self._tracks.clear()

    def _sid_from_mmsi(self, mmsi: int) -> Optional[int]:
        # 与 AISCommsSim 中的内部约定保持一致：mmsi = 999000000 + sid
        sid = int(mmsi) - 999000000
        return sid

    def update_from_ready(self, ready: Dict[AgentId, List]):
        """
        ready: AISCommsSim.step 返回的 {rx_agent: List[RxMsg]}
        """
        for rx_agent, msgs in ready.items():
            for msg in msgs:
                sid = self._sid_from_mmsi(msg.mmsi)
                if sid is None:
                    continue
                cur = self._tracks.get(sid)
                # 以 arrival_time 更新（更晚到达的覆盖）
                if (cur is None) or (msg.arrival_time > cur["arrival_time"]):
                    self._tracks[sid] = dict(
                        x=float(msg.reported_x),
                        y=float(msg.reported_y),
                        sog=float(msg.reported_sog),
                        cog=float(msg.reported_cog),
                        age=float(msg.age),
                        arrival_time=float(msg.arrival_time),
                    )

    def get_estimate(self, sid: int) -> Optional[dict]:
        est = self._tracks.get(int(sid))
        if est is None:
            return None
        if est["age"] > self.max_age:
            return None
        return est


class RuleBaselineAISController(RuleBaselineController):
    """
    AIS-aware 版：
      - 自船仍使用真值 (TinyShip) 做路径跟踪；
      - 他船的风险判断使用 AIS noisy + delayed 估计。
    """
    def __init__(self, env, ais_mgr: AISTrackManager,
                 params: RuleParams = RuleParams()):
        super().__init__(env, params)
        self.ais_mgr = ais_mgr

    def act(self) -> Dict[str, np.ndarray]:
        ships_true = self.env.state.ships

        # 构造 “用于风险判断的 ships_ais”：
        class ShipProxy:
            __slots__ = ("sid", "pos", "goal", "psi", "v", "reached")
            def __init__(self, src):
                self.sid = src.sid
                self.pos = np.array(src.pos, float)
                self.goal = np.array(src.goal, float)
                self.psi = float(src.psi)
                self.v = float(src.v)
                self.reached = bool(src.reached)

        ships_ais: List[ShipProxy] = [ShipProxy(si) for si in ships_true]

        # 对每条船的“他船信息”使用 AIS 估计，这里采用全局轨迹（简化版本）：
        for s in ships_ais:
            est = self.ais_mgr.get_estimate(s.sid)
            # 这里可以选择：自船用真值，他船用 AIS。Demo 简化为：所有船都用 AIS 估计覆盖自身状态
            if est is not None:
                s.pos = np.array([est["x"], est["y"]], float)
                s.v   = float(max(0.0, est["sog"]))
                s.psi = float(est["cog"])

        # 基于 AIS 代理 ships_ais 计算 TCPA/DCPA
        tc, dc, _ = tcpa_dcpa_matrix(ships_ais)

        actions: Dict[str, np.ndarray] = {}

        for i, aid in enumerate(self.env.agents):
            si_true = ships_true[i]   # 自船使用真值做路径跟踪
            si_risk = ships_ais[i]    # 同一索引，对应的 AIS 代理

            if si_true.reached:
                actions[aid] = np.array([0.0, -1.0], np.float32)
                continue

            # 1) 自船路径跟踪仍基于真值
            dpsi_goal, err_abs = self._goal_heading_cmd(si_true)
            dist_to_goal = float(np.linalg.norm(si_true.pos - si_true.goal))
            v_goal = self._goal_speed_cmd(err_abs, dist_to_goal)

            # 2) 避碰：基于 AIS 代理 ships_ais 计算风险
            sev, role, dpsi_avoid, v_avoid, has_gate = \
                self._avoid_cmd_for_ship(i, ships_ais, tc, dc)

            # 3) sev 平滑（按自船 sid 索引）
            sev_s = self._sev_ema.get(si_true.sid, 0.0)
            sev_s = self.P.sev_smooth * sev_s + (1.0 - self.P.sev_smooth) * sev
            if not has_gate:
                sev_s *= self.P.decay_when_safe
            self._sev_ema[si_true.sid] = sev_s

            # 4) 混合
            w = sev_s
            if role == "unknown":
                w = min(w, self.P.min_mix_when_unknown)
            elif role in ("headon", "star"):
                w = max(w, 0.5)

            dpsi = (1.0 - w) * dpsi_goal + w * dpsi_avoid
            v    = (1.0 - w) * v_goal    + w * v_avoid

            actions[aid] = np.array([
                float(np.clip(dpsi, -1.0, 1.0)),
                float(np.clip(v,    -1.0, 1.0))
            ], dtype=np.float32)
        return actions


# ======================================================================
# 四、整合：在 TinyShipEnv 上用 AIS + RuleBaselineAISController 做一次测试
# ======================================================================

def run_single_episode_with_ais(seed: int = 0,
                                cfg_path: Optional[str] = None):
    # 1) 构建小环境
    env = TinyShipEnv()
    env.reset(seed=seed)

    # 2) 构建 AIS 仿真
    ais = AISCommsSim(cfg_path=cfg_path)
    ships = env.state.ships
    ships_ids = [s.sid for s in ships]
    agent_map: Dict[ShipId, AgentId] = {ships[i].sid: env.agents[i] for i in range(len(ships))}
    ais.reset(ships_ids, t0=0.0, agent_map=agent_map)

    ais_mgr = AISTrackManager(ais_sim=ais, max_age=60.0)
    ais_mgr.reset()

    # 3) 控制器：AIS-aware 版本
    ctrl = RuleBaselineAISController(env, ais_mgr)

    # 4) 仿真循环
    t = 0.0
    dt = env.dt
    ep_info = None
    step_count = 0

    while True:
        # 将真值转换为 AIS TrueState
        true_states: Dict[ShipId, TrueState] = {}
        for s in env.state.ships:
            true_states[s.sid] = TrueState(
                x=float(s.pos[0]),
                y=float(s.pos[1]),
                sog=float(s.v),
                cog=float(s.psi)
            )

        # AIS 一步
        ready = ais.step(t, true_states)
        ais_mgr.update_from_ready(ready)

        # 控制器输出动作（基于 AIS noisy 观测）
        actions = ctrl.act()

        done, info = env.step(actions)
        ep_info = info
        t += dt
        step_count += 1

        # 简单打印几步结果
        if step_count % 20 == 0 or done:
            print(f"[t={t:5.1f}s] step={step_count} ships:")
            for s in env.state.ships:
                print(f"  sid={s.sid} pos=({s.pos[0]:.1f},{s.pos[1]:.1f}) "
                      f"psi={math.degrees(s.psi):6.2f}° v={s.v:.2f} m/s "
                      f"reached={s.reached}")
            print("  info:", info)

        if done:
            break

    print("\n=== EPISODE DONE ===")
    print(f"steps={step_count}  T={t:.1f}s")
    print(f"succ={ep_info['succ']} coll={ep_info['coll']} tout={ep_info['tout']}")
    return ep_info


def main():
    print(">>> Run TinyShipEnv + AIS + RuleBaselineAISController demo")
    # 注意 cfg_path 改成你 ais_config.yaml 的实际路径
    info = run_single_episode_with_ais(seed=0,
                                       cfg_path=None)  # 使用 ais_comms 默认配置
    print("Final episode info:", info)


if __name__ == "__main__":
    main()
