from __future__ import annotations
import math
import numpy as np
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv
from typing import Dict, Any, Optional
from ..utils.seeding import RNG
from ..dynamics.ship import Ship
from ..scenario.config import SpawnConfig
from ..scenario.sampler import ScenarioSampler
from ..core.state import WorldState, StepCache
from ..core.actions import decode_actions, map_to_commands
from ..risk.tcpa_dcpa import tcpa_dcpa_matrix
from ..risk.prelimit import speed_soft_cap_before_step
from ..risk.postrisk import post_step_risk
from ..reward.shaping import task_reward, progress_phi
from ..reward.costs import cost_near, cost_rule, cost_coll_time
from ..observe.builder import build_observations, zero_observation
from ..core.termination import finalize_episode_rewards
from ..risk.guard_controller import GuardController


class MiniShipCoreEnv(ParallelEnv):
    """
    只负责“核心环境逻辑”的并行多船环境：

    - 船舶动力学推进
    - 场景采样与重置
    - 基础风险/碰撞/超时判定
    - 基础 reward（task_reward + finalize_episode_rewards）
    - 几何观测构造

    不包含：
    - 拉格朗日对偶与 Governor
    - 护栏（safety guard）
    - AIS 仿真、对偶快照等训练/日志逻辑
    """

    metadata = {"render_modes": ["none"]}

    def __init__(self, cfg: dict, guard_ctrl: GuardController | None = None):
        # --- 核心环境参数 ---
        self.N = int(cfg.get("N", 2))
        self.dt = float(cfg.get("dt", 0.5))
        self.T_max = float(cfg.get("T_max", 220.0))

        self.v_max = float(cfg.get("v_max", 2.0))
        self.v_min = float(cfg.get("v_min", 0.1))
        self.dv_max = float(cfg.get("dv_max", 0.8))
        self.dpsi_max = float(cfg.get("dpsi_max", math.radians(20)))

        # 目标/碰撞阈值
        self.goal_tol = float(cfg.get("goal_tol", 10.0))
        self.collide_thr = float(cfg.get("collide_thr", 12.0))

        # spawn 配置
        self.spawn_area = float(cfg.get("spawn_area", 240.0))
        self.spawn_margin = float(cfg.get("spawn_margin", 12.0))
        self.spawn_min_sep = float(cfg.get("spawn_min_sep", 40.0))
        self.spawn_goal_min_sep = float(cfg.get("spawn_goal_min_sep", 60.0))
        self.spawn_len = float(cfg.get("spawn_len", 160.0))
        self.spawn_retry = int(cfg.get("spawn_retry", 80))
        self.spawn_dir_jitter_deg = float(cfg.get("spawn_dir_jitter_deg", 6.0))
        self.spawn_mode = str(cfg.get("spawn_mode", "random_fixedlen"))

        # 风险阈值（用于风险/限速）
        self.risk_T_thr = float(cfg.get("risk_T_thr", 110.0))
        self.risk_D_thr = float(cfg.get("risk_D_thr", 40.0))

        # 推进前风险限速参数（视为“环境内在物理约束/交通规则”）
        self.vCruiseK = float(cfg.get("vCruiseK", 0.90))
        self.alphaOpen = float(cfg.get("alphaOpen", 0.20))
        self.capPow = float(cfg.get("capPow", 1.150))
        self.capGain = float(cfg.get("capGain", 0.20))
        self.K_release = int(cfg.get("K_release", 3))
        self.M_boost = int(cfg.get("M_boost", 14))
        self.dvBoost = float(cfg.get("dvBoost", 1.80))
        self.thrRisk_gate = float(cfg.get("thrRisk_gate", 0.25))

        # 奖励/收尾
        self.step_cost = float(cfg.get("step_cost", 0.5))
        self.r_arrival = float(cfg.get("r_arrival", 150.0))
        self.r_success_bonus = float(cfg.get("r_success_bonus", 300.0))
        self.r_collision_penalty = float(cfg.get("r_collision_penalty", 800.0))
        self.r_timeout_penalty_base = float(cfg.get("r_timeout_penalty_base", 400.0))
        # 避免极端碰撞奖励爆炸的额外惩罚（可以看作环境的一部分）
        self.r_collision_boom = float(cfg.get("r_collision_boom", 200.0))

        # COLREGs 模糊判别（仍属于环境规则的一部分）
        self.thHeadOn = math.radians(cfg.get("thHeadOn_deg", 15.0))
        self.thCross = math.radians(cfg.get("thCross_deg", 112.5))

        # 调试开关
        self._enable_debug = bool(cfg.get("enable_debug", False))

        # ====== episode-level debug stats (per-agent) ======
        # v_cmd < low_speed_thr 的比例，用于识别“摆烂慢走”
        self.low_speed_thr = float(cfg.get("low_speed_thr", max(self.v_min + 1e-6, 0.2 * self.v_max)))

        self._ep_goal_dist0 = np.zeros(self.N, dtype=np.float64)
        self._ep_goal_dist_min = np.zeros(self.N, dtype=np.float64)

        self._ep_steps_active = np.zeros(self.N, dtype=np.int64)          # 仅统计 ship 未 reached 时的控制步
        self._ep_steps_low_speed = np.zeros(self.N, dtype=np.int64)

        self._ep_v_cmd_sum = np.zeros(self.N, dtype=np.float64)
        self._ep_v_target_sum = np.zeros(self.N, dtype=np.float64)
        self._ep_v_sum = np.zeros(self.N, dtype=np.float64)

        self._ep_heading_err_abs_sum = np.zeros(self.N, dtype=np.float64)
        self._ep_reached_any = np.zeros(self.N, dtype=bool)

        # 组件：采样器、RNG、状态缓存
        self.spawn_cfg = SpawnConfig(
            N=self.N,
            spawn_area=self.spawn_area,
            spawn_margin=self.spawn_margin,
            spawn_min_sep=self.spawn_min_sep,
            spawn_goal_min_sep=self.spawn_goal_min_sep,
            spawn_len=self.spawn_len,
            spawn_retry=self.spawn_retry,
            spawn_dir_jitter_deg=self.spawn_dir_jitter_deg,
            collide_thr=self.collide_thr,
            v_min=self.v_min,
            v_max=self.v_max,
            mode=self.spawn_mode,
        )
        self.scenario = ScenarioSampler(self.spawn_cfg)
        self.rng = RNG()
        self.cache = StepCache()

        # PettingZoo 相关
        self.agents = [f"ship_{i + 1}" for i in range(self.N)]
        self.possible_agents = list(self.agents)
        self.K_neighbors = int(cfg.get("numNeighbors", 4))
        self.F_nei = 11
        self.F_edge = 8

        obs_dim = 8 + self.K_neighbors * self.F_nei + self.K_neighbors * self.F_edge + 1
        self._obs_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._act_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 运行态
        self.state: WorldState | None = None
        self.reached = np.zeros(self.N, bool)
        self._last_phi = None

        # 推进前限速内部状态（交给 speed_soft_cap_before_step 维护）
        self._guard_safeCount = np.zeros(self.N, int)
        self._guard_boostCount = np.zeros(self.N, int)

        # 是否输出调试信息（仍属于环境自身）
        self.enable_debug = bool(cfg.get("enable_debug", False))

        # ====== 护栏相关 ======
        # 如果外部传进来 guard_ctrl，就用外部的；否则默认不使用护栏
        self.guard_ctrl: GuardController | None = guard_ctrl

        # 护栏惩罚系数（直接影响 reward），默认和大 env 一致
        self.lam_guard = float(cfg.get("lam_guard", 0.0))

        # 是否处于评估模式（用于区分训练/评估护栏逻辑）
        self._eval_mode: bool = bool(cfg.get("is_eval", False))

        # === 新增：缓存最近一次 step/reset 的 infos ===
        self._last_infos: Dict[str, Dict[str, Any]] = {}
        
        self._enable_debug = self.enable_debug

    # ========== PettingZoo 接口 ==========

    @property
    def observation_spaces(self):
        return {aid: self._obs_space for aid in self.agents}

    @property
    def action_spaces(self):
        return {aid: self._act_space for aid in self.agents}

    def action_space(self, agent):
        return self._act_space

    def observation_space(self, agent):
        return self._obs_space

    def seed(self, seed: int | None = None):
        self.rng.seed(seed)

    # ---------- helpers: safe scalar/list + ship state extraction ----------
    @staticmethod
    def _as_float(x, default: float = float("nan")) -> float:
        try:
            return float(x)
        except Exception:
            return default

    @staticmethod
    def _as_list(x) -> Any:
        """Make JSON-friendly: np -> list[float], scalar -> float, None -> None."""
        if x is None:
            return None
        try:
            import numpy as _np
            if isinstance(x, _np.ndarray):
                return [float(z) for z in x.reshape(-1)]
        except Exception:
            pass
        # generic iterable
        try:
            if hasattr(x, "__iter__") and not isinstance(x, (str, bytes, dict)):
                return [float(z) for z in x]
        except Exception:
            pass
        return MiniShipCoreEnv._as_float(x, default=x)

    def _ship_xy_psi_v_goal(self, ship: Ship):
        """Return (x,y,psi,v,gx,gy,goal_dist,reached) with robust getattr fallbacks."""
        # pos
        pos = getattr(ship, "pos", None)
        if pos is None:
            pos = getattr(ship, "p", None)
        if pos is None:
            x = y = float("nan")
        else:
            x = self._as_float(pos[0])
            y = self._as_float(pos[1])

        # goal
        goal = getattr(ship, "goal", None)
        if goal is None:
            goal = getattr(ship, "goal_pos", None)
        if goal is None:
            gx = gy = float("nan")
        else:
            gx = self._as_float(goal[0])
            gy = self._as_float(goal[1])

        psi = self._as_float(getattr(ship, "psi", float("nan")))
        v = self._as_float(getattr(ship, "v", float("nan")))

        # reached flag (prefer ship.reached)
        reached = bool(getattr(ship, "reached", False))

        # goal distance
        try:
            dx = gx - x
            dy = gy - y
            goal_dist = float((dx * dx + dy * dy) ** 0.5)
        except Exception:
            goal_dist = float("nan")

        return x, y, psi, v, gx, gy, goal_dist, reached

    @staticmethod
    def _wrap_to_pi(a):
        """Vectorized wrap to (-pi, pi]."""
        a = np.asarray(a, dtype=np.float64)
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    def _compute_heading_err_abs(self) -> np.ndarray:
        """abs(wrap(bearing_to_goal - psi)) for each ship, using current state."""
        assert self.state is not None
        pos = np.stack([s.pos for s in self.state.ships], axis=0).astype(np.float64)
        psi = np.array([s.psi for s in self.state.ships], dtype=np.float64)
        goals = np.stack([s.goal for s in self.state.ships], axis=0).astype(np.float64)
        dx = goals[:, 0] - pos[:, 0]
        dy = goals[:, 1] - pos[:, 1]
        bearing = np.arctan2(dy, dx)
        err = self._wrap_to_pi(bearing - psi)
        return np.abs(err)

    def _compute_goal_dist_vec(self) -> np.ndarray:
        """Goal distance vector for each ship using current state."""
        assert self.state is not None
        pos = np.stack([s.pos for s in self.state.ships], axis=0).astype(np.float64)
        goals = np.stack([s.goal for s in self.state.ships], axis=0).astype(np.float64)
        d = goals - pos
        return np.sqrt(np.sum(d * d, axis=1))

    # ---------------------- reset ----------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed(seed)

        # --- 新增：episode 计数与“终局只打印一次”的门闩 ---
        if not hasattr(self, "_episode_idx"):
            self._episode_idx = 0
        self._episode_idx += 1
        self._ep_end_dbg_printed = False

        # 支持通过 options["eval"] 切换训练/评估模式
        if options is not None and "eval" in options:
            self._eval_mode = bool(options.get("eval"))

        # 清空推进前限速内部状态
        self._guard_safeCount[:] = 0
        self._guard_boostCount[:] = 0

        # 采样初始场景
        starts, goals, psi0, v0 = self.scenario.sample(self.rng.rs)
        ships = [
            Ship(i + 1, starts[i].copy(), goals[i].copy(), float(psi0[i]), float(v0[i]))
            for i in range(self.N)
        ]
        self.reached[:] = False
        self.state = WorldState(ships=ships, t=0.0, step=0)
        self._last_phi = progress_phi(ships)

        # ====== init episode stats ======
        gd0 = self._compute_goal_dist_vec()
        self._ep_goal_dist0[:] = gd0
        self._ep_goal_dist_min[:] = gd0
        self._ep_steps_active[:] = 0
        self._ep_steps_low_speed[:] = 0
        self._ep_v_cmd_sum[:] = 0.0
        self._ep_v_target_sum[:] = 0.0
        self._ep_v_sum[:] = 0.0
        self._ep_heading_err_abs_sum[:] = 0.0
        self._ep_reached_any[:] = False


        infos: Dict[str, Dict[str, Any]] = {}
        for i, aid in enumerate(self.agents):
            ship = self.state.ships[i]
            x, y, psi, v, gx, gy, gd, reached = self._ship_xy_psi_v_goal(ship)
            infos[aid] = {
                "t": 0.0,
                "step": 0,
                # kinematics / goal
                "x": x, "y": y,
                "psi": psi, "v": v,
                "gx": gx, "gy": gy,
                "goal_dist": gd,
                "reached_goal": bool(reached),
                
                # ====== episode stats (at reset) ======
                "goal_dist0": float(self._ep_goal_dist0[i]),
                "goal_dist_min": float(self._ep_goal_dist_min[i]),
                "goal_dist_delta": float(0.0),
                "v_cmd_mean": float(0.0),
                "v_target_mean": float(0.0),
                "v_mean": float(v),  # reset 时取当前 v
                "heading_err_abs_mean": float(0.0),
                "steps_low_speed": float(0.0),
                "low_speed_thr": float(self.low_speed_thr),
                "reached_any_step": bool(self._ep_reached_any[i]),

                "step_cost": float(self.step_cost), # config sanity (verify step_cost really applied)
                "T_max": float(self.T_max),
                "dt": float(self.dt),
            }


        # 仅使用几何观测
        obs_raw = build_observations(
            self.state.ships,
            self.K_neighbors,
            self.spawn_mode,
            self.spawn_area,
            self.spawn_len,
            self.v_max,
        )
        # Remap observation keys from ship_id ("1", "2") to agent_id ("ship_1", "ship_2")
        obs = {}
        for i, aid in enumerate(self.agents):
            ship_id_str = str(i + 1)
            if ship_id_str in obs_raw:
                obs[aid] = obs_raw[ship_id_str]
            else:
                obs[aid] = zero_observation(self._obs_space)
        # cache last infos (for callbacks)
        self._last_infos = infos

        return obs, infos

    # ---------------------- step ----------------------

    def step(self, actions: dict):
        assert self.state is not None, "Environment must be reset before stepping."

        if not hasattr(self, "_step_count"):
            self._step_count = 0
        self._step_count += 1

        # A) 动作解码/映射
        A = decode_actions(actions, self._act_space, self.agents)
        dpsi_rl, v_cmd = map_to_commands(A, self.dpsi_max, self.v_min, self.v_max)

        # B) 推进前风险（tc0/dc0）——供限速和风险评估使用
        self.cache.tc0, self.cache.dc0, _ = tcpa_dcpa_matrix(self.state.ships)
        self.cache.pos = np.stack([s.pos for s in self.state.ships], axis=0)
        self.cache.psi = np.array([s.psi for s in self.state.ships])
        self.cache.v_all = np.array([s.v for s in self.state.ships])

        # B0) 护栏（如果启用）：先在动作层做一次“安全投影”
        N = self.N
        _guard_trig = np.zeros(N, dtype=np.float64)
        _step_guard_mean = 0.0
        _guard_enabled = False

        if self.guard_ctrl is not None:
            dpsi_rl, v_cmd, _guard_trig, _step_guard_mean, _guard_enabled = (
                self.guard_ctrl.apply(
                    ships=self.state.ships,
                    dpsi_rl=dpsi_rl,
                    v_cmd=v_cmd,
                    dt=self.dt,
                    dpsi_max=self.dpsi_max,
                    v_min=self.v_min,
                    v_max=self.v_max,
                    risk_T_thr=self.risk_T_thr,
                    risk_D_thr=self.risk_D_thr,
                    collide_thr=self.collide_thr,
                    step_idx=self.state.step,
                    eval_mode=self._eval_mode,
                    yaw_rate_max=None,
                    brake_gain=0.4,
                    steer_gain=0.4,
                    K_guard=4,
                    use_float32=True,
                )
            )

        # C) 推进前速度软限（基于风险/距离 + v_cmd，经护栏修正后的动作）
        v_target, v_cap, bestX0 = speed_soft_cap_before_step(
            self.state.ships,
            self.cache.tc0,
            self.cache.dc0,
            self.v_max,
            self.v_min,
            self.risk_T_thr,
            self.risk_D_thr,
            self.thrRisk_gate,
            self.vCruiseK,
            self.alphaOpen,
            self.capPow,
            self.capGain,
            self.K_release,
            self.M_boost,
            1.0,
            self._guard_safeCount,
            self._guard_boostCount,
            v_cmd.copy(),
        )

        # ====== update episode stats (pre-advance) ======
        active_mask = np.array([not s.reached for s in self.state.ships], dtype=bool)
        active_f = active_mask.astype(np.float64)
        heading_err_abs = self._compute_heading_err_abs()

        self._ep_steps_active += active_mask.astype(np.int64)
        self._ep_steps_low_speed += ((v_cmd < self.low_speed_thr) & active_mask).astype(np.int64)
        self._ep_v_cmd_sum += v_cmd.astype(np.float64) * active_f
        self._ep_v_target_sum += v_target.astype(np.float64) * active_f
        self._ep_heading_err_abs_sum += heading_err_abs * active_f

        # D) 物理推进（含 dvBoost）
        prev_v = np.array([s.v for s in self.state.ships], dtype=np.float64)
        for i, ship in enumerate(self.state.ships):
            if ship.reached:
                continue
            boost = self._guard_boostCount[i] > 0
            dv_max_eff = self.dv_max * (self.dvBoost if boost else 1.0)
            ship.advance(
                dpsi_rl[i],
                float(v_target[i]),
                self.dt,
                self.dpsi_max,
                dv_max_eff,
                self.v_min,
                self.v_max,
            )
            if boost:
                self._guard_boostCount[i] -= 1

        self.state.t += self.dt
        self.state.step += 1

        # v_mean: use post-advance actual speed for active steps
        v_after = np.array([s.v for s in self.state.ships], dtype=np.float64)
        self._ep_v_sum += v_after * active_f

        # E) 到达判定 + 到达奖励
        arrival_bonus = np.zeros(self.N, np.float64)
        reach_cnt = 0
        for i, ship in enumerate(self.state.ships):
            prev_reached = ship.reached
            ship.check_reached(self.goal_tol)
            self.reached[i] = ship.reached
            if (not prev_reached) and ship.reached:
                arrival_bonus[i] = self.r_arrival
            if ship.reached:
                reach_cnt += 1


        # reached_any_step flag (ever reached during this episode)
        self._ep_reached_any |= self.reached

        # goal_dist_min update (post-advance + post-reach-check)
        gd_now = self._compute_goal_dist_vec()
        self._ep_goal_dist_min = np.minimum(self._ep_goal_dist_min, gd_now)

        # F) 推进后风险
        self.cache.tc, self.cache.dc, _ = tcpa_dcpa_matrix(self.state.ships)
        self.cache.risk, self.cache.dmin_i, self.cache.vj_max = post_step_risk(
            self.state.ships,
            self.cache.tc,
            self.cache.dc,
            self.risk_T_thr,
            self.risk_D_thr,
        )

        # G) 任务奖励（不做拉格朗日合成）
        r_task, self._last_phi = task_reward(
            self.state.ships,
            dpsi_rl,
            self._last_phi,
            self.cache.risk,
            self.v_max,
            self.step_cost,
            arrival_bonus,
        )

        # H) 代价项（仍由环境计算，但不在这里做自适应 λ）
        v_self = np.array([s.v for s in self.state.ships], dtype=np.float64)
        c_near = cost_near(
            self.cache.risk,
            self.cache.vj_max,
            v_self,
            self.cache.dmin_i,
            self.collide_thr,
            tau=6.0,
        )

        dv_act = v_self - prev_v
        c_rule = cost_rule(
            self.state.ships,
            self.cache.tc,
            self.cache.dc,
            self.risk_T_thr,
            self.risk_D_thr,
            self.thHeadOn,
            self.thCross,
            dpsi_rl,
            dv_act,
            self.dv_max,
        )

        coll, pair, c_coll, c_time, timeout = cost_coll_time(
            self.state.ships,
            self.collide_thr,
            self.state.t,
            self.T_max,
        )

        # I) 当前版本：环境 reward 仅使用 r_task + 简单碰撞惩罚
        r_vec = r_task.copy()
        if np.any(c_coll > 0.5):
            r_vec = r_vec - self.r_collision_boom

        # I2) 护栏惩罚（如果启用）
        if self.guard_ctrl is not None and self.lam_guard > 0.0 and _guard_enabled:
            r_vec = r_vec - float(self.lam_guard) * _guard_trig.astype(np.float64)

        # J) 终止与终局奖励
        success_all = bool(np.all(self.reached))
        done_global = coll or success_all or timeout

        # Expose terminal reward finalization details to outer wrappers/debuggers.
        # These are only populated on the *terminal* step.
        r_vec_pre_finalize = None
        r_vec_post_finalize = None
        term_code = None

        # --- 新增：finalize 前后 r_vec 打印（每个 episode 只打印一次终局）---
        dbg_ep_end = bool(self._enable_debug) and (not getattr(self, "_ep_end_dbg_printed", False)) and done_global
        if dbg_ep_end:
            reach_max = float(self.v_max * self.T_max)
            r_pre = r_vec.copy()
            print(
                f"[EP-END][core] ep={getattr(self,'_episode_idx',-1)} "
                f"t={self.state.t:.2f} step={self.state.step} "
                f"done_global=1 coll={int(bool(coll))} timeout={int(bool(timeout))} success_all={int(bool(success_all))} "
                f"reach_max=v_max*T_max={reach_max:.2f}m"
            )
            print("  r_vec_pre_finalize:", np.array2string(r_pre, precision=6, floatmode="fixed"))


        if done_global:
            # 关键：先保存 finalize 之前的 r_vec
            r_pre_finalize = r_vec.copy()

            r_vec, term_code = finalize_episode_rewards(
                r_vec,
                success_all,
                timeout,
                self.state.t,
                self.T_max,
                self.spawn_len,
                self.r_success_bonus,
                self.r_timeout_penalty_base,
                self.r_collision_penalty,
            )
            r_vec_post_finalize = r_vec.copy()

            # --- 新增：finalize 后打印 + per-ship episode 汇总（每个 episode 只打印一次）---
            if dbg_ep_end:
                r_post = r_vec.copy()
                delta = r_post - r_pre_finalize
                print("  r_vec_post_finalize:", np.array2string(r_post, precision=6, floatmode="fixed"))
                print("  delta(post-pre)    :", np.array2string(delta, precision=6, floatmode="fixed"), " term_code=", term_code)

                # 终局原因字符串（与下面 infos.term_reason 逻辑一致）
                if bool(coll):
                    reason = "collision"
                elif bool(success_all):
                    reason = "success"
                else:
                    reason = "timeout"
                print(f"  term_reason={reason}  goal_tol={self.goal_tol:.2f}m  low_speed_thr={self.low_speed_thr:.3f}m/s")

                # goal_dist_end 使用当前时刻的 gd_now（你上面刚更新过 _ep_goal_dist_min，这里再算一次更直观）
                gd_end = self._compute_goal_dist_vec()

                print("  --- per-ship episode summary ---")
                for i, aid in enumerate(self.agents):
                    denom = float(max(int(self._ep_steps_active[i]), 1))
                    v_mean = float(self._ep_v_sum[i] / denom)
                    v_tgt_mean = float(self._ep_v_target_sum[i] / denom)
                    low_speed_ratio = float(self._ep_steps_low_speed[i] / denom)

                    print(
                        f"  [{aid}] "
                        f"goal_dist0={float(self._ep_goal_dist0[i]):.2f}  "
                        f"goal_dist_min={float(self._ep_goal_dist_min[i]):.2f}  "
                        f"goal_dist_end={float(gd_end[i]):.2f}  "
                        f"reached_any_step={int(bool(self._ep_reached_any[i]))}  "
                        f"reached_now={int(bool(self.reached[i]))}  "
                        f"v_mean={v_mean:.3f}  "
                        f"v_target_mean={v_tgt_mean:.3f}  "
                        f"steps_low_speed={low_speed_ratio:.3f} "
                        f"(low/active={int(self._ep_steps_low_speed[i])}/{int(self._ep_steps_active[i])})"
                    )

                # 标记：本 episode 终局已打印
                self._ep_end_dbg_printed = True


        # K) 封装返回

        # ==== RewardDBG：前 5 次打印分解的奖励与代价 ====
        if getattr(self, "_enable_debug", False):
            if not hasattr(self, "_reward_dbg_count"):
                self._reward_dbg_count = 0

            if self._reward_dbg_count < 5:
                self._reward_dbg_count += 1
                print(f"[RewardDBG][core] step={self._step_count}, dbg#{self._reward_dbg_count}")
                # 下面这些变量在当前作用域里如果存在就打印
                if "r_task" in locals():
                    print("  r_task:", r_task)
                if "c_near" in locals():
                    print("  c_near:", c_near,
                          "c_rule:", c_rule,
                          "c_coll:", c_coll,
                          "c_time:", c_time)
                if "r_vec" in locals():
                    print("  r_vec:", r_vec)
                if "done_global" in locals():
                    print("  done_global:", done_global)


        rewards = {aid: float(r_vec[i]) for i, aid in enumerate(self.agents)}
        terminations = {aid: False for aid in self.agents}
        truncations = {aid: False for aid in self.agents}

        infos: Dict[str, Dict[str, Any]] = {aid: {"t": self.state.t, "step": self.state.step} for aid in self.agents}

        # 把本步护栏开关 & 触发均值写入 infos
        for aid in self.agents:
            infos[aid]["guard_enabled"] = bool(_guard_enabled)
            infos[aid]["guard_trig_step_mean"] = float(_step_guard_mean)

        # ---- per-agent kinematics + goal + actions/commands (JSON-friendly) ----
        # raw RL action (as received by core env)
        for i, aid in enumerate(self.agents):
            ship = self.state.ships[i]
            x, y, psi, v, gx, gy, gd, reached = self._ship_xy_psi_v_goal(ship)

            info = infos[aid]
            info.update({
                "x": x, "y": y,
                "psi": psi, "v": v,
                "gx": gx, "gy": gy,
                "goal_dist": gd,
                "reached_goal": bool(reached),
                # for debugging why timeout: how many reached by now
                "reach_cnt": int(np.sum(self.reached)),
                "success_all": bool(np.all(self.reached)),
                "timeout": bool(timeout),
                "coll": bool(coll),
                "step_cost": float(self.step_cost),
            })

            # raw action -> decoded A -> commands
            # actions[aid] may be np array; make it list
            info["a_raw"] = self._as_list(actions.get(aid, None)) if isinstance(actions, dict) else None
            # A is aligned with agents ordering
            try:
                info["a_decoded"] = self._as_list(A[i])
            except Exception:
                pass

            # commands
            info["dpsi_cmd"] = self._as_float(dpsi_rl[i])
            info["v_cmd"] = self._as_float(v_cmd[i])
            info["v_target"] = self._as_float(v_target[i])
            info["v_cap"] = self._as_float(v_cap[i])

            # dynamics delta
            try:
                info["prev_v"] = self._as_float(prev_v[i])
                info["dv_act"] = self._as_float((v_self[i] - prev_v[i]))
            except Exception:
                pass


            # ====== episode stats snapshot (per-agent) ======
            denom = float(max(int(self._ep_steps_active[i]), 1))
            info["goal_dist0"] = float(self._ep_goal_dist0[i])
            info["goal_dist_min"] = float(self._ep_goal_dist_min[i])
            info["goal_dist_delta"] = float(self._ep_goal_dist0[i] - float(gd))

            info["v_cmd_mean"] = float(self._ep_v_cmd_sum[i] / denom)
            info["v_target_mean"] = float(self._ep_v_target_sum[i] / denom)
            info["v_mean"] = float(self._ep_v_sum[i] / denom)
            info["heading_err_abs_mean"] = float(self._ep_heading_err_abs_sum[i] / denom)

            info["steps_low_speed"] = float(self._ep_steps_low_speed[i] / denom)
            info["low_speed_thr"] = float(self.low_speed_thr)
            info["reached_any_step"] = bool(self._ep_reached_any[i])

        # 终局统一终止标注
        if done_global:
            if coll:
                reason = "collision"
                for aid in self.agents:
                    terminations[aid] = True
                    truncations[aid] = False
                    infos[aid]["term_reason"] = reason
            elif success_all:
                reason = "success"
                for aid in self.agents:
                    terminations[aid] = True
                    truncations[aid] = False
                    infos[aid]["term_reason"] = reason
            else:
                reason = "timeout"
                for aid in self.agents:
                    terminations[aid] = False
                    truncations[aid] = True
                    infos[aid]["term_reason"] = reason

            # Expose terminal bookkeeping (global) via per-agent infos for RLlib logging.
            pre_list = r_vec_pre_finalize.tolist() if r_vec_pre_finalize is not None else None
            post_list = r_vec_post_finalize.tolist() if r_vec_post_finalize is not None else None
            for aid in self.agents:
                infos[aid]["term_code"] = int(term_code) if term_code is not None else None
                if pre_list is not None:
                    infos[aid]["r_vec_pre_finalize"] = pre_list
                if post_list is not None:
                    infos[aid]["r_vec_post_finalize"] = post_list

        # Terminal diagnostic fields for outer wrappers/debuggers (only on terminal step).
        # Store as per-agent scalars so wrappers can read without vector semantics.
        if done_global:
            for i, aid in enumerate(self.agents):
                try:
                    infos[aid]["r_vec_pre_finalize"] = float(r_vec_pre_finalize[i]) if r_vec_pre_finalize is not None else None
                    infos[aid]["r_vec_post_finalize"] = float(r_vec_post_finalize[i]) if r_vec_post_finalize is not None else float(r_vec[i])
                except Exception:
                    infos[aid]["r_vec_pre_finalize"] = None
                    infos[aid]["r_vec_post_finalize"] = None
                infos[aid]["term_code"] = int(term_code) if term_code is not None else None

            # 终局护栏触发率统计
            if self.guard_ctrl is not None:
                ep_guard_rate = self.guard_ctrl.get_episode_guard_rate()
                for aid in self.agents:
                    infos[aid]["guard_trig_rate_ep"] = float(ep_guard_rate)

        # 代价分量写入 infos，方便外部对偶模块/日志使用
        for i, aid in enumerate(self.agents):
            infos[aid]["c_near"] = float(c_near[i])
            infos[aid]["c_rule"] = float(c_rule[i])
            infos[aid]["c_coll"] = float(c_coll[i])
            infos[aid]["c_time"] = float(c_time[i])
            infos[aid]["risk"] = float(self.cache.risk[i])

        # L) 构建下一步观测
        if done_global:
            observations = {
                aid: zero_observation(self._obs_space)
                for aid in self.agents
            }
        else:
            obs_raw = build_observations(
                self.state.ships,
                self.K_neighbors,
                self.spawn_mode,
                self.spawn_area,
                self.spawn_len,
                self.v_max,
            )
            # Remap observation keys from ship_id ("1", "2") to agent_id ("ship_1", "ship_2")
            observations = {}
            for i, aid in enumerate(self.agents):
                ship_id_str = str(i + 1)
                if ship_id_str in obs_raw:
                    observations[aid] = obs_raw[ship_id_str]
                else:
                    observations[aid] = zero_observation(self._obs_space)

        # === 关键：把最后一步 infos 缓存下来，供 RLlib 回调读取 ===
        self._last_infos = infos

        return observations, rewards, terminations, truncations, infos

    # ========== 其他接口 ==========

    def render(self):
        return None

    def close(self):
        return None


# miniship/envs/miniship_core_env.py

from pettingzoo.utils.conversions import parallel_to_aec


def parallel_env(**env_config):
    """PettingZoo 标准入口（parallel API）"""
    return MiniShipCoreEnv(env_config)


def env(**env_config):
    """若以后需要 AEC API，可以用 parallel_to_aec 包一层"""
    return parallel_to_aec(parallel_env(**env_config))
