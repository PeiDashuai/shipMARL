
"""
[DEPRECATED] 这个文件已经废弃。
真正的 MiniShipParallelEnv 在 miniship/core/miniship_core_env.py 里。

保留这个文件只是为了防止旧代码 import 失败。
任何新的修改都应该去改 miniship_core_env.py。
"""


from __future__ import annotations
import math, numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv

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
from ..reward.lagrangian import (
    DualState, combine_reward,
    dual_update_end_of_episode,
)
from ..observe.builder import build_observations, zero_observation
from ..core.termination import finalize_episode_rewards
from ..risk.safety_guard import safety_guard_hard, safety_guard_ghost, safety_guard_hybrid
# from ..risk.safety_guard_strict import strict_guard_multi

# ==== AIS 仿真相关（外部模块） ====
from ais_comms.ais_comms import AISCommsSim
from ais_comms.obs_builder import AISObservationBuilder
from ais_comms.datatypes import TrueState, ShipId, AgentId, Ts


class MiniShipParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["none"]}

    def __init__(self, cfg: dict):
        # --- 基本参数 ---
        self.N = int(cfg.get("N", 2))
        self.dt = float(cfg.get("dt", 0.5))
        self.T_max = float(cfg.get("T_max", 220.0))
        self.v_max = float(cfg.get("v_max", 2.0))
        self.v_min = float(cfg.get("v_min", 0.1))
        self.dv_max = float(cfg.get("dv_max", 0.8))
        self.dpsi_max = float(cfg.get("dpsi_max", math.radians(20)))

        self.goal_tol = float(cfg.get("goal_tol", 10.0))
        self.collide_thr = float(cfg.get("collide_thr", 12.0))

        # spawn
        self.spawn_area = float(cfg.get("spawn_area", 240.0))
        self.spawn_margin = float(cfg.get("spawn_margin", 12.0))
        self.spawn_min_sep = float(cfg.get("spawn_min_sep", 40.0))
        self.spawn_goal_min_sep = float(cfg.get("spawn_goal_min_sep", 60.0))
        self.spawn_len = float(cfg.get("spawn_len", 160.0))
        self.spawn_retry = int(cfg.get("spawn_retry", 80))
        self.spawn_dir_jitter_deg = float(cfg.get("spawn_dir_jitter_deg", 6.0))
        self.spawn_mode = str(cfg.get("spawn_mode", "circle_center"))

        # 风险阈值
        self.risk_T_thr = float(cfg.get("risk_T_thr", 110.0))
        self.risk_D_thr = float(cfg.get("risk_D_thr", 40.0))

        # 推进前风险限速参数
        self.vCruiseK = float(cfg.get("vCruiseK", 0.90))
        self.alphaOpen = float(cfg.get("alphaOpen", 0.20))
        self.capPow = float(cfg.get("capPow", 1.150))
        self.capGain = float(cfg.get("capGain", 0.20))
        self.K_release = int(cfg.get("K_release", 3))
        self.M_boost = int(cfg.get("M_boost", 14))
        self.dvBoost = float(cfg.get("dvBoost", 1.80))
        self.thrRisk_gate = float(cfg.get("thrRisk_gate", 0.25))

        # 奖励/收尾
        self.step_cost = float(cfg.get("step_cost", 0.05))
        self.rClip = float(cfg.get("rClip", 5.0))
        self.r_arrival = float(cfg.get("r_arrival", 100.0))
        self.r_success_bonus = float(cfg.get("r_success_bonus", 350.0))
        self.r_collision_penalty = float(cfg.get("r_collision_penalty", 120.0))
        self.r_timeout_penalty_base = float(cfg.get("r_timeout_penalty_base", 120.0))

        # COLREGs 模糊判别
        self.thHeadOn = math.radians(cfg.get("thHeadOn_deg", 15.0))
        self.thCross = math.radians(cfg.get("thCross_deg", 112.5))

        # %%% PATCH-GOV-ENV (1/2): add episode outcome states %%%
        self._gov_has_collision = False         # 本集是否发生过碰撞（优先级最高）
        self._gov_reached_flags = None          # 本集各智能体是否到达（list[bool] 长度=N）
        self._gov_timed_out    = False          # 本集是否因超时截断

        self.dual_freeze = bool(cfg.get("dual_freeze", False))
        self.r_collision_boom = float(cfg.get("r_collision_boom", 1200.0))

        self.near_sticky = int(cfg.get("near_sticky", 6))   # 高风险维持步数
        self._near_cnt = np.zeros(self.N, dtype=np.int32)

        # --- Safety guard switches (train/eval) ---
        self.guard_train = bool(cfg.get("guard_train", True))   # 训练期默认开
        self.guard_eval  = bool(cfg.get("guard_eval",  False))  # 评估期默认关
        self._eval_mode  = bool(cfg.get("is_eval", False))      # 当前是否处于评估模式

        # ===== 推荐默认 =====
        self.guard_mode = str(getattr(cfg, "guard_mode", cfg.get("guard_mode", "hybrid")))
        self.guard_warmup_steps = int(getattr(cfg, "guard_warmup_steps", 80))
        self.guard_hybrid_p = float(getattr(cfg, "guard_hybrid_p", 0.2))
        self.lam_guard = float(getattr(cfg, "lam_guard", cfg.get("lam_guard", 2.0)))

        self.guard_eval = bool(cfg.get("guard_eval", False))
        self.guard_eval_mode = str(getattr(cfg, "guard_eval_mode", cfg.get("guard_eval_mode", "none")))

        self._guard_trig_sum = 0.0   # 累计每步的平均触发值
        self._guard_trig_steps = 0   # 统计参与的步数

        # 拉格朗日层
        lam0 = cfg.get("lambda_init", {"near": 3.0, "rule": 2.0, "coll": 20.0, "time": 1.0})
        self.dual = DualState(
            lam_near=float(lam0.get("near", 3.0)),
            lam_rule=float(lam0.get("rule", 2.0)),
            lam_coll=0.0,
            lam_time=float(lam0.get("time", 1.0)),
            eta_near=float(cfg.get("dual_eta", {"near": 0.02}).get("near", 0.02)),
            eta_rule=float(cfg.get("dual_eta", {"rule": 0.02}).get("rule", 0.02)),
            eta_coll=0.0,
            eta_time=float(cfg.get("dual_eta", {"time": 0.02}).get("time", 0.02)),
            beta=float(cfg.get("dual_beta", 0.95)),
            ctarget_near=float(cfg.get("ctarget", {"near": 0.05}).get("near", 0.05)),
            ctarget_rule=float(cfg.get("ctarget", {"rule": 0.02}).get("rule", 0.02)),
            ctarget_coll=0.0,
            ctarget_time=float(cfg.get("ctarget", {"time": 0.08}).get("time", 0.08)),
        )

        # 组件：采样器、RNG、状态缓存
        self.spawn_cfg = SpawnConfig(
            N=self.N, spawn_area=self.spawn_area, spawn_margin=self.spawn_margin,
            spawn_min_sep=self.spawn_min_sep, spawn_goal_min_sep=self.spawn_goal_min_sep,
            spawn_len=self.spawn_len, spawn_retry=self.spawn_retry,
            spawn_dir_jitter_deg=self.spawn_dir_jitter_deg, collide_thr=self.collide_thr,
            v_min=self.v_min, v_max=self.v_max, mode=self.spawn_mode,
        )
        self.scenario = ScenarioSampler(self.spawn_cfg)
        self.rng = RNG()

        # PettingZoo 相关
        self.agents = [f"ship_{i+1}" for i in range(self.N)]
        self.possible_agents = list(self.agents)

        # ========== AIS 相关开关与组件 ==========
        self.use_ais_obs: bool = bool(cfg.get("use_ais_obs", False))
        self.ais_cfg_path: str | None = cfg.get("ais_cfg_path", None)
        self.ais_base_seed: int = int(cfg.get("ais_base_seed", 0))

        self.ais_sim: AISCommsSim | None = None
        self.ais_obs_builder: AISObservationBuilder | None = None
        self.ais_agent_of_ship: dict[ShipId, AgentId] | None = None
        self.ais_ship_ids: list[ShipId] | None = None

        if self.use_ais_obs:
            # AIS 仿真使用独立的 numpy RNG（与 env.rng.rs 解耦）
            ais_seed = self.ais_base_seed
            rng_np = np.random.default_rng(ais_seed)
            self.ais_sim = AISCommsSim(
                rng=rng_np,
                cfg_path=self.ais_cfg_path,
                base_seed=ais_seed,
            )
            # 当前 AISObservationBuilder 只接受 slot_K、slot_ttl 两个参数
            self.ais_obs_builder = AISObservationBuilder(
                slot_K=self.ais_sim.obs_slot_K,
                slot_ttl=self.ais_sim.obs_slot_ttl,
            )

        # 观测维度（与 build_observations 完全一致）：
        # self(6) + K * F_nei(8) + K * F_edge(8) + id(1)
        self.K_neighbors = int(cfg.get("numNeighbors", 4))
        self.F_nei = 8
        self.F_edge = 8

        obs_dim = 6 + self.K_neighbors * self.F_nei + self.K_neighbors * self.F_edge + 1
        self._obs_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self._act_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 运行态
        self.state: WorldState | None = None
        self.cache = StepCache()
        self.reached = np.zeros(self.N, bool)
        self._last_phi = None
        self._guard_safeCount = np.zeros(self.N, int)
        self._guard_boostCount = np.zeros(self.N, int)

        self.enable_debug = bool(cfg.get("enable_debug", False))

        # %%% PATCH-GOV-ENV (2/2): dual governor buffers & knobs %%%
        # episode outcome滑窗 + EMA + 冷却
        self._dual_buf = []  # 收集最近W个episode的{'tout','coll','rule'}
        self._dual_ema = {"tout": None, "coll": None, "rule": None}
        self._dual_cool = 0  # 冷却计数器（>0时不更新λ）

        # 目标带（可按你的口径再微调）
        self._gov_band = {
            "tout": (0.20, 0.35),  # 期望timeout在此区间
            "coll": (0.04, 0.10),  # 期望collision在此区间
            "rule": (0.02, 0.08),  # 期望规则违规在此区间
        }
        # 对偶步长（小！去抖动）
        self._gov_eta = {"tout": 0.02, "coll": 0.02, "rule": 0.01}

        # λ 的硬限幅（与训练日志尺度匹配）
        self._gov_clamp = {
            "near": (2.0, 3.5),
            "rule": (0.8, 3.0),
            "coll": (15.0, 22.0),
            "time": (1.0, 3.0),
        }
        # 窗口/EMA/冷却
        self._gov_W = 64
        self._gov_beta = 0.9
        self._gov_cooldown = 8

    # ================== 对偶快照（供 RLlib 同步） ==================

    def get_dual_snapshot(self):
        """
        提取对偶/治理层的“纯数值快照”，保证 JSON 可序列化，且字段向后兼容。
        注意：不要返回 numpy 标量或私有对象。
        """
        d = self.dual

        def _f(x, default=None):
            try:
                v = getattr(d, x)
            except Exception:
                return default
            return None if v is None else float(v)

        snap = {
            # lambdas
            "lam_near": _f("lam_near", 3.0),
            "lam_rule": _f("lam_rule", 2.0),
            "lam_coll": _f("lam_coll", 20.0),
            "lam_time": _f("lam_time", 1.0),

            # EMA of constraints (episode-level)
            "ema_near": _f("ema_near", 0.33),
            "ema_rule": _f("ema_rule", 0.33),
            "ema_coll": _f("ema_coll", 0.0),
            "ema_time": _f("ema_time", 0.0),

            # governor EMAs of outcomes
            "ema_succ": _f("ema_succ", 0.0),
            "ema_tout": _f("ema_tout", 1.0),

            # knobs (只带必要的)
            "eta_scale": _f("eta_scale", 0.33),

            # 简单版本号，便于排查同步是否生效
            "version": int(getattr(d, "version", 0)) + 1,
        }
        return snap

    def set_dual(self, snap: dict):
        """
        用快照覆盖本地对偶状态。任何缺失字段均跳过，确保多版本兼容。
        """
        if not snap:
            return
        d = self.dual

        def _set(name, cast=float):
            if name in snap and snap[name] is not None:
                try:
                    setattr(d, name, cast(snap[name]))
                except Exception:
                    pass

        # lambdas
        _set("lam_near")
        _set("lam_rule")
        _set("lam_coll")
        _set("lam_time")

        # EMAs
        _set("ema_near")
        _set("ema_rule")
        _set("ema_coll")
        _set("ema_time")

        # governor EMAs（注意你代码里叫 ema_succ/ema_tout）
        _set("ema_succ")
        _set("ema_tout")

        # knobs
        _set("eta_scale")

        # version
        if "version" in snap:
            try:
                d.version = int(snap["version"])
            except Exception:
                d.version = getattr(d, "version", 0) + 1
        else:
            d.version = getattr(d, "version", 0) + 1

    def _time_rate_controller(self, tout_ep: float):
        """
        对 lam_time 的超时率自适应控制器：
        - 始终工作（冷却/带内/越带之后都可调），与窗口治理互补
        - 目标：把 timeout 率拉向 τ_tgt；调节幅度很小，避免干扰策略稳定性
        """
        import numpy as np

        # 缓存初始化
        if not hasattr(self, "_tout_hist"):
            self._tout_hist = []

        # 记录本集 0/1 超时标记
        self._tout_hist.append(float(tout_ep))
        W = 64
        if len(self._tout_hist) > W:
            self._tout_hist.pop(0)

        tout_rate = float(np.mean(self._tout_hist))

        # 目标和步幅（温和！）
        tau_tgt = 0.005   # 0.5% 目标超时率
        up_gain = 0.05    # 超标上调比例（一次最多 +5%）
        dn_gain = 0.02    # 低于目标下调比例（一次 -2%）
        tol_hi = 1.25     # 上容忍系数
        tol_lo = 0.75     # 下容忍系数

        lo, hi = self._gov_clamp["time"]  # 复用你已有的限幅配置
        lam_t = float(self.dual.lam_time)

        if tout_rate > tau_tgt * tol_hi:
            lam_t = lam_t * (1.0 + up_gain)
        elif tout_rate < tau_tgt * tol_lo:
            lam_t = lam_t * (1.0 - dn_gain)

        self.dual.lam_time = float(np.clip(lam_t, lo, hi))

    def _dual_window_govern(self, succ_ep: float, coll_ep: float, tout_ep: float,
                            rule_rate_ep: float, lam_prev: dict[str, float]):
        """
        Episode级治理：
        - 维护滑窗&EMA（tout/coll/rule）
        - 在目标带内：near/rule/coll 回滚+进入冷却，但不回滚 time；
          同时总是运行“超时率自适应控制器”微调 lam_time
        - 越带：按小步从 lam_prev 微调（含 time），之后仍运行“超时率控制器”作细修
        """
        import numpy as np

        # ----------(A) 滑窗 & EMA 维护 ----------
        entry = {
            "tout": float(tout_ep),
            "coll": float(coll_ep),
            "rule": float(rule_rate_ep),
        }
        self._dual_buf.append(entry)
        if len(self._dual_buf) > self._gov_W:
            self._dual_buf.pop(0)

        # 计算滑窗均值
        wins = {
            k: float(np.mean([x[k] for x in self._dual_buf]))
            for k in ["tout", "coll", "rule"]
        }

        # EMA 更新
        for k in wins:
            m = self._dual_ema[k]
            self._dual_ema[k] = (
                self._gov_beta * m + (1.0 - self._gov_beta) * wins[k]
            ) if m is not None else wins[k]

        ema = self._dual_ema
        band = self._gov_band

        # ----------(B) 冷却期：near夹持 + lam_time控制器，立即返回 ----------
        if self._dual_cool > 0:
            self._dual_cool -= 1
            # near 仅做限幅防漂移
            self.dual.lam_near = float(
                np.clip(self.dual.lam_near, *self._gov_clamp["near"])
            )
            # 在冷却期也运行 lam_time 自适应（可细调 timeout）
            self._time_rate_controller(tout_ep)
            return "cooldown"

        # ----------(C) 是否越带 ----------
        flagged = []
        for k, (lo, hi) in band.items():
            v = ema[k]
            if v < lo * 0.98:
                flagged.append((k, -1, (lo - v) / max(1e-6, lo)))
            elif v > hi * 1.02:
                flagged.append((k, +1, (v - hi) / max(1e-6, hi)))

        # ----------(D) 带内：near/rule/coll 回滚 + 冷却；time 不回滚 ----------
        if not flagged:
            self.dual.lam_near = lam_prev["near"]
            self.dual.lam_rule = lam_prev["rule"]
            self.dual.lam_coll = lam_prev["coll"]
            # 不回滚 lam_time，保持当前值以便持续收敛 timeout
            self._dual_cool = self._gov_cooldown

            # 仍然运行一次 lam_time 自适应微调（细化 timeout 率）
            self._time_rate_controller(tout_ep)
            return "freeze"

        # ----------(E) 越带：从 lam_prev 小步微调 ----------
        def _step(name_lambda: str, k_key: str, sgn: int, mag: float):
            step = self._gov_eta[k_key] * float(np.clip(mag, 0.0, 1.0))
            newv = lam_prev[name_lambda] + sgn * step
            lo, hi = self._gov_clamp[name_lambda]
            return float(np.clip(newv, lo, hi))

        for k_key, sgn, mag in flagged:
            if k_key == "tout":
                self.dual.lam_time = _step("time", "tout", sgn, mag)
            elif k_key == "coll":
                self.dual.lam_coll = _step("coll", "coll", sgn, mag)
            elif k_key == "rule":
                self.dual.lam_rule = _step("rule", "rule", sgn, mag)

        # near 仍只做限幅（不主动调）
        self.dual.lam_near = float(
            np.clip(lam_prev["near"], *self._gov_clamp["near"])
        )

        # ----------(F) 越带后也运行一次 lam_time 自适应微调 ----------
        self._time_rate_controller(tout_ep)
        return "nudged"

    # ================== PettingZoo 接口 ==================

    @property
    def observation_spaces(self):
        return {aid: self._obs_space for aid in self.agents}

    @property
    def action_spaces(self):
        return {aid: self._act_space for aid in self.agents}

    def seed(self, seed: int | None = None):
        self.rng.seed(seed)

    # --------- 辅助：Ship -> TrueState（供 AIS 仿真使用） ---------

    def _build_true_states(self) -> dict[ShipId, TrueState]:
        """
        将当前 WorldState 中的 Ship 列表转换为 AISCommsSim 使用的 TrueState 字典。
        """
        states: dict[ShipId, TrueState] = {}
        for i, ship in enumerate(self.state.ships):
            sid: ShipId = int(ship.id)  # Ship 的 id 已经是 1..N
            x, y = float(ship.pos[0]), float(ship.pos[1])
            v = float(ship.v)
            psi = float(ship.psi)
            vx = v * math.cos(psi)
            vy = v * math.sin(psi)
            states[sid] = TrueState(
                sid,
                x=x,
                y=y,
                vx=vx,
                vy=vy,
            )
        return states

    def _build_own_true_for_agents(self, true_states: dict[ShipId, TrueState]) -> dict[AgentId, TrueState]:
        """
        构建 agent_id -> TrueState 的映射，供 AISObservationBuilder 使用。
        """
        own_true: dict[AgentId, TrueState] = {}
        for i, ship in enumerate(self.state.ships):
            sid: ShipId = int(ship.id)
            aid: AgentId = self.agents[i]
            own_true[aid] = true_states[sid]
        return own_true

    # ---------------------- reset ----------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed(seed)

        if options is not None and "eval" in options:
            self._eval_mode = bool(options.get("eval"))

        self._guard_safeCount[:] = 0
        self._guard_boostCount[:] = 0
        self._guard_trig_sum = 0.0
        self._guard_trig_steps = 0

        starts, goals, psi0, v0 = self.scenario.sample(self.rng.rs)
        ships = [
            Ship(i + 1, starts[i].copy(), goals[i].copy(), float(psi0[i]), float(v0[i]))
            for i in range(self.N)
        ]
        self.reached[:] = False
        self.state = WorldState(ships=ships, t=0.0, step=0)
        self._last_phi = progress_phi(ships)

        infos = {aid: {"t": 0.0, "step": 0} for aid in self.agents}

        # ====== 若启用 AIS 观测：在 reset 中同步初始化 AIS 仿真 ======
        if self.use_ais_obs and self.ais_sim is not None and self.ais_obs_builder is not None:
            self.ais_ship_ids = [int(s.id) for s in self.state.ships]
            self.ais_agent_of_ship = {
                int(self.state.ships[i].id): self.agents[i]
                for i in range(self.N)
            }

            # AISCommsSim reset
            self.ais_sim.reset(
                ships=self.ais_ship_ids,
                t0=float(self.state.t),
                agent_map=self.ais_agent_of_ship,
            )

            # AISObservationBuilder reset
            self.ais_obs_builder.reset(agent_ids=self.agents, t0=float(self.state.t))

            # 构造 TrueState 并推进 AIS 一步（生成初始 ready 报文）
            true_states = self._build_true_states()
            ready = self.ais_sim.step(float(self.state.t), true_states)
            own_true = self._build_own_true_for_agents(true_states)

            obs = self.ais_obs_builder.build(
                ready_msgs=ready,
                t=float(self.state.t),
                own_true=own_true,
            )
        else:
            # 否则使用几何观测构建
            obs = build_observations(
                self.state.ships,
                self.K_neighbors,
                self.spawn_mode,
                self.spawn_area,
                self.v_max,
            )

        return obs, infos

    # ---------------------- step ----------------------

    def step(self, actions: dict):
        # A) 动作解码/映射
        A = decode_actions(actions, self._act_space, self.agents)

        # --- 记录真正的 pre 快照（推进前） ---
        _pre_snap = {
            "pos": np.stack([s.pos.copy() for s in self.state.ships]).tolist(),
            "psi": [float(s.psi) for s in self.state.ships],
            "v":   [float(s.v) for s in self.state.ships],
            "goal": np.stack([s.goal.copy() for s in self.state.ships]).tolist(),
            "reached": [bool(s.reached) for s in self.state.ships],
            "guard_boost_before": self._guard_boostCount.copy().tolist(),
            "guard_safe_before":  self._guard_safeCount.copy().tolist(),
        }

        dpsi_rl, v_cmd = map_to_commands(A, self.dpsi_max, self.v_min, self.v_max)

        # B) 推进前风险（先算 tc0/dc0）
        self.cache.tc0, self.cache.dc0, _ = tcpa_dcpa_matrix(self.state.ships)

        # 缓存当前状态（避免函数内重复stack/array）
        self.cache.pos = np.stack([s.pos for s in self.state.ships], axis=0)
        self.cache.psi = np.array([s.psi for s in self.state.ships])
        self.cache.v_all = np.array([s.v for s in self.state.ships])

        # B0) 先护栏（严格投影到安全集），再做推进前速度软限
        _use_guard = (self.guard_eval if self._eval_mode else self.guard_train)

        # 确保后续奖励里可用
        _guard_trig = np.zeros(self.N, dtype=np.float64)

        if _use_guard:
            guard_kwargs = dict(
                ships=self.state.ships,
                dpsi=dpsi_rl,
                v_cmd=v_cmd,
                dt=self.dt,
                dpsi_max=self.dpsi_max,
                v_min=self.v_min,
                v_max=self.v_max,
                risk_T_thr=self.risk_T_thr,
                risk_D_thr=self.risk_D_thr,
                collide_thr=self.collide_thr,
                tc=self.cache.tc0,
                dc=self.cache.dc0,
                yaw_rate_max=None,
                brake_gain=0.4,
                steer_gain=0.4,
                K_guard=4,
                use_float32=True,
            )

            if self._eval_mode:
                # —— 评估期：根据 guard_eval + guard_eval_mode 决定是否/如何应用护栏
                if not self.guard_eval or self.guard_eval_mode == "none":
                    _guard_trig = np.zeros(self.N, dtype=np.float64)
                else:
                    mode = self.guard_eval_mode.lower()
                    if mode == "ghost":
                        dpsi_rl, v_cmd, _guard_trig, _ = safety_guard_ghost(**guard_kwargs)
                    elif mode == "hard":
                        dpsi_rl, v_cmd, _guard_trig, _ = safety_guard_hard(**guard_kwargs)
                    elif mode == "strict":
                        # 如果你有 strict_guard_multi 就用它；否则删掉这个分支
                        # dpsi_rl, v_cmd, _guard_trig = strict_guard_multi(**guard_kwargs)
                        dpsi_rl, v_cmd, _guard_trig = dpsi_rl, v_cmd, np.zeros(self.N, dtype=np.float64)
                    else:
                        _guard_trig = np.zeros(self.N, dtype=np.float64)
            else:
                # —— 训练期：沿用你原来的 hard/ghost/hybrid 逻辑
                mode = getattr(self, "guard_mode", "hard")
                warmup = getattr(self, "guard_warmup_steps", 100)
                hybrid_p = getattr(self, "guard_hybrid_p", 0.5)
                if mode == "hard":
                    dpsi_rl, v_cmd, _guard_trig, _ = safety_guard_hard(**guard_kwargs)
                elif mode == "ghost":
                    dpsi_rl, v_cmd, _guard_trig, _ = safety_guard_ghost(**guard_kwargs)
                elif mode == "hybrid":
                    if self.state.step < warmup:
                        dpsi_rl, v_cmd, _guard_trig, _ = safety_guard_hard(**guard_kwargs)
                    else:
                        dpsi_rl, v_cmd, _guard_trig, _ = safety_guard_hybrid(
                            **guard_kwargs,
                            mode_prob=float(hybrid_p),
                        )
                else:
                    _guard_trig = np.zeros(self.N, dtype=np.float64)
        else:
            _guard_trig = np.zeros(self.N, dtype=np.float64)  # 没启护栏时别忘了赋零

        # --- 护栏调用结束 ---

        # 记录本步平均触发（对N条船取均值，更稳）
        _step_guard_mean = float(np.mean(_guard_trig)) if _guard_trig is not None else 0.0
        self._guard_trig_sum += _step_guard_mean
        self._guard_trig_steps += 1

        # 现在再做推进前速度软限，v_target 会体现场内刹车后的 v_cmd
        v_target, v_cap, bestX0 = speed_soft_cap_before_step(
            self.state.ships,
            self.cache.tc0,
            self.cache.dc0,
            self.v_max,
            self.risk_T_thr,
            self.risk_D_thr,
            self.thrRisk_gate + 0.04,   # ← 放松风险门
            self.vCruiseK,
            self.alphaOpen,
            self.capPow,
            0.9 * self.capGain,         # ← 降低强度
            self.K_release + 6,
            self.M_boost + 1,
            1.35,                        # ← 更快恢复
            self._guard_safeCount,
            self._guard_boostCount,
            v_cmd.copy(),
        )

        # C) 物理推进（含 dvBoost）
        prev_v = np.array([s.v for s in self.state.ships], dtype=np.float64)
        dv_max_eff_used = []  # 本步每条船实际使用的 dv_max（含 boost）

        for i, ship in enumerate(self.state.ships):
            if ship.reached:
                dv_max_eff_used.append(float(self.dv_max))  # 无所谓，用普通值占位
                continue
            boost = self._guard_boostCount[i] > 0
            dv_max_eff = self.dv_max * (self.dvBoost if boost else 1.0)
            dv_max_eff_used.append(float(dv_max_eff))  # 记录“本步实际用到的值”
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

        # D) 到达判定 + 到达奖励
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

        # E) 推进后风险
        self.cache.tc, self.cache.dc, _ = tcpa_dcpa_matrix(self.state.ships)
        self.cache.risk, self.cache.dmin_i, self.cache.vj_max = post_step_risk(
            self.state.ships,
            self.cache.tc,
            self.cache.dc,
            self.risk_T_thr,
            self.risk_D_thr,
        )

        # F) 奖励/成本
        r_task, self._last_phi = task_reward(
            self.state.ships,
            dpsi_rl,
            self._last_phi,
            self.cache.risk,
            self.v_max,
            self.step_cost,
            arrival_bonus,
        )
        v_self = np.array([s.v for s in self.state.ships], dtype=np.float64)
        c_near = cost_near(
            self.cache.risk,
            self.cache.vj_max,
            v_self,
            self.cache.dmin_i,
            self.collide_thr,
            tau=6.0,
        )

        # 粘性：若本步 risk 超过门限 (如 cache.risk>0.35)，near计数+1，否则衰减
        risk_gate = 0.38                 # ↑ 0.35 -> 0.38，少触发粘性
        over = (self.cache.risk > risk_gate).astype(np.int32)
        self._near_cnt = np.clip(self._near_cnt + over - (1 - over), 0, self.near_sticky)

        # 原：1 + 0.2*(cnt/M)  ->  现：1 + 0.1*(cnt/M)，且封顶 1.12
        stick_gain = 1.0 + 0.10 * (self._near_cnt.astype(np.float64) / max(1, self.near_sticky))
        stick_gain = np.minimum(stick_gain, 1.12)
        c_near = c_near * stick_gain

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

        # G) 合成奖励（拉格朗日）
        r_vec = combine_reward(
            r_task,
            c_near,
            c_rule,
            c_coll,
            c_time,
            self.dual,
            clip=self.rClip,
        )

        lam_guard = float(getattr(self, "lam_guard", 2.0))
        if lam_guard > 0.0 and _use_guard:
            r_vec = r_vec - float(lam_guard) * _guard_trig.astype(np.float64)

        if np.any(c_coll > 0.5):          # 本步检测到任意碰撞对儿
            r_vec = r_vec - self.r_collision_boom

        # H) 终止与对偶更新 + Governor
        success_all = bool(np.all(self.reached))
        done_global = coll or success_all or timeout
        term_code = 0
        if done_global:
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

            coll_ep = 1.0 if coll else 0.0
            succ_ep = 1.0 if (not coll and success_all) else 0.0
            tout_ep = 1.0 if (not coll and not success_all and timeout) else 0.0

            if not self.dual_freeze:
                lam_prev = {
                    "near": float(self.dual.lam_near),
                    "rule": float(self.dual.lam_rule),
                    "coll": float(self.dual.lam_coll),
                    "time": float(self.dual.lam_time),
                }

                dual_update_end_of_episode(
                    float(np.mean(c_near)),
                    float(np.mean(c_rule)),
                    float(np.max(c_coll)),
                    float(np.max(c_time)),
                    self.dual,
                )

                # 这里 rule_rate_ep 用“本集平均 c_rule 是否>0”近似
                rule_rate_ep = 1.0 if float(np.mean(c_rule)) > 0.0 else 0.0

                _ = self._dual_window_govern(
                    succ_ep,
                    coll_ep,
                    tout_ep,
                    rule_rate_ep,
                    lam_prev,
                )

            # 再根据优先级标注 succ/tout（coll_ep 已经确定）
            if coll_ep < 0.5:
                if success_all:
                    succ_ep = 1.0
                elif timeout:
                    tout_ep = 1.0

        # I) 封装返回
        rewards = {aid: float(r_vec[i]) for i, aid in enumerate(self.agents)}
        terminations = {aid: False for aid in self.agents}
        truncations = {aid: False for aid in self.agents}

        infos = {aid: {"t": self.state.t, "step": self.state.step} for aid in self.agents}
        for i, aid in enumerate(self.agents):
            infos[aid]["guard_enabled"] = bool(_use_guard)
            infos[aid]["guard_trig_step_mean"] = _step_guard_mean

        # 统一的终止标注（整局原因一致）
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

            # 整局护栏触发率统计
            ep_guard_rate = self._guard_trig_sum / max(1, self._guard_trig_steps)
            for aid in self.agents:
                infos[aid]["guard_trig_rate_ep"] = float(ep_guard_rate)

        # ========== 构建下一步观测 ==========
        if done_global:
            # 终局：返回零观测
            observations = {
                aid: zero_observation(self._obs_space)
                for aid in self.agents
            }
        else:
            if self.use_ais_obs and self.ais_sim is not None and self.ais_obs_builder is not None:
                # --- AIS 模式：用真实 AIS 仿真输出 noisy 观测 ---
                true_states = self._build_true_states()
                ready = self.ais_sim.step(float(self.state.t), true_states)
                own_true = self._build_own_true_for_agents(true_states)
                observations = self.ais_obs_builder.build(
                    ready_msgs=ready,
                    t=float(self.state.t),
                    own_true=own_true,
                )
            else:
                # --- 几何模式 ---
                observations = build_observations(
                    self.state.ships,
                    self.K_neighbors,
                    self.spawn_mode,
                    self.spawn_area,
                    self.v_max,
                )

        # ===== 把拉格朗日乘子写入 infos（每步）=====
        if hasattr(self, "dual"):
            ln = float(self.dual.lam_near)
            lr = float(self.dual.lam_rule)
            lc = 0.0
            lt = float(self.dual.lam_time)
            for aid in self.agents:
                infos[aid]["lambda_near"] = ln
                infos[aid]["lambda_rule"] = lr
                infos[aid]["lambda_coll"] = lc
                infos[aid]["lambda_time"] = lt

        return observations, rewards, terminations, truncations, infos

    # ========== 其他接口 ==========
    def render(self):
        return None

    def close(self):
        return None

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]
