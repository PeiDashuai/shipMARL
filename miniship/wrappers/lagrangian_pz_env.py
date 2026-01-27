# miniship/wrappers/lagrangian_pz_env.py

from __future__ import annotations
from typing import Dict, Any
import numpy as np
from pettingzoo import ParallelEnv

from miniship.core import miniship_core_env
from miniship.reward.dual_manager import DualManager
from miniship.risk.guard_controller import GuardController
from ..observe.ais_obs_wrapper import AISObsWrapper


class MiniShipLagrangianParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["none"]}

    def __init__(self, cfg: Dict[str, Any]):
        """
        在 MiniShipCoreEnv 外套一层拉格朗日包装：

        - 底层物理/风险/终止逻辑由 MiniShipCoreEnv 负责
        - 可选：在 CoreEnv 外再套一层 AISObsWrapper，将观测改为 AIS 格式
        - 本 wrapper 只做：
            * 使用 DualManager 合成拉格朗日奖励
            * episode 结束时更新 λ
            * 把当前 λ 写进 infos
        """
        # === 新增：总开关 ===
        self.use_lagrangian: bool = bool(cfg.get("use_lagrangian", True))


        # === NEW: RLlib/Callback 对齐开关 ===
        # 若你在 RLlib callbacks 里做 reward shaping（推荐做法），这里应默认不在 env 内再 shaping，
        # 否则会出现“双重扣罚”。需要 env 内 shaping 时再显式打开。
        self.lagrangian_reward_in_env: bool = bool(cfg.get("lagrangian_reward_in_env", False))
        # 对偶更新是否在 env 内执行（多数情况下你希望 driver 侧统一更新，因此默认 False）
        self.dual_update_in_env: bool = bool(cfg.get("dual_update_in_env", False))


        # 1) 创建底层核心环境（几何真值 + 护栏）
        use_guard = bool(cfg.get("use_guard", True))
        guard_ctrl = GuardController(cfg, N=int(cfg.get("N", 2))) if use_guard else None
        self.core: ParallelEnv = miniship_core_env.MiniShipCoreEnv(cfg, guard_ctrl=guard_ctrl)

        # 2) 是否使用 AIS noisy 观测（从 cfg 中读取开关）
        self.use_ais_obs = bool(cfg.get("use_ais_obs", False))
        print("[MiniShipLagrangianParallelEnv] use_ais_obs =", self.use_ais_obs)

        if self.use_ais_obs:
            # AISObsWrapper 只改观测，不改 reward/infos/终止逻辑
            self.env: ParallelEnv = AISObsWrapper(self.core, cfg)
        else:
            # 不启 AIS 时，直接暴露 core（几何观测）
            self.env = self.core

        # 3) 对偶管理器（用同一份 cfg）
        self.dual_mgr = DualManager(cfg)
        # NEW: 本 wrapper 自身也维护一个版本号，便于日志对齐（driver 为准，env 只是镜像）
        self._dual_version: int = 0


        # 4) 一些超参数（与老大 env 中保持一致）
        self.rClip = float(cfg.get("rClip", 5.0))
        #self.rClip = None
        self.dual_freeze = bool(cfg.get("dual_freeze", False))

        # 5) 直接引用当前 env 的 agent 列表和空间（可能是 core，也可能是 AIS 包装后的 env）
        self.agents = list(self.env.agents)
        self.possible_agents = list(self.env.possible_agents)
        self._obs_space = self.env.observation_space(self.agents[0])
        self._act_space = self.env.action_space(self.agents[0])

        # 元信息沿用当前 env
        self.metadata = getattr(self.env, "metadata", {"render_modes": ["none"]})
        # Episode-level accumulators (used for dual updates + episode metrics).
        self._reset_ep_stats()

        # NEW: 支持通过 env_config 注入 driver snapshot（尤其用于 evaluation_config）
        boot_snap = cfg.get("dual_snapshot", None)
        if isinstance(boot_snap, dict):
            self.set_dual(boot_snap)

    # ---------- PettingZoo 标准接口转发 ----------

    @property
    def observation_spaces(self):
        # PettingZoo 并行 env 约定：dict[agent] -> space
        # 注意：这里用 self.env（可能已经包了 AIS），不是 self.core
        return self.env.observation_spaces

    @property
    def action_spaces(self):
        return self.env.action_spaces

    def action_space(self, agent):
        return self.env.action_space(agent)

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def seed(self, seed: int | None = None):
        if hasattr(self.env, "seed"):
            self.env.seed(seed)

    def _reset_ep_stats(self) -> None:
        # These are episode-level accumulators used for:
        #  (1) dual updates (which must be based on episode metrics, not a single terminal step), and
        #  (2) consistent episode logging via infos keys.
        self._ep_steps: int = 0
        self._ep_c_near_sum: float = 0.0
        self._ep_c_rule_sum: float = 0.0
        self._ep_c_coll_max: float = 0.0
        self._ep_c_time_max: float = 0.0

    # =========================
    # Dual snapshot I/O (for RLlib driver sync)
    # =========================
    def get_dual_snapshot(self) -> Dict[str, Any]:
        """
        给 driver / 训练脚本读取的快照接口。
        约定尽量对齐 MiniShipCallbacks.get_dual_snapshot():
          {
            "dual_version": int,
            "lam": {"near":..., "rule":...},
            "running_c": {...}   # 若 DualManager 支持则附带
          }
        """
        snap: Dict[str, Any] = {"dual_version": int(self._dual_version)}
        if self.dual_mgr is None:
            snap["lam"] = {}
            snap["running_c"] = {}
            return snap

        # lambdas
        lam = {}
        try:
            if hasattr(self.dual_mgr, "get_lambdas") and callable(getattr(self.dual_mgr, "get_lambdas")):
                lam = dict(self.dual_mgr.get_lambdas() or {})
            elif hasattr(self.dual_mgr, "lambdas"):
                lam = dict(getattr(self.dual_mgr, "lambdas") or {})
            elif hasattr(self.dual_mgr, "lam"):
                lam = dict(getattr(self.dual_mgr, "lam") or {})
        except Exception:
            lam = {}
        snap["lam"] = {k: float(v) for k, v in lam.items()}

        # running_c (optional)
        rc = {}
        try:
            if hasattr(self.dual_mgr, "get_running_c") and callable(getattr(self.dual_mgr, "get_running_c")):
                rc = dict(self.dual_mgr.get_running_c() or {})
            elif hasattr(self.dual_mgr, "running_c"):
                rc = dict(getattr(self.dual_mgr, "running_c") or {})
            elif hasattr(self.dual_mgr, "c_ema"):
                rc = dict(getattr(self.dual_mgr, "c_ema") or {})
        except Exception:
            rc = {}
        snap["running_c"] = {k: float(v) for k, v in rc.items()}

        return snap

    def set_dual(self, snap: Dict[str, Any]) -> None:
        """
        接收 driver 推送的对偶快照，并写入 dual_mgr（env 侧只做镜像，不做“自主”更新）。
        兼容字段名：
          - lam / lambdas / lambda
          - dual_version
        """
        if not isinstance(snap, dict) or self.dual_mgr is None:
            return

        if "dual_version" in snap:
            try:
                self._dual_version = int(snap["dual_version"])
            except Exception:
                pass

        lam_in = snap.get("lam", None)
        if lam_in is None:
            lam_in = snap.get("lambdas", None)
        if lam_in is None:
            lam_in = snap.get("lambda", None)

        if isinstance(lam_in, dict):
            # 优先调用 DualManager 提供的 setter（若存在）
            try:
                if hasattr(self.dual_mgr, "set_lambdas") and callable(getattr(self.dual_mgr, "set_lambdas")):
                    self.dual_mgr.set_lambdas(lam_in)
                    return
                if hasattr(self.dual_mgr, "set_lambda") and callable(getattr(self.dual_mgr, "set_lambda")):
                    for k, v in lam_in.items():
                        self.dual_mgr.set_lambda(k, float(v))
                    return
                if hasattr(self.dual_mgr, "set_snapshot") and callable(getattr(self.dual_mgr, "set_snapshot")):
                    self.dual_mgr.set_snapshot(snap)
                    return
            except Exception:
                pass

            # 兜底：直接写属性
            for attr in ("lambdas", "lam"):
                if hasattr(self.dual_mgr, attr):
                    try:
                        d = getattr(self.dual_mgr, attr)
                        if isinstance(d, dict):
                            for k, v in lam_in.items():
                                d[k] = float(v)
                            setattr(self.dual_mgr, attr, d)
                            break
                    except Exception:
                        pass


    @staticmethod
    def _all_agents_done(d: dict) -> bool:
        # PettingZoo parallel envs do not necessarily provide "__all__".
        vals = [v for k, v in d.items() if k != "__all__"]
        return bool(vals) and all(bool(v) for v in vals)

    @staticmethod
    def _infer_term_reason(infos: dict) -> str | None:
        # Prefer the canonical key from the core env.
        for info in infos.values():
            reason = info.get("term_reason")
            if reason:
                return reason

        # Fallback for older upstream implementations.
        if any(info.get("collision", False) or info.get("term_coll", False) for info in infos.values()):
            return "collision"
        if any(info.get("success", False) or info.get("term_succ", False) for info in infos.values()):
            return "success"
        if any(info.get("timeout", False) or info.get("term_tout", False) for info in infos.values()):
            return "timeout"

        return None


    # ---------------------- reset ----------------------

    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self._reset_ep_stats()

        # PettingZoo 并行 env：reset 后 agents 可能刷新
        try:
            self.agents = list(self.env.agents)
        except Exception:
            pass

        if not self.use_lagrangian:
            return obs, infos

        # FIX: 移除未定义的 cfg/self.dual_cfg 链路；DualManager 已在 __init__ 创建
        try:
            if self.dual_mgr is not None and hasattr(self.dual_mgr, "reset") and callable(getattr(self.dual_mgr, "reset")):
                self.dual_mgr.reset()
        except Exception:
            pass

        return obs, infos


    # ---------------------- step ----------------------

    def step(self, actions):
        obs, base_rewards, terminations, truncations, infos = self.env.step(actions)

        # Ensure per-agent infos exist for all agents we are returning.
        agent_ids = list(base_rewards.keys()) if isinstance(base_rewards, dict) else []
        if not agent_ids:
            agent_ids = list(infos.keys())
        for aid in agent_ids:
            infos.setdefault(aid, {})

        # =========================
        # 1) Per-step costs (from core env) + episode accumulators
        # =========================
        c_near = np.array([float(infos[aid].get("c_near", 0.0)) for aid in agent_ids], dtype=np.float32)
        c_rule = np.array([float(infos[aid].get("c_rule", 0.0)) for aid in agent_ids], dtype=np.float32)
        c_coll = np.array([float(infos[aid].get("c_coll", 0.0)) for aid in agent_ids], dtype=np.float32)
        c_time = np.array([float(infos[aid].get("c_time", 0.0)) for aid in agent_ids], dtype=np.float32)

        step_c_near_mean = float(np.mean(c_near)) if c_near.size else 0.0
        step_c_rule_mean = float(np.mean(c_rule)) if c_rule.size else 0.0
        step_c_coll_max = float(np.max(c_coll)) if c_coll.size else 0.0
        step_c_time_max = float(np.max(c_time)) if c_time.size else 0.0

        self._ep_steps += 1
        self._ep_c_near_sum += step_c_near_mean
        self._ep_c_rule_sum += step_c_rule_mean
        self._ep_c_coll_max = max(self._ep_c_coll_max, step_c_coll_max)
        self._ep_c_time_max = max(self._ep_c_time_max, step_c_time_max)

        # =========================
        # 2) Reward shaping + per-step infos keys (unified)
        # =========================
        # 默认：RLlib 回调里做 shaping，因此 env 内不再改 reward（避免双重扣罚）。
        rewards = base_rewards
        lambdas = {}
        try:
            if self.dual_mgr is not None and hasattr(self.dual_mgr, "get_lambdas") and callable(getattr(self.dual_mgr, "get_lambdas")):
                lambdas = dict(self.dual_mgr.get_lambdas() or {})
            elif self.dual_mgr is not None and hasattr(self.dual_mgr, "lambdas"):
                lambdas = dict(getattr(self.dual_mgr, "lambdas") or {})
            elif self.dual_mgr is not None and hasattr(self.dual_mgr, "lam"):
                lambdas = dict(getattr(self.dual_mgr, "lam") or {})
        except Exception:
            lambdas = {}

        # 若显式打开 env 内 shaping，则这里才执行 combine_reward 并覆盖 rewards
        if self.use_lagrangian and (self.dual_mgr is not None) and self.lagrangian_reward_in_env:
            r_vec = np.array([float(base_rewards.get(aid, 0.0)) for aid in agent_ids], dtype=np.float32)
            try:
                r_shaped_vec = self.dual_mgr.combine_reward(
                    r_vec=r_vec, c_near=c_near, c_rule=c_rule, c_coll=c_coll, c_time=c_time
                )
            except Exception:
                r_shaped_vec = r_vec

            rewards = {}
            for i, aid in enumerate(agent_ids):
                r_task = float(base_rewards.get(aid, 0.0))
                r_shaped = float(r_shaped_vec[i])
                # Optional clip
                if self.rClip is not None:
                    try:
                        r_shaped = float(np.clip(r_shaped, -float(self.rClip), float(self.rClip)))
                    except Exception:
                        pass
                r_lag = r_shaped - r_task
                infos[aid].update(
                    dict(
                        r_task=r_task,
                        r_lag=r_lag,
                        r_shaped=r_shaped,
                        r_total=r_shaped,
                        lambda_near=float(lambdas.get("near", 0.0)),
                        lambda_rule=float(lambdas.get("rule", 0.0)),
                        lambda_coll=float(lambdas.get("coll", 0.0)),
                        lambda_time=float(lambdas.get("time", 0.0)),
                        dual_version=int(self._dual_version),
                    )
                )
                rewards[aid] = r_shaped
        else:
            # 不在 env 内 shaping：reward 保持 base_rewards，但 infos 仍暴露 λ（用于对齐/日志）
            for aid in agent_ids:
                r_task = float(base_rewards.get(aid, 0.0))
                infos[aid].update(
                    dict(
                        r_task=r_task,
                        r_lag=0.0,
                        r_shaped=r_task,
                        r_total=r_task,
                        lambda_near=float(lambdas.get("near", 0.0)),
                        lambda_rule=float(lambdas.get("rule", 0.0)),
                        lambda_coll=float(lambdas.get("coll", 0.0)),
                        lambda_time=float(lambdas.get("time", 0.0)),
                        dual_version=int(self._dual_version),
                    )
                )
        # =========================
        # 3) Episode-end stats (FIXED) + infos keys unification (FIXED)
        # =========================
        done_all = bool(terminations.get("__all__", False) or truncations.get("__all__", False))
        if not done_all:
            done_all = self._all_agents_done(terminations) or self._all_agents_done(truncations)

        if done_all:
            denom = max(1, int(self._ep_steps))
            c_near_mean = float(self._ep_c_near_sum) / denom
            c_rule_mean = float(self._ep_c_rule_sum) / denom
            c_coll_max = float(self._ep_c_coll_max)
            c_time_max = float(self._ep_c_time_max)

            # Validity: if upstream provides term_valid, the whole-episode validity should be an AND over agents.
            term_valid = all(bool(info.get("term_valid", True)) for info in infos.values())

            reason = self._infer_term_reason(infos)
            succ_ep = int(reason == "success")
            coll_ep = int(reason == "collision")
            tout_ep = int(reason == "timeout")
            other_ep = int((succ_ep + coll_ep + tout_ep) == 0)

            # Dual update MUST use episode metrics (mean/max over the episode), not the terminal-step costs.
            # NEW: 默认不在 env 内更新对偶；如需启用则 dual_update_in_env=True 且未 freeze
            if self.use_lagrangian and (self.dual_mgr is not None) and self.dual_update_in_env and (not self.dual_freeze):

                self.dual_mgr.on_episode_end(
                    c_near_mean=c_near_mean,
                    c_rule_mean=c_rule_mean,
                    c_coll_max=c_coll_max,
                    c_time_max=c_time_max,
                    term_valid=term_valid,
                    succ_ep=bool(succ_ep),
                    coll_ep=bool(coll_ep),
                    tout_ep=bool(tout_ep),
                )

            for aid in agent_ids:
                infos[aid].update(
                    dict(
                        # Unified episode stats (what callbacks/loggers should use).
                        termstats_valid=int(term_valid),
                        succ_ep=int(succ_ep),
                        coll_ep=int(coll_ep),
                        tout_ep=int(tout_ep),
                        other_ep=int(other_ep),
                        # Common aliases (keep both to avoid downstream breakage).
                        succ_ep_bin=int(succ_ep),
                        coll_ep_bin=int(coll_ep),
                        tout_ep_bin=int(tout_ep),
                        term_reason=reason if reason is not None else infos[aid].get("term_reason", None),
                        success=bool(succ_ep),
                        collision=bool(coll_ep),
                        timeout=bool(tout_ep),
                        term_succ=bool(succ_ep),
                        term_coll=bool(coll_ep),
                        term_tout=bool(tout_ep),
                        # Episode-level costs (for inspection + logging).
                        c_near_mean=c_near_mean,
                        c_rule_mean=c_rule_mean,
                        c_coll_max=c_coll_max,
                        c_time_max=c_time_max,
                    )
                )

        return obs, rewards, terminations, truncations, infos

    # ---------------------- 其他接口转发 ----------------------

    def render(self):
        if hasattr(self.env, "render"):
            return self.env.render()
        return None

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
