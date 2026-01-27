"""
_ais_comms_env_impl.py — MiniShip AIS+PF+Staging Environment Implementation

This module implements the full integration of:
  1. Core physics/RL environment (MiniShipLagrangianParallelEnv)
  2. AIS communication simulation (AISCommsSim)
  3. Per-agent PF tracking (AISTrackManagerPF)
  4. Comprehensive staging recording for analysis

Data Flow:
  reset():
    - Core env reset → ship initial states
    - AIS reset → episode params sampled
    - PF reset → trackers initialized
    - Stage3: episode_init record (env_config, ais_params, ship_init_states)
    - Stage4: episode_start event

  step():
    - Core env step → obs, rewards, terminations
    - AIS step → communication simulation
    - PF ingest → tracking updates
    - Stage3: step record (ship_states, rl_data, comm_stats, pf_estimates)
    - Stage4: events (collision, arrival, episode_end)

Analysis Support:
  All data is joinable by (run_uuid, episode_uid, step_idx) enabling:
    - Episode-level analysis: AIS params → PF accuracy → RL performance
    - Step-level analysis: communication delay → tracking error → reward
    - Multi-worker alignment: deterministic episode_uid across shards
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from ais_comms.ais_comms import AISCommsSim
from ais_comms.datatypes import AgentId, ShipId, TrueState
from ais_comms.track_manager_pf import AISTrackManagerPF

from miniship.envs._ais_identity import EpisodeIdAllocator, EpisodeIdentity, RunIdentity
from miniship.envs._ais_staging import (
    StagingSink,
    build_ship_state_dict,
    build_rl_data_dict,
    build_pf_estimate_dict,
)
from miniship.wrappers.lagrangian_pz_env import MiniShipLagrangianParallelEnv


class MiniShipAISCommsEnv:
    """
    MiniShip PettingZoo-parallel env wrapper with comprehensive staging.

    Features:
      - AIS communication simulation with realistic channel effects
      - Per-agent PF tracking for noisy state estimation
      - Complete data recording for downstream analysis
      - Multi-worker support with deterministic episode identifiers

    This file intentionally keeps ONLY env orchestration logic.
    AIS/PF algorithms and stage writing live elsewhere.
    """

    metadata = {"name": "MiniShipAISCommsEnv"}

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = dict(cfg)

        # ---- Phase 2 identity (fail-fast; no fallback) ----
        self._run = RunIdentity.from_cfg(self.cfg)

        # ---- Underlying core env (physics/reward/termination) ----
        self._core = MiniShipLagrangianParallelEnv(self.cfg)

        # ---- AIS comms + PF tracker ----
        ais_cfg_path = self.cfg.get("ais_cfg_path", None)
        if not ais_cfg_path:
            raise KeyError("[MiniShipAISCommsEnv] cfg must include ais_cfg_path")
        base_seed = int(self.cfg.get("seed_base", 0))
        fraud_enable = bool(self.cfg.get("fraud_enable", False))
        self._ais = AISCommsSim(cfg_path=str(ais_cfg_path), base_seed=base_seed, fraud_enable=fraud_enable)

        stage4_ctor_ctx = {
            "run_uuid": self._run.run_uuid,
            "out_dir": self._run.out_dir,
            "mode": self._run.mode,
            "worker_index": self._run.worker_index,
            "vector_index": self._run.vector_index,
        }
        self._track_mgr = AISTrackManagerPF(stage4_ctor_ctx=stage4_ctor_ctx)

        # ---- Phase 1 staging sink (writer ownership) ----
        self._staging_enable = bool(self.cfg.get("staging_enable", True))
        self._staging: Optional[StagingSink] = None
        if self._staging_enable:
            self._staging = StagingSink.from_run(self._run)

        # ---- Recording options ----
        # Control what data is recorded (can disable for performance)
        self._record_step_data = bool(self.cfg.get("staging_record_steps", True))
        self._record_pf_estimates = bool(self.cfg.get("staging_record_pf", True))

        # episode identity allocator
        self._ep_alloc = EpisodeIdAllocator(self._run)
        self._ep: Optional[EpisodeIdentity] = None
        self._step_idx: int = 0

        # mappings
        self._int_agents: List[AgentId] = []
        self._ship_of_agent: Dict[AgentId, ShipId] = {}
        self._agent_of_ship: Dict[ShipId, AgentId] = {}

        # Cache for step recording
        self._last_actions: Dict[str, Any] = {}
        self._last_true_states: Dict[ShipId, TrueState] = {}

    # ---------------- PettingZoo parallel API delegation ----------------

    @property
    def possible_agents(self):
        return self._core.possible_agents

    @property
    def agents(self):
        return self._core.agents

    def observation_space(self, agent):
        return self._core.observation_space(agent)

    def action_space(self, agent):
        return self._core.action_space(agent)

    def render(self, *args, **kwargs):
        return self._core.render(*args, **kwargs)

    def close(self):
        if self._staging is not None:
            self._staging.close()
        return self._core.close()

    # ---------------- Internal helpers ----------------

    def _parse_sid_from_agent(self, agent_id: str) -> Optional[int]:
        s = str(agent_id)
        if s.isdigit():
            return int(s)
        if s.startswith("ship_") and s[5:].isdigit():
            return int(s[5:])
        return None

    def _refresh_mappings(self, agents: List[str]) -> None:
        self._int_agents = []
        self._ship_of_agent = {}
        self._agent_of_ship = {}
        for a in agents:
            sid = self._parse_sid_from_agent(a)
            if sid is None:
                raise RuntimeError(f"[MiniShipAISCommsEnv] cannot parse ship id from agent={a!r}")
            int_a = f"ship_{sid}"
            self._int_agents.append(int_a)
            if int_a in self._ship_of_agent:
                raise RuntimeError(f"[MiniShipAISCommsEnv] duplicate internal agent id: {int_a}")
            self._ship_of_agent[int_a] = sid
            if sid in self._agent_of_ship:
                raise RuntimeError(f"[MiniShipAISCommsEnv] duplicate ship id mapping: sid={sid}")
            self._agent_of_ship[sid] = int_a

    def _get_true_states(self) -> Tuple[float, Dict[ShipId, TrueState]]:
        st = getattr(self._core, "state", None)
        if st is None:
            raise RuntimeError("[MiniShipAISCommsEnv] core env has no .state; cannot build TrueState")
        t = float(getattr(st, "t", 0.0))
        ships = getattr(st, "ships", None)
        if ships is None:
            raise RuntimeError("[MiniShipAISCommsEnv] core env state has no .ships")
        out: Dict[ShipId, TrueState] = {}
        for i, ship in enumerate(ships):
            sid = int(getattr(ship, "sid", getattr(ship, "ship_id", i + 1)))
            x = float(ship.pos[0])
            y = float(ship.pos[1])
            v = float(getattr(ship, "v", 0.0))
            psi = float(getattr(ship, "psi", 0.0))
            vx = v * math.cos(psi)
            vy = v * math.sin(psi)
            yaw = float(math.atan2(vy, vx)) if (abs(vx) + abs(vy)) > 1e-12 else float(psi)
            out[sid] = TrueState(ship_id=sid, t=t, x=x, y=y, vx=vx, vy=vy, yaw_east_ccw_rad=yaw)
        return t, out

    def _build_env_config(self) -> Dict[str, Any]:
        """Extract environment configuration for recording."""
        cfg = self.cfg
        return {
            "N": int(cfg.get("N", 2)),
            "dt": float(cfg.get("dt", 0.5)),
            "T_max": float(cfg.get("T_max", 220.0)),
            "v_max": float(cfg.get("v_max", 2.0)),
            "v_min": float(cfg.get("v_min", 0.1)),
            "goal_tol": float(cfg.get("goal_tol", 10.0)),
            "collide_thr": float(cfg.get("collide_thr", 12.0)),
            "spawn_area": float(cfg.get("spawn_area", 240.0)),
            "spawn_len": float(cfg.get("spawn_len", 160.0)),
            "spawn_mode": str(cfg.get("spawn_mode", "random_fixedlen")),
            "use_lagrangian": bool(cfg.get("use_lagrangian", True)),
            "use_guard": bool(cfg.get("use_guard", True)),
            "ais_cfg_path": str(cfg.get("ais_cfg_path", "")),
        }

    def _build_ship_init_states(self, true_states: Dict[ShipId, TrueState]) -> List[Dict[str, Any]]:
        """Build initial ship state records."""
        st = getattr(self._core, "state", None)
        ships = getattr(st, "ships", None) if st else None

        results = []
        for sid, ts in sorted(true_states.items()):
            agent_id = self._agent_of_ship.get(sid, f"ship_{sid}")

            # Get goal from ship object
            goal_x, goal_y, goal_dist = 0.0, 0.0, 0.0
            if ships:
                for ship in ships:
                    ship_sid = int(getattr(ship, "sid", getattr(ship, "ship_id", 0)))
                    if ship_sid == sid:
                        goal = getattr(ship, "goal", None)
                        if goal is not None:
                            goal_x = float(goal[0])
                            goal_y = float(goal[1])
                            dx = goal_x - ts.x
                            dy = goal_y - ts.y
                            goal_dist = math.hypot(dx, dy)
                        break

            results.append(build_ship_state_dict(
                ship_id=sid,
                agent_id=agent_id,
                x=ts.x,
                y=ts.y,
                psi=ts.yaw_east_ccw_rad,
                v=ts.sog,
                goal_x=goal_x,
                goal_y=goal_y,
                goal_dist=goal_dist,
                reached=False,
                vx=ts.vx,
                vy=ts.vy,
            ))
        return results

    def _build_step_ship_states(
        self,
        true_states: Dict[ShipId, TrueState],
        infos: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build ship state records for current step."""
        results = []
        for sid, ts in sorted(true_states.items()):
            agent_id = self._agent_of_ship.get(sid, f"ship_{sid}")
            agent_info = infos.get(agent_id, {})

            results.append(build_ship_state_dict(
                ship_id=sid,
                agent_id=agent_id,
                x=ts.x,
                y=ts.y,
                psi=ts.yaw_east_ccw_rad,
                v=ts.sog,
                goal_x=float(agent_info.get("gx", 0.0)),
                goal_y=float(agent_info.get("gy", 0.0)),
                goal_dist=float(agent_info.get("goal_dist", 0.0)),
                reached=bool(agent_info.get("reached_goal", False)),
                vx=ts.vx,
                vy=ts.vy,
            ))
        return results

    def _build_step_rl_data(
        self,
        actions: Dict[str, Any],
        rewards: Dict[str, float],
        infos: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build RL data records for current step."""
        results = []
        for agent_id in sorted(self._int_agents):
            info = infos.get(agent_id, {})
            action = actions.get(agent_id, [0.0, 0.0])

            # Convert action to list
            if hasattr(action, "tolist"):
                action_raw = action.tolist()
            elif isinstance(action, (list, tuple)):
                action_raw = list(action)
            else:
                action_raw = [float(action)]

            results.append(build_rl_data_dict(
                agent_id=agent_id,
                action_raw=action_raw,
                dpsi_cmd=float(info.get("dpsi_cmd", 0.0)),
                v_cmd=float(info.get("v_cmd", 0.0)),
                v_target=float(info.get("v_target", 0.0)),
                r_task=float(info.get("r_task", 0.0)),
                r_shaped=float(info.get("r_shaped", 0.0)),
                r_total=float(rewards.get(agent_id, 0.0)),
                c_near=float(info.get("c_near", 0.0)),
                c_rule=float(info.get("c_rule", 0.0)),
                c_coll=float(info.get("c_coll", 0.0)),
                c_time=float(info.get("c_time", 0.0)),
                risk=float(info.get("risk", 0.0)),
                guard_triggered=bool(info.get("guard_trig_step_mean", 0.0) > 0.5),
            ))
        return results

    def _infer_term_reason(self, infos: Dict[str, Any]) -> str:
        """Infer termination reason from infos."""
        for agent_id, info in infos.items():
            if agent_id in ("__all__", "__common__"):
                continue
            reason = info.get("term_reason")
            if reason:
                return str(reason)
        # Fallback
        for agent_id, info in infos.items():
            if agent_id in ("__all__", "__common__"):
                continue
            if info.get("collision", False) or info.get("coll", False):
                return "collision"
            if info.get("success", False) or info.get("success_all", False):
                return "success"
            if info.get("timeout", False):
                return "timeout"
        return "unknown"

    # ---------------- Reset with comprehensive staging ----------------

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, infos = self._core.reset(seed=seed, options=options)

        # episode identity (Phase 2)
        self._ep = self._ep_alloc.next()
        self._step_idx = 0

        # derive agent mappings from current obs keys (core env agents)
        agents = [a for a in obs.keys() if a not in ("__all__", "__common__")]
        self._refresh_mappings(agents)

        # init true states
        t0, true_states = self._get_true_states()
        ship_ids = sorted(true_states.keys())
        self._last_true_states = true_states

        # reset AIS and PF
        self._ais.reset(ships=ship_ids, t0=t0, agent_map=self._agent_of_ship)
        self._track_mgr.reset(ship_ids=ship_ids, t0=t0, init_states=true_states, agent_ids=self._int_agents)
        self._track_mgr.set_agent_ship_map(self._agent_of_ship)

        # stage4 context for PF (Phase 2)
        self._track_mgr.set_stage4_context({
            **self._run.to_dict(),
            **self._ep.to_dict(),
            "stage3_episode_uid": self._ep.episode_uid,
            "stage3_episode_idx": self._ep.episode_idx,
        })

        # ---- Comprehensive staging: episode_init ----
        if self._staging is not None:
            # Get complete AIS parameters
            ais_params = self._ais.get_episode_params()

            # Build env config
            env_config = self._build_env_config()
            env_config["seed"] = seed

            # Build ship init states
            ship_init_states = self._build_ship_init_states(true_states)

            # Emit comprehensive episode init
            self._staging.emit_stage3_episode_init(
                ep=self._ep,
                env_config=env_config,
                ais_params=ais_params,
                ship_init_states=ship_init_states,
                extra={
                    "run_uuid": self._run.run_uuid,
                    "worker_index": self._run.worker_index,
                    "vector_index": self._run.vector_index,
                },
            )

            # Emit episode start event
            self._staging.emit_stage4_episode_start(
                ep=self._ep,
                t0=t0,
                extra={"N": len(ship_ids)},
            )

        infos = dict(infos) if isinstance(infos, dict) else {}
        infos["__common__"] = {
            **(infos.get("__common__", {}) if isinstance(infos.get("__common__", {}), dict) else {}),
            "episode_uid": self._ep.episode_uid,
            "episode_idx": self._ep.episode_idx,
        }
        return obs, infos

    # ---------------- Step with comprehensive staging ----------------

    def step(self, actions: Dict[str, Any]):
        obs, rews, terms, truncs, infos = self._core.step(actions)

        if self._ep is None:
            raise RuntimeError("[MiniShipAISCommsEnv] BUG: step called before reset")

        # Cache actions for recording
        self._last_actions = dict(actions)

        # Update AIS and PF based on current truth
        t, true_states = self._get_true_states()
        ready = self._ais.step(t=t, true_states=true_states)
        self._track_mgr.ingest(t=t, ready=ready, true_states=true_states)

        # ---- Comprehensive staging: step record ----
        if self._staging is not None and self._record_step_data:
            # Build ship states
            ship_states = self._build_step_ship_states(true_states, infos)

            # Build RL data
            rl_data = self._build_step_rl_data(actions, rews, infos)

            # Get comm stats
            comm_stats = self._ais.metrics_snapshot()

            # Get PF estimates (optional, can be expensive)
            pf_estimates = None
            if self._record_pf_estimates:
                pf_estimates = self._track_mgr.get_pf_estimates_for_staging(t, true_states)

            # Emit comprehensive step record
            self._staging.emit_stage3_step(
                ep=self._ep,
                step_idx=self._step_idx,
                t_sim=t,
                ship_states=ship_states,
                rl_data=rl_data,
                comm_stats=comm_stats,
                pf_estimates=pf_estimates,
            )

        self._step_idx += 1
        self._last_true_states = true_states

        # ---- Check for episode end ----
        done_all = bool(terms.get("__all__", False)) or bool(truncs.get("__all__", False))
        if not done_all:
            # Check per-agent termination
            done_all = all(
                terms.get(a, False) or truncs.get(a, False)
                for a in self._int_agents
            )

        if done_all and self._staging is not None:
            term_reason = self._infer_term_reason(infos)

            # Build final stats
            final_stats = {
                "ais_metrics_final": self._ais.metrics_snapshot(),
                "total_steps": self._step_idx,
            }

            # Build per-agent stats
            per_agent_stats = []
            for agent_id in sorted(self._int_agents):
                info = infos.get(agent_id, {})
                per_agent_stats.append({
                    "agent_id": agent_id,
                    "success": bool(info.get("success", False)),
                    "collision": bool(info.get("collision", False)),
                    "timeout": bool(info.get("timeout", False)),
                    "goal_dist_final": float(info.get("goal_dist", 0.0)),
                    "reached": bool(info.get("reached_goal", False)),
                })

            # Emit episode end records
            self._staging.emit_stage3_episode_end(
                ep=self._ep,
                t_end=t,
                total_steps=self._step_idx,
                term_reason=term_reason,
                final_stats=final_stats,
                per_agent_stats=per_agent_stats,
            )

            self._staging.emit_stage4_episode_end(
                ep=self._ep,
                t_end=t,
                term_reason=term_reason,
                stats=final_stats,
            )

            # Emit collision event if applicable
            if term_reason == "collision":
                # Find collision pair
                positions = {
                    sid: (float(ts.x), float(ts.y))
                    for sid, ts in true_states.items()
                }
                ship_pair = tuple(sorted(true_states.keys())[:2])
                self._staging.emit_stage4_collision(
                    ep=self._ep,
                    t=t,
                    ship_pair=ship_pair,
                    positions=positions,
                )

        infos = dict(infos) if isinstance(infos, dict) else {}
        infos["__common__"] = {
            **(infos.get("__common__", {}) if isinstance(infos.get("__common__", {}), dict) else {}),
            "episode_uid": self._ep.episode_uid,
            "episode_idx": self._ep.episode_idx,
            "step_idx": self._step_idx,
        }
        return obs, rews, terms, truncs, infos
