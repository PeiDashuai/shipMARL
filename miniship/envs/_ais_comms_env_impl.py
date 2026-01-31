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

import random
import time

from ais_comms.ais_comms import AISCommsSim
from ais_comms.datatypes import AgentId, ShipId, TrueState
from ais_comms.track_manager_pf import AISTrackManagerPF

from miniship.envs._ais_identity import EpisodeIdAllocator, EpisodeIdentity, RunIdentity

# Schema version for data format tracking (increment on breaking changes)
STAGING_SCHEMA_VERSION = "2.0.0"
from miniship.envs._ais_staging import (
    StagingSink,
    build_ship_state_dict,
    build_rl_data_dict,
    build_pf_estimate_dict,
)
from miniship.wrappers.lagrangian_pz_env import MiniShipLagrangianParallelEnv
from miniship.dynamics.ship import Ship
from miniship.observe.builder import build_observations
import numpy as np


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
        # IMPORTANT: Disable AISObsWrapper in _core since we handle observations via PF
        # This prevents double AIS simulation (AISObsWrapper uses hardcoded params, not yaml config)
        core_cfg = dict(self.cfg)
        core_cfg["use_ais_obs"] = False  # We build PF-based observations ourselves
        self._core = MiniShipLagrangianParallelEnv(core_cfg)

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
            # Build run-level metadata for reproducibility
            run_metadata = self._build_run_metadata()
            self._staging = StagingSink.from_run(self._run, run_metadata=run_metadata)

        # ---- Recording options ----
        # Control what data is recorded (can disable for performance)
        self._record_step_data = bool(self.cfg.get("staging_record_steps", True))
        self._record_pf_estimates = bool(self.cfg.get("staging_record_pf", True))
        # Per-ship/per-link comm stats (default ON for causality analysis)
        self._record_per_ship_comm = bool(self.cfg.get("staging_record_per_ship_comm", True))
        self._record_per_link_comm = bool(self.cfg.get("staging_record_per_link_comm", True))

        # episode identity allocator
        self._ep_alloc = EpisodeIdAllocator(self._run)
        self._ep: Optional[EpisodeIdentity] = None
        self._step_idx: int = 0
        self._t0: float = 0.0  # Episode start time

        # mappings
        self._int_agents: List[AgentId] = []
        self._ship_of_agent: Dict[AgentId, ShipId] = {}
        self._agent_of_ship: Dict[ShipId, AgentId] = {}

        # Cache for step recording
        self._last_actions: Dict[str, Any] = {}
        self._last_true_states: Dict[ShipId, TrueState] = {}

        # Episode-level accumulators for summary stats
        self._ep_reward_sum: Dict[str, float] = {}
        self._ep_cost_sum: Dict[str, float] = {}  # near, rule, coll, time
        self._ep_pf_errors: List[Dict[str, float]] = []  # per-step PF errors

        # Per-step delta tracking for comm stats (cumulative → delta conversion)
        self._prev_per_ship_comm: Dict[int, Dict[str, Any]] = {}
        self._prev_per_link_comm: Dict[str, Dict[str, Any]] = {}

        # ---- Observation building parameters ----
        # CRITICAL: Read all observation params from CORE env to ensure consistency
        # Previously had different defaults causing observation mismatch (e.g., v_max: 3.0 vs 2.0)
        core_env = self._core.core  # MiniShipCoreEnv
        self._K_neighbors = int(core_env.K_neighbors)
        self._spawn_mode = str(core_env.spawn_mode)
        self._spawn_area = float(core_env.spawn_area)
        self._spawn_len = float(core_env.spawn_len)
        self._v_max = float(core_env.v_max)

        # PF uncertainty thresholds for observation validity
        self._pf_stale_threshold = float(cfg.get("pf_stale_threshold", 5.0))  # seconds
        self._pf_silence_threshold = float(cfg.get("pf_silence_threshold", 10.0))  # seconds

        # Cache for goals (from true states)
        self._ship_goals: Dict[ShipId, np.ndarray] = {}

        print(f"[MiniShipAISCommsEnv] PF-based observations enabled, K={self._K_neighbors}, v_max={self._v_max}")

    # ---------------- PettingZoo parallel API delegation ----------------

    @property
    def core_env(self):
        """Expose core env for callbacks to access _last_infos through wrapper chain."""
        return getattr(self._core, "core_env", None) or getattr(self._core, "core", None) or self._core

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
        _dbg_step = getattr(self, '_dbg_pf_obs_step', 0)
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
            # Debug: print actual ship.v
            if _dbg_step < 2 and i == 0:
                print(f"[GET_TRUE_STATES] step={_dbg_step} sid={sid} ship.v={v:.3f} v_max={self._v_max:.3f}")
        return t, out

    def _build_pf_observations(
        self,
        t: float,
        true_states: Dict[ShipId, TrueState],
    ) -> Dict[str, np.ndarray]:
        """
        Build observations from PF estimates.

        For each agent:
          - Ego ship uses TRUE state (agent knows its own position exactly)
          - Neighbor ships use PF estimates from this agent's tracker
          - Neighbors with invalid/stale PF tracks are marked with ais_valid=False

        Returns:
            Dict[agent_id, observation_array]
        """
        obs_out: Dict[str, np.ndarray] = {}

        # Debug: track PF estimate availability
        _dbg_step = getattr(self, '_dbg_pf_obs_step', 0)
        _dbg_print = (_dbg_step < 3)  # Only print first 3 steps
        self._dbg_pf_obs_step = _dbg_step + 1

        for agent_id in self._int_agents:
            ego_sid = self._ship_of_agent.get(agent_id)
            if ego_sid is None:
                continue

            # Build Ship objects for observation construction
            ships_for_obs: List[Ship] = []
            _dbg_pf_valid = 0
            _dbg_pf_invalid = 0

            # Process all ships in consistent order
            for sid in sorted(true_states.keys()):
                true_st = true_states[sid]
                goal = self._ship_goals.get(sid, np.array([0.0, 0.0], dtype=np.float64))

                if sid == ego_sid:
                    # Ego ship: use TRUE state (agent knows its own position)
                    ship = Ship(
                        sid=sid,
                        pos=np.array([true_st.x, true_st.y], dtype=np.float64),
                        goal=goal,
                        psi=true_st.yaw_east_ccw_rad,
                        v=true_st.sog,
                    )
                    ship.reached = False  # Will be updated from core env
                    ship.ais_valid = True
                    ship.ais_u_stale = 0.0
                    ship.ais_u_silence = 0.0
                else:
                    # Neighbor ship: use PF estimate from this agent's tracker
                    x_pred = self._track_mgr.get_estimate(agent_id, sid, t)

                    # Debug: compare PF estimate vs true state for neighbors
                    if _dbg_print and agent_id == self._int_agents[0]:
                        true_x, true_y = true_st.x, true_st.y
                        true_vx, true_vy = true_st.vx, true_st.vy
                        true_yaw = true_st.yaw_east_ccw_rad
                        if x_pred is not None:
                            pf_x, pf_y = x_pred[0], x_pred[1]
                            pf_vx, pf_vy = x_pred[2], x_pred[3]
                            pf_yaw = x_pred[4]
                            pos_err = math.sqrt((pf_x - true_x)**2 + (pf_y - true_y)**2)
                            vel_err = math.sqrt((pf_vx - true_vx)**2 + (pf_vy - true_vy)**2)
                            yaw_err = abs(pf_yaw - true_yaw)
                            print(f"[PF_EST_CMP] step={_dbg_step} nei_sid={sid} "
                                  f"TRUE pos=({true_x:.1f},{true_y:.1f}) vel=({true_vx:.2f},{true_vy:.2f}) yaw={true_yaw:.2f}")
                            print(f"[PF_EST_CMP] step={_dbg_step} nei_sid={sid} "
                                  f"  PF pos=({pf_x:.1f},{pf_y:.1f}) vel=({pf_vx:.2f},{pf_vy:.2f}) yaw={pf_yaw:.2f}")
                            print(f"[PF_EST_CMP] step={_dbg_step} nei_sid={sid} "
                                  f"  ERR pos={pos_err:.1f}m vel={vel_err:.2f}m/s yaw={math.degrees(yaw_err):.1f}deg")
                        else:
                            print(f"[PF_EST_CMP] step={_dbg_step} nei_sid={sid} PF estimate is None!")

                    if x_pred is not None:
                        # PF estimate available: [x, y, vx, vy, yaw]
                        est_x = float(x_pred[0])
                        est_y = float(x_pred[1])
                        est_vx = float(x_pred[2])
                        est_vy = float(x_pred[3])
                        est_yaw = float(x_pred[4])
                        est_sog = math.sqrt(est_vx ** 2 + est_vy ** 2)

                        ship = Ship(
                            sid=sid,
                            pos=np.array([est_x, est_y], dtype=np.float64),
                            goal=goal,
                            psi=est_yaw,
                            v=est_sog,
                        )
                        ship.reached = False
                        ship.ais_valid = True

                        # Compute uncertainty metrics from PF track
                        track_age = self._get_track_age(agent_id, sid, t)
                        ship.ais_u_stale = float(np.clip(
                            track_age / self._pf_stale_threshold, 0.0, 1.0
                        ))
                        ship.ais_u_silence = 0.0  # TODO: track silence time
                        _dbg_pf_valid += 1
                    else:
                        # No PF estimate: mark as invalid, use zero/fallback
                        # Policy should learn to ignore neighbors with ais_valid=False
                        ship = Ship(
                            sid=sid,
                            pos=np.array([0.0, 0.0], dtype=np.float64),
                            goal=goal,
                            psi=0.0,
                            v=0.0,
                        )
                        ship.reached = False
                        ship.ais_valid = False
                        ship.ais_u_stale = 1.0
                        ship.ais_u_silence = 1.0
                        _dbg_pf_invalid += 1

                ships_for_obs.append(ship)

            if _dbg_print and agent_id == self._int_agents[0]:
                # Find ego ship in the list
                ego_ship = None
                for s in ships_for_obs:
                    if s.sid == ego_sid:
                        ego_ship = s
                        break
                ego_pos = tuple(ego_ship.pos) if ego_ship else None
                ego_goal = tuple(ego_ship.goal) if ego_ship else None
                print(f"[PF_OBS_DBG] step={_dbg_step} agent={agent_id} ego_sid={ego_sid} pf_valid={_dbg_pf_valid} pf_invalid={_dbg_pf_invalid} "
                      f"ego_pos={ego_pos} ego_goal={ego_goal}")

            # Build observation for this agent
            obs_dict = build_observations(
                ships_for_obs,
                self._K_neighbors,
                self._spawn_mode,
                self._spawn_area,
                self._spawn_len,
                self._v_max,
            )

            # Extract observation for this agent's ship
            # obs_dict keys are ship_id strings like "1", "2"
            ego_sid_str = str(ego_sid)
            if ego_sid_str in obs_dict:
                obs_out[agent_id] = obs_dict[ego_sid_str]
            else:
                # Fallback: find by index
                for idx, sid in enumerate(sorted(true_states.keys())):
                    if sid == ego_sid:
                        sid_str = str(idx + 1)
                        if sid_str in obs_dict:
                            obs_out[agent_id] = obs_dict[sid_str]
                        break

        return obs_out

    def _get_track_age(self, agent_id: str, sid: int, t: float) -> float:
        """Get the age of a PF track (time since last measurement)."""
        try:
            st = self._track_mgr.agent_states.get(agent_id)
            if st is None:
                return float('inf')
            tr = st.tracks.get(sid)
            if tr is None or tr.pf is None:
                return float('inf')
            last_ts = getattr(tr.pf, 'last_ts', None)
            if last_ts is None:
                return float('inf')
            return max(0.0, t - float(last_ts))
        except Exception:
            return float('inf')

    def _cache_ship_goals(self, true_states: Dict[ShipId, TrueState]) -> None:
        """Cache ship goals from core env state."""
        self._ship_goals.clear()
        st = getattr(self._core, "state", None)
        if st is None:
            print("[_cache_ship_goals] WARNING: self._core.state is None")
            return
        ships = getattr(st, "ships", None)
        if ships is None:
            print("[_cache_ship_goals] WARNING: state.ships is None")
            return
        for ship in ships:
            sid = int(getattr(ship, "sid", getattr(ship, "ship_id", -1)))
            if sid >= 0 and hasattr(ship, "goal"):
                self._ship_goals[sid] = np.array(ship.goal, dtype=np.float64)
        print(f"[_cache_ship_goals] Cached goals for {len(self._ship_goals)} ships: {list(self._ship_goals.keys())}")

    def _compute_config_hashes(self) -> Dict[str, str]:
        """Compute SHA256 hashes of config files for reproducibility."""
        import hashlib
        import subprocess

        hashes = {}

        # Git commit hash
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                hashes["git_commit"] = result.stdout.strip()[:12]
        except Exception:
            hashes["git_commit"] = "unknown"

        # AIS config file hash
        ais_cfg_path = self.cfg.get("ais_cfg_path", "")
        if ais_cfg_path:
            try:
                with open(ais_cfg_path, "rb") as f:
                    hashes["ais_cfg_sha256"] = hashlib.sha256(f.read()).hexdigest()[:16]
            except Exception:
                hashes["ais_cfg_sha256"] = "unknown"

        # PF config hash (based on full PF config)
        pf_cfg = self._get_pf_config()
        try:
            import json
            pf_str = json.dumps(pf_cfg, sort_keys=True, default=str)
            hashes["pf_cfg_sha256"] = hashlib.sha256(pf_str.encode()).hexdigest()[:16]
        except Exception:
            hashes["pf_cfg_sha256"] = "unknown"

        return hashes

    def _build_run_metadata(self) -> Dict[str, Any]:
        """
        Build run-level metadata for shard_header.

        Captures training configuration for reproducibility:
        - Training hyperparameters (from cfg)
        - Config file hashes
        - RL/safety mechanism settings
        """
        import hashlib

        cfg = self.cfg
        metadata = {}

        # Training hyperparameters (subset relevant for analysis)
        train_args = {
            "N": int(cfg.get("N", 2)),
            "dt": float(cfg.get("dt", 0.5)),
            "T_max": float(cfg.get("T_max", 220.0)),
            "use_lagrangian": bool(cfg.get("use_lagrangian", True)),
            "use_guard": bool(cfg.get("use_guard", True)),
            "use_ais_obs": bool(cfg.get("use_ais_obs", False)),
            "ais_cfg_path": str(cfg.get("ais_cfg_path", "")),
            # RL hyperparameters (if present)
            "lr": float(cfg.get("lr", cfg.get("learning_rate", 0.0))),
            "train_batch_size": int(cfg.get("train_batch_size", 0)),
            "sgd_minibatch_size": int(cfg.get("sgd_minibatch_size", 0)),
            "num_sgd_iter": int(cfg.get("num_sgd_iter", 0)),
            "entropy_coeff": float(cfg.get("entropy_coeff", 0.0)),
            # Lagrangian settings
            "dual_freeze": bool(cfg.get("dual_freeze", False)),
            "dual_update_in_env": bool(cfg.get("dual_update_in_env", False)),
            # Safety/guard settings
            "guard_mode": str(cfg.get("guard_mode", "")),
        }
        metadata["train_args"] = train_args

        # Config file hashes (for detecting content changes)
        ais_cfg_path = cfg.get("ais_cfg_path", "")
        if ais_cfg_path:
            try:
                with open(ais_cfg_path, "rb") as f:
                    metadata["ais_cfg_sha256"] = hashlib.sha256(f.read()).hexdigest()[:16]
                metadata["ais_cfg_path"] = str(ais_cfg_path)
            except Exception:
                metadata["ais_cfg_sha256"] = "unknown"

        return metadata

    def _get_pf_config(self) -> Dict[str, Any]:
        """
        Extract complete PF configuration for reproducibility.
        This is the 'configuration contract' for the particle filter.
        """
        tm = self._track_mgr
        return {
            # Algorithm params
            "num_particles": int(getattr(tm, "num_particles", getattr(tm, "N", 256))),
            "resample_threshold_ratio": float(getattr(tm, "resample_threshold_ratio", 0.5)),
            "pf_seed_base": int(getattr(tm, "pf_seed_base", 2025)),

            # Process noise (motion model uncertainty)
            "process_std_a": float(getattr(tm, "process_std_a", 0.5)),
            "process_std_yaw_deg": float(getattr(tm, "process_std_yaw_deg", 10.0)),

            # Measurement noise (observation uncertainty)
            "meas_std_pos": float(getattr(tm, "meas_std_pos", 8.0)),
            "meas_std_sog": float(getattr(tm, "meas_std_sog", 0.2)),
            "meas_std_cog_deg": float(getattr(tm, "meas_std_cog_deg", 8.0)),

            # Relock/gating params
            "soft_relock_dist": float(getattr(tm, "soft_relock_dist", 20.0)),
            "hard_relock_dist": float(getattr(tm, "hard_relock_dist", 40.0)),
            "soft_relock_beta": float(getattr(tm, "soft_relock_beta", 0.3)),

            # Preproc params
            "preproc_window_sec": float(getattr(tm, "preproc_window_sec", 60.0)),
            "preproc_max_buffer": int(getattr(tm, "preproc_max_buffer", 256)),
            "preproc_max_pos_jump": float(getattr(tm, "preproc_max_pos_jump", 80.0)),
            "preproc_max_cog_jump_deg": float(getattr(tm, "preproc_max_cog_jump_deg", 90.0)),

            # Track management
            "max_age": float(getattr(tm, "max_age", 20.0)),
        }

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
        lagrangian_state: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Build RL data records for current step."""
        # Get lambda values from lagrangian state
        if lagrangian_state is None:
            lagrangian_state = self._get_lagrangian_state()
        lambdas = lagrangian_state.get("lambdas", {})
        dual_version = lagrangian_state.get("dual_version", 0)
        guard_state = lagrangian_state.get("guard", {})

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

            # Extract guard details from info
            guard_triggered = bool(info.get("guard_trig_step_mean", 0.0) > 0.5)
            guard_margin = float(info.get("guard_margin", info.get("safety_margin", 0.0)))
            guard_override = bool(info.get("guard_action_override", info.get("action_overridden", False)))

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
                guard_triggered=guard_triggered,
                # Lagrangian dual variables
                lambda_near=float(lambdas.get("near", 0.0)),
                lambda_rule=float(lambdas.get("rule", 0.0)),
                lambda_coll=float(lambdas.get("coll", 0.0)),
                lambda_time=float(lambdas.get("time", 0.0)),
                dual_version=int(dual_version),
                # Guard controller details
                guard_margin=guard_margin,
                guard_action_override=guard_override,
            ))
        return results

    def _clean_comm_stats(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean comm_stats to use canonical field names only.
        Removes redundant aliases to prevent semantic drift.
        """
        # Canonical fields (single authority, no aliases)
        return {
            # Packet rates
            "ppr": float(raw.get("ppr", 0.0)),        # Packet Pass Rate
            "pdr": float(raw.get("pdr", 0.0)),        # Packet Delivery Rate
            "drop_rate": float(raw.get("drop_rate", 0.0)),
            # Delay (seconds)
            "delay_mean": float(raw.get("delay_avg", raw.get("delay_mean_s", 0.0))),
            "delay_p95": float(raw.get("delay_p95", raw.get("delay_p95_s", 0.0))),
            "delay_max": float(raw.get("delay_max", raw.get("delay_max_s", 0.0))),
            # Age (seconds)
            "age_mean": float(raw.get("age_avg", raw.get("age_mean_s", 0.0))),
            "age_p95": float(raw.get("age_p95", raw.get("age_p95_s", 0.0))),
            "age_max": float(raw.get("age_max", raw.get("age_max_s", 0.0))),
            # Channel quality
            "bad_occupancy": float(raw.get("bad_occupancy", 0.0)),
            "reorder_count": int(raw.get("reorder_count", 0)),
            "reorder_rate": float(raw.get("reorder_rate", 0.0)),
            # Clock drift effect
            "age_delay_gap_mean": float(raw.get("age_delay_gap_avg", 0.0)),
        }

    def _get_lagrangian_state(self) -> Dict[str, Any]:
        """
        Extract current Lagrangian dual variables and guard state.
        This enables analysis of how constraints affect policy behavior.
        """
        state: Dict[str, Any] = {
            "dual_version": 0,
            "lambdas": {"near": 0.0, "rule": 0.0, "coll": 0.0, "time": 0.0},
            "running_c": {},
            "guard": {"enabled": False},
        }

        # Try to get dual manager from core env chain
        core = self._core
        dual_mgr = None
        guard_ctrl = None

        # Traverse wrapper chain to find DualManager and GuardController
        for _ in range(10):  # Max depth
            if hasattr(core, "dual_mgr") and core.dual_mgr is not None:
                dual_mgr = core.dual_mgr
            if hasattr(core, "core") and hasattr(core.core, "guard_ctrl"):
                guard_ctrl = getattr(core.core, "guard_ctrl", None)
            if hasattr(core, "_dual_version"):
                state["dual_version"] = int(getattr(core, "_dual_version", 0))

            # Move to inner env
            if hasattr(core, "env"):
                core = core.env
            elif hasattr(core, "core"):
                core = core.core
            else:
                break

        # Extract lambda values from dual manager
        if dual_mgr is not None:
            try:
                if hasattr(dual_mgr, "get_lambdas") and callable(dual_mgr.get_lambdas):
                    lam = dual_mgr.get_lambdas() or {}
                elif hasattr(dual_mgr, "lambdas"):
                    lam = getattr(dual_mgr, "lambdas", {}) or {}
                elif hasattr(dual_mgr, "lam"):
                    lam = getattr(dual_mgr, "lam", {}) or {}
                else:
                    lam = {}
                state["lambdas"] = {k: float(v) for k, v in lam.items()}
            except Exception:
                pass

            # Extract running cost EMA
            try:
                if hasattr(dual_mgr, "get_running_c") and callable(dual_mgr.get_running_c):
                    rc = dual_mgr.get_running_c() or {}
                elif hasattr(dual_mgr, "running_c"):
                    rc = getattr(dual_mgr, "running_c", {}) or {}
                elif hasattr(dual_mgr, "c_ema"):
                    rc = getattr(dual_mgr, "c_ema", {}) or {}
                else:
                    rc = {}
                state["running_c"] = {k: float(v) for k, v in rc.items()}
            except Exception:
                pass

        # Extract guard controller state
        if guard_ctrl is not None:
            try:
                state["guard"] = {
                    "enabled": True,
                    "threshold": float(getattr(guard_ctrl, "threshold", getattr(guard_ctrl, "thr", 0.0))),
                    "margin": float(getattr(guard_ctrl, "margin", 0.0)),
                }
            except Exception:
                state["guard"] = {"enabled": True}

        return state

    def _get_per_ship_comm_stats(self) -> Dict[int, Dict[str, Any]]:
        """
        Get per-ship communication statistics for diagnostic analysis.

        Enables attributing PF errors to specific ship's communication quality:
        - Which target ship's messages are delayed/dropped
        - TX/drop/reorder counts per ship (for "which ship's comms are bad")
        - Last RX timing (for causality analysis: comm delay → PF error → risk)

        Returns:
            Dict[ship_id, {
                tx_count, drop_count, rx_count, reorder_count,
                ppr, pdr, drop_rate, reorder_rate,
                delay_mean/p95/max, age_mean/p95/max,
                last_rx_delay_s, last_rx_age_s, last_rx_report_ts, last_rx_arrival_ts
            }]
        """
        per_ship: Dict[int, Dict[str, Any]] = {}

        try:
            if hasattr(self._ais, "get_per_ship_metrics"):
                raw = self._ais.get_per_ship_metrics()
                for sid, metrics in raw.items():
                    per_ship[int(sid)] = {
                        # Counts (new: tx_count, drop_count, reorder_count)
                        "tx_count": int(metrics.get("tx_count", 0)),
                        "drop_count": int(metrics.get("drop_count", 0)),
                        "rx_count": int(metrics.get("rx_count", 0)),
                        "reorder_count": int(metrics.get("reorder_count", 0)),
                        # Derived rates (new: ppr, pdr, drop_rate, reorder_rate)
                        "ppr": float(metrics.get("ppr", 0.0)),
                        "pdr": float(metrics.get("pdr", 0.0)),
                        "drop_rate": float(metrics.get("drop_rate", 0.0)),
                        "reorder_rate": float(metrics.get("reorder_rate", 0.0)),
                        # Delay/age stats
                        "delay_mean": float(metrics.get("delay_mean", 0.0)),
                        "delay_p95": float(metrics.get("delay_p95", 0.0)),
                        "delay_max": float(metrics.get("delay_max", 0.0)),
                        "age_mean": float(metrics.get("age_mean", 0.0)),
                        "age_p95": float(metrics.get("age_p95", 0.0)),
                        "age_max": float(metrics.get("age_max", 0.0)),
                        # Last RX info (critical for PF error attribution)
                        "last_rx_delay_s": float(metrics.get("last_rx_delay_s", 0.0)),
                        "last_rx_age_s": float(metrics.get("last_rx_age_s", 0.0)),
                        "last_rx_report_ts": float(metrics.get("last_rx_report_ts", 0.0)),
                        "last_rx_arrival_ts": float(metrics.get("last_rx_arrival_ts", 0.0)),
                    }
        except Exception:
            pass

        return per_ship

    def _get_per_link_comm_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get per-link (rx_agent, tx_ship) communication statistics.

        Enables answering "same target, different receiver" questions:
        - Does agent_i receive ship_k's messages worse than agent_j?
        - Link-level causality: link quality → agent's PF error → agent's action

        Returns:
            Dict["rx_agent:tx_ship", {
                tx_count, drop_count, rx_count, reorder_count,
                ppr, pdr, drop_rate, reorder_rate,
                delay_mean/p95/max, age_mean/p95/max
            }]
        """
        per_link: Dict[str, Dict[str, Any]] = {}

        try:
            if hasattr(self._ais, "get_per_link_metrics"):
                raw = self._ais.get_per_link_metrics()
                for link_key, metrics in raw.items():
                    # link_key is (rx_agent, tx_ship) tuple
                    rx_agent, tx_ship = link_key
                    key_str = f"{rx_agent}:{tx_ship}"
                    per_link[key_str] = {
                        "rx_agent_id": str(rx_agent),
                        "tx_ship_id": int(tx_ship),
                        # Counts
                        "tx_count": int(metrics.get("tx_count", 0)),
                        "drop_count": int(metrics.get("drop_count", 0)),
                        "rx_count": int(metrics.get("rx_count", 0)),
                        "reorder_count": int(metrics.get("reorder_count", 0)),
                        # Derived rates
                        "ppr": float(metrics.get("ppr", 0.0)),
                        "pdr": float(metrics.get("pdr", 0.0)),
                        "drop_rate": float(metrics.get("drop_rate", 0.0)),
                        "reorder_rate": float(metrics.get("reorder_rate", 0.0)),
                        # Delay/age stats
                        "delay_mean": float(metrics.get("delay_mean", 0.0)),
                        "delay_p95": float(metrics.get("delay_p95", 0.0)),
                        "delay_max": float(metrics.get("delay_max", 0.0)),
                        "age_mean": float(metrics.get("age_mean", 0.0)),
                        "age_p95": float(metrics.get("age_p95", 0.0)),
                        "age_max": float(metrics.get("age_max", 0.0)),
                    }
        except Exception:
            pass

        return per_link

    def _compute_step_delta_comm(
        self,
        current: Dict[Any, Dict[str, Any]],
        previous: Dict[Any, Dict[str, Any]],
        count_keys: tuple = ("tx_count", "rx_count", "drop_count", "reorder_count"),
    ) -> Dict[Any, Dict[str, Any]]:
        """
        Compute step-level delta for comm stats.

        For count fields: delta = current - previous
        For rate/mean fields: keep current value (they're already episode-level averages)

        Returns a dict with same structure but "_delta" suffix added to count fields.
        """
        delta = {}
        for key, curr_metrics in current.items():
            prev_metrics = previous.get(key, {})
            delta_metrics = {}
            for field, value in curr_metrics.items():
                if field in count_keys:
                    # Compute delta for count fields
                    prev_val = prev_metrics.get(field, 0)
                    delta_metrics[f"{field}_delta"] = int(value) - int(prev_val)
                    delta_metrics[field] = value  # Also keep cumulative
                else:
                    delta_metrics[field] = value  # Keep as-is for rates/means
            delta[key] = delta_metrics
        return delta

    def _emit_step_risk_events(
        self,
        t: float,
        infos: Dict[str, Any],
        rl_data: List[Dict[str, Any]],
        pf_estimates: List[Dict[str, Any]] | None,
        true_states: Dict[int, Any],
    ) -> None:
        """
        Detect and emit Stage-4 risk events for case-driven analysis.

        Events emitted:
        - guard_trigger: Safety guard activated
        - cost_spike: Cost exceeds threshold (near_miss proxy, rule violation)
        - near_miss: Close approach detected
        """
        if self._staging is None or self._ep is None:
            return

        # Thresholds for event detection (configurable)
        NEAR_MISS_DISTANCE = float(self.cfg.get("near_miss_distance", 50.0))  # meters
        COST_SPIKE_NEAR = float(self.cfg.get("cost_spike_near", 0.5))
        COST_SPIKE_RULE = float(self.cfg.get("cost_spike_rule", 0.3))

        # ---- Guard trigger events ----
        for rd in rl_data:
            agent_id = rd.get("agent_id", "")
            guard_triggered = rd.get("guard_triggered", False)
            if guard_triggered:
                self._staging.emit_stage4_guard_trigger(
                    ep=self._ep,
                    t=t,
                    step_idx=self._step_idx,
                    agent_id=agent_id,
                    guard_margin=float(rd.get("guard_margin", 0.0)),
                    action_override=bool(rd.get("guard_action_override", False)),
                    original_action=rd.get("action_raw"),
                )

        # ---- Cost spike events (near-miss proxy, rule violation) ----
        for rd in rl_data:
            agent_id = rd.get("agent_id", "")
            c_near = float(rd.get("c_near", 0.0))
            c_rule = float(rd.get("c_rule", 0.0))

            if c_near >= COST_SPIKE_NEAR:
                self._staging.emit_stage4_cost_spike(
                    ep=self._ep,
                    t=t,
                    step_idx=self._step_idx,
                    agent_id=agent_id,
                    cost_type="c_near",
                    cost_value=c_near,
                    threshold=COST_SPIKE_NEAR,
                )

            if c_rule >= COST_SPIKE_RULE:
                self._staging.emit_stage4_rule_violation(
                    ep=self._ep,
                    t=t,
                    step_idx=self._step_idx,
                    agent_id=agent_id,
                    rule_type="colreg",
                    c_rule=c_rule,
                )

        # ---- Near-miss detection (distance-based) ----
        # Check pairwise distances between ships
        ship_ids = sorted(true_states.keys())
        for i, sid_i in enumerate(ship_ids):
            for sid_j in ship_ids[i + 1:]:
                ts_i = true_states.get(sid_i)
                ts_j = true_states.get(sid_j)
                if ts_i is None or ts_j is None:
                    continue

                # Compute distance
                import math
                xi, yi = float(getattr(ts_i, "x", 0)), float(getattr(ts_i, "y", 0))
                xj, yj = float(getattr(ts_j, "x", 0)), float(getattr(ts_j, "y", 0))
                dist = math.hypot(xi - xj, yi - yj)

                if dist < NEAR_MISS_DISTANCE:
                    self._staging.emit_stage4_near_miss(
                        ep=self._ep,
                        t=t,
                        step_idx=self._step_idx,
                        ship_pair=(sid_i, sid_j),
                        cpa_distance=dist,
                        tcpa=0.0,  # CPA is now (distance is current)
                        positions={sid_i: (xi, yi), sid_j: (xj, yj)},
                    )

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

    def _build_episode_summary(self, t_end: float, infos: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build comprehensive episode summary statistics.

        Includes:
          - Return sums (per-agent and global)
          - Cost sums (near, rule, coll, time)
          - PF error statistics (mean/p95/max for pos/vel/heading)
          - Episode-level comm stats
        """
        import numpy as np

        # ---- Return sums ----
        return_sum_per_agent = dict(self._ep_reward_sum)
        return_sum_total = sum(return_sum_per_agent.values())

        # ---- Cost sums ----
        cost_sums = dict(self._ep_cost_sum)

        # ---- PF error statistics ----
        pf_stats = {}
        if self._ep_pf_errors:
            pos_errors = [e["pos_error_mean"] for e in self._ep_pf_errors]
            vel_errors = [e["vel_error_mean"] for e in self._ep_pf_errors]
            heading_errors = [e["heading_error_mean"] for e in self._ep_pf_errors]

            # Position error stats
            pf_stats["pos_error_mean"] = float(np.mean(pos_errors))
            pf_stats["pos_error_p95"] = float(np.percentile(pos_errors, 95)) if len(pos_errors) >= 2 else float(np.max(pos_errors))
            pf_stats["pos_error_max"] = float(np.max(pos_errors))

            # Velocity error stats
            pf_stats["vel_error_mean"] = float(np.mean(vel_errors))
            pf_stats["vel_error_p95"] = float(np.percentile(vel_errors, 95)) if len(vel_errors) >= 2 else float(np.max(vel_errors))
            pf_stats["vel_error_max"] = float(np.max(vel_errors))

            # Heading error stats (already in radians)
            pf_stats["heading_error_mean"] = float(np.mean(heading_errors))
            pf_stats["heading_error_p95"] = float(np.percentile(heading_errors, 95)) if len(heading_errors) >= 2 else float(np.max(heading_errors))
            pf_stats["heading_error_max"] = float(np.max(heading_errors))
        else:
            # No PF data recorded
            pf_stats = {
                "pos_error_mean": 0.0, "pos_error_p95": 0.0, "pos_error_max": 0.0,
                "vel_error_mean": 0.0, "vel_error_p95": 0.0, "vel_error_max": 0.0,
                "heading_error_mean": 0.0, "heading_error_p95": 0.0, "heading_error_max": 0.0,
            }

        # ---- Episode-level comm stats (canonical names) ----
        comm_final_raw = self._ais.metrics_snapshot()
        comm_stats = self._clean_comm_stats(comm_final_raw)

        # ---- Per-ship comm stats (for diagnostics) ----
        per_ship_comm_final = self._get_per_ship_comm_stats() if self._record_per_ship_comm else None

        # ---- Per-link comm stats (rx_agent, tx_ship) for causality analysis ----
        per_link_comm_final = self._get_per_link_comm_stats() if self._record_per_link_comm else None

        # ---- Final Lagrangian state ----
        lagrangian_final = self._get_lagrangian_state()

        # ---- Timing ----
        duration = t_end - self._t0

        return {
            # Schema version for compatibility checking
            "schema_version": STAGING_SCHEMA_VERSION,

            # Timing
            "t0": self._t0,
            "t_end": t_end,
            "duration": duration,
            "total_steps": self._step_idx,
            "dt": float(self.cfg.get("dt", 0.5)),
            "seed": getattr(self, "_episode_seed", None),  # For reproducibility

            # Return sums
            "return_sum_total": return_sum_total,
            "return_sum_per_agent": return_sum_per_agent,

            # Cost sums
            "cost_sum_near": cost_sums.get("near", 0.0),
            "cost_sum_rule": cost_sums.get("rule", 0.0),
            "cost_sum_coll": cost_sums.get("coll", 0.0),
            "cost_sum_time": cost_sums.get("time", 0.0),
            "cost_sum_total": sum(cost_sums.values()),

            # PF tracking error statistics
            "pf_error": pf_stats,

            # Comm stats (episode-level)
            "comm_stats": comm_stats,
            "per_ship_comm": per_ship_comm_final,  # Per-ship diagnostics (if enabled)
            "per_link_comm": per_link_comm_final,  # Per-link (rx_agent, tx_ship) diagnostics

            # Final Lagrangian state (for constraint analysis)
            "lagrangian_final": lagrangian_final,
        }

    # ---------------- Reset with comprehensive staging ----------------

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        # Generate reproducible seed if None (critical for reproducibility)
        if seed is None:
            seed = int(time.time_ns() % (2**31)) ^ random.randint(0, 2**31 - 1)
        self._episode_seed = seed  # Save for recording

        obs, infos = self._core.reset(seed=seed, options=options)

        # episode identity (Phase 2)
        self._ep = self._ep_alloc.next()
        self._step_idx = 0

        # derive agent mappings from current obs keys (core env agents)
        agents = [a for a in obs.keys() if a not in ("__all__", "__common__")]
        self._refresh_mappings(agents)

        # init true states
        t0, true_states = self._get_true_states()
        self._t0 = t0  # Save for episode_init t0
        ship_ids = sorted(true_states.keys())
        self._last_true_states = true_states

        # Cache ship goals for PF observation building
        self._cache_ship_goals(true_states)

        # Reset debug counter for PF observations
        self._dbg_pf_obs_step = 0

        # Reset episode accumulators
        self._ep_reward_sum = {aid: 0.0 for aid in self._int_agents}
        self._ep_cost_sum = {"near": 0.0, "rule": 0.0, "coll": 0.0, "time": 0.0}
        self._ep_pf_errors = []
        self._ep_lagrangian_states = []  # Track lambda evolution

        # Reset delta tracking for per-step comm stats
        self._prev_per_ship_comm = {}
        self._prev_per_link_comm = {}

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

            # Build env config with t0, dt, seed, and config hashes
            env_config = self._build_env_config()
            env_config["seed"] = self._episode_seed  # Always record actual seed used
            env_config["t0"] = t0  # Time axis anchor
            env_config["schema_version"] = STAGING_SCHEMA_VERSION  # Version tracking
            env_config.update(self._compute_config_hashes())

            # Build ship init states
            ship_init_states = self._build_ship_init_states(true_states)

            # Get complete PF config (configuration contract)
            pf_config = self._get_pf_config()

            # Get initial Lagrangian state
            lagrangian_init = self._get_lagrangian_state()

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
                    "pf_config": pf_config,  # Full PF configuration contract
                    "lagrangian_init": lagrangian_init,  # Initial dual variables
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

        # ---- Build observations from PF estimates ----
        # Replace core env observations with PF-based observations
        # This is the key fix: agents see PF-estimated neighbor positions, not true positions
        pf_obs = self._build_pf_observations(t, true_states)

        # Verify PF observations - fall back to core obs if PF fails
        _dbg_step = getattr(self, '_dbg_pf_obs_step', 0)
        if pf_obs and all(aid in pf_obs for aid in obs.keys()):
            # Debug: compare observation values between core and PF
            if _dbg_step < 2:
                for aid in list(obs.keys())[:1]:
                    core_obs = obs[aid]
                    pf_obs_arr = pf_obs[aid]
                    # Print first 8 dims (ego features) - should now match!
                    print(f"[PF_OBS_CMP] step={_dbg_step} {aid} v_max={self._v_max}")
                    print(f"[PF_OBS_CMP] step={_dbg_step} {aid} CORE ego[0:8]={[f'{x:.3f}' for x in core_obs[:8]]}")
                    print(f"[PF_OBS_CMP] step={_dbg_step} {aid}   PF ego[0:8]={[f'{x:.3f}' for x in pf_obs_arr[:8]]}")
                    # Check if ego features match (they should now!)
                    ego_diff = sum(abs(core_obs[i] - pf_obs_arr[i]) for i in range(8))
                    print(f"[PF_OBS_CMP] step={_dbg_step} {aid} ego_diff_sum={ego_diff:.6f} {'OK' if ego_diff < 0.01 else 'MISMATCH!'}")

                    # Compare neighbor features (index 8 onwards, 11 dims per neighbor)
                    K = self._K_neighbors
                    for k in range(K):
                        nei_start = 8 + k * 11
                        nei_end = nei_start + 11
                        if nei_end <= len(core_obs):
                            core_nei = core_obs[nei_start:nei_end]
                            pf_nei = pf_obs_arr[nei_start:nei_end]
                            nei_diff = sum(abs(core_nei[i] - pf_nei[i]) for i in range(11))
                            # Print neighbor comparison (first 8 = geometry, last 3 = u_stale, u_silence, valid)
                            print(f"[PF_OBS_CMP] step={_dbg_step} {aid} nei[{k}] CORE={[f'{x:.2f}' for x in core_nei]}")
                            print(f"[PF_OBS_CMP] step={_dbg_step} {aid} nei[{k}]   PF={[f'{x:.2f}' for x in pf_nei]} diff={nei_diff:.3f}")
            obs = pf_obs
        else:
            # Fallback: use core env observations if PF failed
            if _dbg_step < 5:
                print(f"[PF_OBS_DBG] WARNING: PF obs incomplete, using core obs. "
                      f"pf_keys={list(pf_obs.keys()) if pf_obs else None} "
                      f"core_keys={list(obs.keys())}")

        # ---- Accumulate episode stats ----
        for agent_id in self._int_agents:
            self._ep_reward_sum[agent_id] = self._ep_reward_sum.get(agent_id, 0.0) + float(rews.get(agent_id, 0.0))
            info = infos.get(agent_id, {})
            self._ep_cost_sum["near"] += float(info.get("c_near", 0.0))
            self._ep_cost_sum["rule"] += float(info.get("c_rule", 0.0))
            self._ep_cost_sum["coll"] += float(info.get("c_coll", 0.0))
            self._ep_cost_sum["time"] += float(info.get("c_time", 0.0))

        # ---- Comprehensive staging: step record ----
        pf_estimates = None
        lagrangian_state = None
        if self._staging is not None and self._record_step_data:
            # Get current Lagrangian state (lambdas, guard, etc.)
            lagrangian_state = self._get_lagrangian_state()

            # Build ship states
            ship_states = self._build_step_ship_states(true_states, infos)

            # Build RL data (with Lagrangian state)
            rl_data = self._build_step_rl_data(actions, rews, infos, lagrangian_state)

            # Get comm stats (canonical fields only)
            comm_stats_raw = self._ais.metrics_snapshot()
            comm_stats = self._clean_comm_stats(comm_stats_raw)

            # Get per-ship comm stats (optional, for diagnostics)
            per_ship_comm = None
            per_ship_comm_delta = None
            if self._record_per_ship_comm:
                per_ship_comm = self._get_per_ship_comm_stats()
                # Compute step-level delta (tx/rx/drop/reorder counts)
                per_ship_comm_delta = self._compute_step_delta_comm(
                    per_ship_comm, self._prev_per_ship_comm
                )
                self._prev_per_ship_comm = per_ship_comm  # Update for next step

            # Get per-link comm stats (optional, for causality analysis)
            per_link_comm = None
            per_link_comm_delta = None
            if self._record_per_link_comm:
                per_link_comm = self._get_per_link_comm_stats()
                # Compute step-level delta (tx/rx/drop/reorder counts)
                per_link_comm_delta = self._compute_step_delta_comm(
                    per_link_comm, self._prev_per_link_comm
                )
                self._prev_per_link_comm = per_link_comm  # Update for next step

            # Get PF estimates (optional, can be expensive)
            if self._record_pf_estimates:
                pf_estimates = self._track_mgr.get_pf_estimates_for_staging(t, true_states)

            # Build extra dict for diagnostics
            extra = None
            if per_ship_comm or per_link_comm:
                extra = {}
                if per_ship_comm:
                    # Include both cumulative and delta stats
                    extra["per_ship_comm"] = per_ship_comm_delta  # Delta includes both cumulative + _delta fields
                if per_link_comm:
                    extra["per_link_comm"] = per_link_comm_delta  # Delta includes both cumulative + _delta fields

            # Emit comprehensive step record
            self._staging.emit_stage3_step(
                ep=self._ep,
                step_idx=self._step_idx,
                t_sim=t,
                ship_states=ship_states,
                rl_data=rl_data,
                comm_stats=comm_stats,
                pf_estimates=pf_estimates,
                extra=extra,
            )

            # =========================================================================
            # Stage-4 risk events detection and emission
            # =========================================================================
            self._emit_step_risk_events(t, infos, rl_data, pf_estimates, true_states)

        # Accumulate PF errors for episode summary
        if pf_estimates:
            step_pf_err = {
                "pos_error_mean": sum(p.get("pos_error", 0.0) for p in pf_estimates) / len(pf_estimates),
                "vel_error_mean": sum(p.get("vel_error", 0.0) for p in pf_estimates) / len(pf_estimates),
                "heading_error_mean": sum(abs(p.get("heading_error", 0.0)) for p in pf_estimates) / len(pf_estimates),
            }
            self._ep_pf_errors.append(step_pf_err)

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

            # Build comprehensive final stats
            final_stats = self._build_episode_summary(t, infos)

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
