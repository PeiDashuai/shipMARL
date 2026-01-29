"""
_ais_staging.py — Environment-side staging interface

This module provides a thin wrapper around StageRecorder for use in
MiniShipAISCommsEnv. It handles the translation between env-side identity
(RunIdentity, EpisodeIdentity) and staging-side identity (StagingIdentity).

Design:
  - StagingSink holds the recorder and current episode context
  - All emit methods accept EpisodeIdentity for consistent tracking
  - Supports comprehensive data collection for analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from miniship.envs._ais_identity import EpisodeIdentity, RunIdentity

from staging.recorder import (
    RunIdentity as StagingRunIdentity,
    StageRecorder,
    ShipStateSnapshot,
    StepRLData,
    PFEstimate,
)


@dataclass
class StagingSink:
    """
    Env-side staging interface for comprehensive data recording.

    This wrapper:
      1. Translates env-side identities to staging identities
      2. Provides type-safe emit methods
      3. Manages episode lifecycle
    """
    run: RunIdentity
    rec: StageRecorder

    @staticmethod
    def from_run(run: RunIdentity, run_metadata: Optional[Dict[str, Any]] = None) -> "StagingSink":
        """Create StagingSink from env-side RunIdentity.

        Args:
            run: Environment-side run identity
            run_metadata: Optional run-level metadata for reproducibility:
                - train_args: Training script arguments (lr, batch_size, etc.)
                - ais_cfg_hash: Hash of AIS config file
                - pf_cfg_hash: Hash of PF config
                (git info is auto-captured by StageRecorder)
        """
        srun = StagingRunIdentity(
            run_uuid=run.run_uuid,
            out_dir=run.out_dir,
            mode=run.mode,
            worker_index=run.worker_index,
            vector_index=run.vector_index,
        )
        return StagingSink(run=run, rec=StageRecorder(srun, run_metadata=run_metadata))

    def close(self) -> None:
        """Close the underlying recorder."""
        self.rec.close()

    # =========================================================================
    # Episode-level records
    # =========================================================================

    def emit_stage3_episode(self, ep: EpisodeIdentity, payload: Dict[str, Any]) -> None:
        """Backward-compatible episode init (minimal payload)."""
        self.rec.emit_stage3_episode(
            episode_uid=ep.episode_uid,
            episode_idx=ep.episode_idx,
            payload=payload,
        )

    def emit_stage3_episode_init(
        self,
        ep: EpisodeIdentity,
        env_config: Dict[str, Any],
        ais_params: Dict[str, Any],
        ship_init_states: List[Dict[str, Any]],
        extra: Dict[str, Any] | None = None,
    ) -> None:
        """
        Comprehensive episode initialization record.

        Args:
            ep: Episode identity
            env_config: Full environment configuration
            ais_params: Complete AIS parameters (episode_params + effective)
            ship_init_states: Initial state of each ship
            extra: Additional metadata
        """
        self.rec.emit_stage3_episode_init(
            episode_uid=ep.episode_uid,
            episode_idx=ep.episode_idx,
            env_config=env_config,
            ais_params=ais_params,
            ship_init_states=ship_init_states,
            extra=extra,
        )

    def emit_stage3_episode_end(
        self,
        ep: EpisodeIdentity,
        t_end: float,
        total_steps: int,
        term_reason: str,
        final_stats: Dict[str, Any],
        per_agent_stats: List[Dict[str, Any]] | None = None,
    ) -> None:
        """Episode end summary record."""
        self.rec.emit_stage3_episode_end(
            episode_uid=ep.episode_uid,
            episode_idx=ep.episode_idx,
            t_end=t_end,
            total_steps=total_steps,
            term_reason=term_reason,
            final_stats=final_stats,
            per_agent_stats=per_agent_stats,
        )

    # =========================================================================
    # Step-level records
    # =========================================================================

    def emit_stage3_step(
        self,
        ep: EpisodeIdentity,
        step_idx: int,
        t_sim: float,
        ship_states: List[Dict[str, Any]],
        rl_data: List[Dict[str, Any]],
        comm_stats: Dict[str, Any] | None = None,
        pf_estimates: List[Dict[str, Any]] | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        """
        Comprehensive per-step record combining all data.

        This is the main recording method for step-level analysis.
        """
        self.rec.emit_stage3_step(
            episode_uid=ep.episode_uid,
            step_idx=step_idx,
            t_sim=t_sim,
            ship_states=ship_states,
            rl_data=rl_data,
            comm_stats=comm_stats,
            pf_estimates=pf_estimates,
            extra=extra,
        )

    def emit_stage3_comm_stats(
        self,
        ep: EpisodeIdentity,
        step_idx: int,
        payload: Dict[str, Any],
    ) -> None:
        """Backward-compatible comm stats record."""
        self.rec.emit_stage3_comm_stats(
            episode_uid=ep.episode_uid,
            step_idx=step_idx,
            payload=payload,
        )

    def emit_stage3_pf_estimates(
        self,
        ep: EpisodeIdentity,
        step_idx: int,
        t_sim: float,
        estimates: List[Dict[str, Any]],
    ) -> None:
        """PF tracking estimates record."""
        self.rec.emit_stage3_pf_estimates(
            episode_uid=ep.episode_uid,
            step_idx=step_idx,
            t_sim=t_sim,
            estimates=estimates,
        )

    # =========================================================================
    # Stage-4 events
    # =========================================================================

    def emit_stage4_event(
        self,
        ep: EpisodeIdentity,
        event: str,
        payload: Dict[str, Any],
    ) -> None:
        """Generic stage4 event."""
        self.rec.emit_stage4_event(
            stage3_episode_uid=ep.episode_uid,
            stage3_episode_idx=ep.episode_idx,
            event=event,
            payload=payload,
        )

    def emit_stage4_episode_start(
        self,
        ep: EpisodeIdentity,
        t0: float,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        """Episode start event."""
        self.rec.emit_stage4_episode_start(
            episode_uid=ep.episode_uid,
            episode_idx=ep.episode_idx,
            t0=t0,
            extra=extra,
        )

    def emit_stage4_episode_end(
        self,
        ep: EpisodeIdentity,
        t_end: float,
        term_reason: str,
        stats: Dict[str, Any],
    ) -> None:
        """Episode end event."""
        self.rec.emit_stage4_episode_end(
            episode_uid=ep.episode_uid,
            episode_idx=ep.episode_idx,
            t_end=t_end,
            term_reason=term_reason,
            stats=stats,
        )

    def emit_stage4_collision(
        self,
        ep: EpisodeIdentity,
        t: float,
        ship_pair: tuple[int, int],
        positions: Dict[int, tuple[float, float]],
    ) -> None:
        """Collision event."""
        self.rec.emit_stage4_collision(
            episode_uid=ep.episode_uid,
            episode_idx=ep.episode_idx,
            t=t,
            ship_pair=ship_pair,
            positions=positions,
        )

    def emit_stage4_arrival(
        self,
        ep: EpisodeIdentity,
        t: float,
        ship_id: int,
        position: tuple[float, float],
        goal: tuple[float, float],
    ) -> None:
        """Ship arrival event."""
        self.rec.emit_stage4_arrival(
            episode_uid=ep.episode_uid,
            episode_idx=ep.episode_idx,
            t=t,
            ship_id=ship_id,
            position=position,
            goal=goal,
        )


# Helper functions for building record data
def build_ship_state_dict(
    ship_id: int,
    agent_id: str,
    x: float,
    y: float,
    psi: float,
    v: float,
    goal_x: float,
    goal_y: float,
    goal_dist: float,
    reached: bool = False,
    vx: float | None = None,
    vy: float | None = None,
) -> Dict[str, Any]:
    """Build ship state dictionary for recording."""
    import math
    if vx is None:
        vx = v * math.cos(psi)
    if vy is None:
        vy = v * math.sin(psi)
    return {
        "ship_id": ship_id,
        "agent_id": agent_id,
        "x": x,
        "y": y,
        "psi": psi,
        "v": v,
        "vx": vx,
        "vy": vy,
        "goal_x": goal_x,
        "goal_y": goal_y,
        "goal_dist": goal_dist,
        "reached": reached,
    }


def build_rl_data_dict(
    agent_id: str,
    action_raw: List[float],
    dpsi_cmd: float,
    v_cmd: float,
    v_target: float,
    r_task: float,
    r_shaped: float,
    r_total: float,
    c_near: float,
    c_rule: float,
    c_coll: float,
    c_time: float,
    risk: float,
    guard_triggered: bool = False,
    # Lagrangian dual variables (for policy analysis)
    lambda_near: float = 0.0,
    lambda_rule: float = 0.0,
    lambda_coll: float = 0.0,
    lambda_time: float = 0.0,
    dual_version: int = 0,
    # Guard controller details
    guard_margin: float = 0.0,
    guard_action_override: bool = False,
) -> Dict[str, Any]:
    """Build RL data dictionary for recording."""
    return {
        "agent_id": agent_id,
        "action_raw": action_raw,
        "dpsi_cmd": dpsi_cmd,
        "v_cmd": v_cmd,
        "v_target": v_target,
        "r_task": r_task,
        "r_shaped": r_shaped,
        "r_total": r_total,
        "c_near": c_near,
        "c_rule": c_rule,
        "c_coll": c_coll,
        "c_time": c_time,
        "risk": risk,
        "guard_triggered": guard_triggered,
        # Lagrangian state (enables analyzing constraint-policy interaction)
        "lambda_near": lambda_near,
        "lambda_rule": lambda_rule,
        "lambda_coll": lambda_coll,
        "lambda_time": lambda_time,
        "dual_version": dual_version,
        # Guard details (enables analyzing safety override patterns)
        "guard_margin": guard_margin,
        "guard_action_override": guard_action_override,
    }


# ===========================================================================
# PF Estimate Schema Version (for data contract tracking)
# ===========================================================================
PF_ESTIMATE_SCHEMA_VERSION = "2.0"

# Schema changelog:
#   v1.0: Basic estimate + error fields
#   v2.0: Added PF diagnostic fields for divergence/freeze/reorder detection
#         - Track timestamps: last_reported_ts, last_update_ts, last_arrival_ts
#         - Last measurement: last_meas_x, last_meas_y
#         - PF health: neff, resampled, collapsed, track_valid
#         - Measurement noise: sigma_pos, sigma_sog, sigma_yaw, meas_age
#         - Derived indicators: info_age, silence


def build_pf_estimate_dict(
    ship_id: int,
    est_x: float,
    est_y: float,
    est_vx: float,
    est_vy: float,
    est_psi: float,
    true_x: float,
    true_y: float,
    true_vx: float,
    true_vy: float,
    true_psi: float,
    track_age: float = 0.0,
    num_particles: int = 0,
    eff_particles: float = 0.0,
    # ===========================================================================
    # PF diagnostic fields (v2.0) for divergence/freeze/reorder detection
    # These fields enable causality analysis: Comm quality → PF accuracy → RL
    # ===========================================================================
    # Track timestamps
    last_reported_ts: float | None = None,   # last ts_rep fused (message timestamp)
    last_update_ts: float | None = None,     # last t_env fused (PF axis)
    last_arrival_ts: float | None = None,    # last ts_arr (network arrival)
    # Last fused measurement (projected to t_env)
    last_meas_x: float | None = None,
    last_meas_y: float | None = None,
    # PF health indicators
    neff: float | None = None,               # effective sample size after last update
    resampled: bool | None = None,           # was resampling triggered?
    collapsed: bool | None = None,           # did weights collapse?
    track_valid: bool = True,                # is track within staleness threshold?
    # Measurement noise used in last update
    sigma_pos: float | None = None,          # position noise std (m)
    sigma_sog: float | None = None,          # SOG noise std (m/s)
    sigma_yaw: float | None = None,          # yaw noise std (rad)
    meas_age: float | None = None,           # age used in last update (s)
    # Derived indicators
    info_age: float | None = None,           # t_env - last_reported_ts (s)
    silence: float | None = None,            # t_env - last_update_ts (s)
) -> Dict[str, Any]:
    """
    Build PF estimate dictionary for recording.

    v2.0 adds diagnostic fields for post-hoc causality analysis:
      - Track timestamps for timing analysis
      - PF health indicators (neff, resampled, collapsed) for divergence detection
      - Measurement noise parameters for uncertainty quantification
      - info_age/silence for freeze/staleness detection
    """
    import math
    est_sog = math.hypot(est_vx, est_vy)
    true_sog = math.hypot(true_vx, true_vy)
    pos_error = math.hypot(est_x - true_x, est_y - true_y)
    vel_error = abs(est_sog - true_sog)

    # Heading error (wrapped to [-pi, pi])
    heading_error = est_psi - true_psi
    while heading_error > math.pi:
        heading_error -= 2 * math.pi
    while heading_error < -math.pi:
        heading_error += 2 * math.pi

    return {
        "schema_version": PF_ESTIMATE_SCHEMA_VERSION,
        "ship_id": ship_id,
        # Estimate
        "est_x": est_x,
        "est_y": est_y,
        "est_vx": est_vx,
        "est_vy": est_vy,
        "est_psi": est_psi,
        "est_sog": est_sog,
        # Truth
        "true_x": true_x,
        "true_y": true_y,
        "true_vx": true_vx,
        "true_vy": true_vy,
        "true_psi": true_psi,
        "true_sog": true_sog,
        # Errors
        "pos_error": pos_error,
        "vel_error": vel_error,
        "heading_error": heading_error,
        # Track quality (legacy)
        "track_age": track_age,
        "num_particles": num_particles,
        "eff_particles": eff_particles,
        # ===========================================================================
        # PF diagnostic fields (v2.0)
        # ===========================================================================
        # Track timestamps
        "last_reported_ts": last_reported_ts,
        "last_update_ts": last_update_ts,
        "last_arrival_ts": last_arrival_ts,
        # Last fused measurement
        "last_meas_x": last_meas_x,
        "last_meas_y": last_meas_y,
        # PF health indicators
        "neff": neff,
        "resampled": resampled,
        "collapsed": collapsed,
        "track_valid": track_valid,
        # Measurement noise
        "sigma_pos": sigma_pos,
        "sigma_sog": sigma_sog,
        "sigma_yaw": sigma_yaw,
        "meas_age": meas_age,
        # Derived indicators
        "info_age": info_age,
        "silence": silence,
    }
