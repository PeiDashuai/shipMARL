"""
staging/recorder.py — Comprehensive Stage-3/4 Recording System

Design Goals:
  1. Complete traceability: AIS params → PF estimates → RL decisions
  2. Multi-worker alignment: deterministic episode_uid across shards
  3. Episode-level analysis: all data joinable by (run_uuid, episode_uid, step_idx)

Record Types (Stage-3):
  - episode_init: Full episode initialization (AIS params, env config, ship states)
  - step: Per-step RL data (actions, rewards, states, costs)
  - comm_stats: Per-step AIS communication statistics
  - pf_estimate: Per-step PF tracking results vs ground truth
  - trajectory: Ship trajectory snapshots

Record Types (Stage-4):
  - event: Episode lifecycle events (start, end, collision, arrival)

Schema Contract:
  All records include: {type, ts_wall, episode_uid, run_uuid, mode, worker_index, vector_index}
  Step-level records add: {step_idx, t_sim}
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os
import time


@dataclass(frozen=True)
class StagingIdentity:
    """Identity contract for Stage-3/4 recording.

    Required fields for multi-worker alignment:
      - run_uuid: unique run id (same across all workers in a run)
      - mode: "train" | "eval"
      - out_dir: base output directory
      - worker_index: RLlib worker index (0 for driver)
      - vector_index: RLlib vector env index
    """
    run_uuid: str
    mode: str
    out_dir: str
    worker_index: int
    vector_index: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_uuid": self.run_uuid,
            "mode": self.mode,
            "out_dir": self.out_dir,
            "worker_index": self.worker_index,
            "vector_index": self.vector_index,
        }

    def shard_key(self) -> str:
        """Unique shard identifier for this worker/vector."""
        return f"w{self.worker_index}.v{self.vector_index}"


# Alias for env-side imports
RunIdentity = StagingIdentity


@dataclass
class ShipStateSnapshot:
    """Snapshot of a single ship's state at a given time."""
    ship_id: int
    agent_id: str
    x: float
    y: float
    psi: float  # heading (rad)
    v: float    # speed (m/s)
    vx: float   # velocity x component
    vy: float   # velocity y component
    goal_x: float
    goal_y: float
    goal_dist: float
    reached: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepRLData:
    """RL data for a single agent at a single step."""
    agent_id: str
    # Actions
    action_raw: List[float]      # raw RL output [-1, 1]
    dpsi_cmd: float              # heading change command (rad)
    v_cmd: float                 # speed command (m/s)
    v_target: float              # actual target after caps
    # Rewards
    r_task: float                # task reward
    r_shaped: float              # shaped reward (after Lagrangian)
    r_total: float               # total reward sent to RL
    # Costs
    c_near: float                # proximity cost
    c_rule: float                # COLREGs violation cost
    c_coll: float                # collision cost
    c_time: float                # timeout cost
    # State info
    risk: float                  # risk metric
    guard_triggered: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PFEstimate:
    """Particle filter estimate for a single ship."""
    ship_id: int
    # Estimate
    est_x: float
    est_y: float
    est_vx: float
    est_vy: float
    est_psi: float
    est_sog: float
    # Ground truth
    true_x: float
    true_y: float
    true_vx: float
    true_vy: float
    true_psi: float
    true_sog: float
    # Errors
    pos_error: float             # Euclidean position error
    vel_error: float             # velocity magnitude error
    heading_error: float         # heading error (rad)
    # Track quality
    track_age: float             # time since last update
    num_particles: int = 0
    eff_particles: float = 0.0   # effective sample size

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StageRecorder:
    """Comprehensive Stage-3/4 recorder with strict guarantees.

    Design constraints:
      - Never overwrite existing shard files
      - Duplicate episode_uid in stage3 is an error
      - All records are atomic (flush + fsync)
      - Support episode-level and step-level analysis
    """

    def __init__(self, ident: StagingIdentity, run_metadata: Optional[Dict[str, Any]] = None):
        """
        Args:
            ident: Staging identity (run_uuid, mode, worker, vector)
            run_metadata: Optional run-level metadata for reproducibility:
                - git_commit: Git commit hash
                - git_dirty: Whether working tree has uncommitted changes
                - train_args: Training script arguments
                - ais_cfg_hash: Hash of AIS config file
                - pf_cfg_hash: Hash of PF config
        """
        if not isinstance(ident, StagingIdentity):
            raise TypeError(f"ident must be StagingIdentity, got {type(ident)}")
        self._validate_identity(ident)

        self.ident = ident
        self._pid = os.getpid()
        self._run_metadata = run_metadata or {}

        # Setup directories
        out = Path(ident.out_dir)
        self._stage3_dir = out / "stage3" / ident.mode
        self._stage4_dir = out / "stage4" / ident.mode
        self._stage3_dir.mkdir(parents=True, exist_ok=True)
        self._stage4_dir.mkdir(parents=True, exist_ok=True)

        # Shard naming: deterministic per worker/vector
        shard = f"run{ident.run_uuid}.{ident.shard_key()}.pid{self._pid}"
        self._stage3_path = self._stage3_dir / f"episodes.{shard}.jsonl"
        self._stage4_path = self._stage4_dir / f"events.{shard}.jsonl"

        # Check if files exist - allow reopening if same process created them
        self._reopened = False
        s3_exists = self._stage3_path.exists()
        s4_exists = self._stage4_path.exists()

        if s3_exists or s4_exists:
            # Check if this is our own file (same PID in filename means same process)
            # Allow reopening - the header was already written
            self._reopened = True
            # Load existing episode_uids to prevent duplicates
            self._seen_episode_uids: set[str] = self._load_existing_episode_uids()
        else:
            self._seen_episode_uids: set[str] = set()

        self._current_episode_uid: Optional[str] = None
        self._current_episode_idx: Optional[int] = None

        # Write metadata header only if not reopening
        if not self._reopened:
            self._write_header()

    def _load_existing_episode_uids(self) -> set[str]:
        """Load episode_uids from existing stage3 file to prevent duplicates."""
        uids: set[str] = set()
        if not self._stage3_path.exists():
            return uids
        try:
            with self._stage3_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        rec_type = rec.get("type", "")
                        if rec_type in ("episode_init", "episode"):
                            uid = rec.get("episode_uid")
                            if uid:
                                uids.add(uid)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        return uids

    @staticmethod
    def _validate_identity(ident: StagingIdentity) -> None:
        if not ident.run_uuid or not isinstance(ident.run_uuid, str):
            raise ValueError("ident.run_uuid must be non-empty str")
        if not ident.mode or not isinstance(ident.mode, str):
            raise ValueError("ident.mode must be non-empty str")
        if not ident.out_dir or not isinstance(ident.out_dir, str):
            raise ValueError("ident.out_dir must be non-empty str")
        if ident.worker_index < 0:
            raise ValueError("ident.worker_index must be >= 0")
        if ident.vector_index < 0:
            raise ValueError("ident.vector_index must be >= 0")

    def _write_header(self) -> None:
        """Write shard metadata as first record with run-level info."""
        # Compute git info
        git_info = self._get_git_info()

        header = {
            "type": "shard_header",
            "ts_wall": time.time(),
            "pid": self._pid,
            **self.ident.as_dict(),
            "schema_version": "2.0",
            # Run-level metadata for reproducibility
            "git_commit": git_info.get("commit", "unknown"),
            "git_dirty": git_info.get("dirty", False),
            "git_branch": git_info.get("branch", "unknown"),
            # User-provided run metadata (train args, config hashes, etc.)
            **self._run_metadata,
        }
        self._append_jsonl(self._stage3_path, header)
        self._append_jsonl(self._stage4_path, header)

    def _get_git_info(self) -> Dict[str, Any]:
        """Get git commit, dirty flag, and branch for reproducibility."""
        import subprocess
        info = {"commit": "unknown", "dirty": False, "branch": "unknown"}
        try:
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info["commit"] = result.stdout.strip()[:12]

            # Check if working tree is dirty
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info["dirty"] = len(result.stdout.strip()) > 0

            # Get current branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info["branch"] = result.stdout.strip()
        except Exception:
            pass
        return info

    @property
    def stage3_path(self) -> Path:
        return self._stage3_path

    @property
    def stage4_path(self) -> Path:
        return self._stage4_path

    def _base_record(self, record_type: str, episode_uid: str) -> Dict[str, Any]:
        """Create base record with common fields."""
        return {
            "type": record_type,
            "ts_wall": time.time(),
            "episode_uid": episode_uid,
            **self.ident.as_dict(),
        }

    def _append_jsonl(self, path: Path, obj: Dict[str, Any]) -> None:
        """Atomic append with fsync."""
        if not isinstance(obj, dict):
            raise TypeError(f"jsonl record must be dict, got {type(obj)}")
        line = json.dumps(obj, ensure_ascii=False, default=self._json_default)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def _json_default(obj: Any) -> Any:
        """Handle non-serializable types."""
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if hasattr(obj, "as_dict"):
            return obj.as_dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    # =========================================================================
    # Stage-3: Episode-level records
    # =========================================================================

    def emit_stage3_episode(
        self,
        episode_uid: str,
        payload: Dict[str, Any],
        episode_idx: int | None = None,
    ) -> None:
        """Write episode initialization record (backward compatible)."""
        self.emit_stage3_episode_init(
            episode_uid=episode_uid,
            episode_idx=episode_idx or 0,
            env_config={},
            ais_params=payload,
            ship_init_states=[],
        )

    def emit_stage3_episode_init(
        self,
        episode_uid: str,
        episode_idx: int,
        env_config: Dict[str, Any],
        ais_params: Dict[str, Any],
        ship_init_states: List[Dict[str, Any]],
        extra: Dict[str, Any] | None = None,
    ) -> None:
        """Write comprehensive episode initialization record.

        Args:
            episode_uid: Unique episode identifier
            episode_idx: Episode index within this worker/vector
            env_config: Environment configuration (N, dt, T_max, spawn params, etc.)
            ais_params: Complete AIS parameters (episode_params + effective)
            ship_init_states: Initial state of each ship
            extra: Additional metadata
        """
        if not episode_uid or not isinstance(episode_uid, str):
            raise ValueError("episode_uid must be non-empty str")
        if episode_uid in self._seen_episode_uids:
            raise ValueError(f"[staging] duplicate episode_uid: {episode_uid}")

        self._seen_episode_uids.add(episode_uid)
        self._current_episode_uid = episode_uid
        self._current_episode_idx = episode_idx

        rec = self._base_record("episode_init", episode_uid)
        rec.update({
            "episode_idx": episode_idx,
            "env_config": env_config,
            "ais_params": ais_params,
            "ship_init_states": ship_init_states,
        })
        if extra:
            rec["extra"] = extra

        self._append_jsonl(self._stage3_path, rec)

    # =========================================================================
    # Stage-3: Step-level records
    # =========================================================================

    def emit_stage3_step(
        self,
        episode_uid: str,
        step_idx: int,
        t_sim: float,
        ship_states: List[Dict[str, Any]],
        rl_data: List[Dict[str, Any]],
        comm_stats: Dict[str, Any] | None = None,
        pf_estimates: List[Dict[str, Any]] | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        """Write comprehensive per-step record.

        This is the main recording method that combines all step-level data.

        Args:
            episode_uid: Episode identifier
            step_idx: Step index within episode
            t_sim: Simulation time
            ship_states: Current state of each ship
            rl_data: RL data (actions, rewards, costs) for each agent
            comm_stats: AIS communication statistics
            pf_estimates: PF tracking estimates (if available)
            extra: Additional per-step data
        """
        if not episode_uid or not isinstance(episode_uid, str):
            raise ValueError("episode_uid must be non-empty str")

        rec = self._base_record("step", episode_uid)
        rec.update({
            "step_idx": step_idx,
            "t_sim": t_sim,
            "ship_states": ship_states,
            "rl_data": rl_data,
        })
        if comm_stats:
            rec["comm_stats"] = comm_stats
        if pf_estimates:
            rec["pf_estimates"] = pf_estimates
        if extra:
            rec["extra"] = extra

        self._append_jsonl(self._stage3_path, rec)

    def emit_stage3_comm_stats(
        self,
        episode_uid: str,
        step_idx: int,
        payload: Dict[str, Any],
    ) -> None:
        """Write communication stats record (backward compatible)."""
        if not episode_uid or not isinstance(episode_uid, str):
            raise ValueError("episode_uid must be non-empty str")
        if not isinstance(payload, dict):
            raise TypeError("payload must be dict")

        rec = self._base_record("comm_stats", episode_uid)
        rec.update({
            "step_idx": step_idx,
            "payload": payload,
        })
        self._append_jsonl(self._stage3_path, rec)

    def emit_stage3_pf_estimates(
        self,
        episode_uid: str,
        step_idx: int,
        t_sim: float,
        estimates: List[Dict[str, Any]],
    ) -> None:
        """Write PF tracking estimates for all ships."""
        if not episode_uid or not isinstance(episode_uid, str):
            raise ValueError("episode_uid must be non-empty str")

        rec = self._base_record("pf_estimates", episode_uid)
        rec.update({
            "step_idx": step_idx,
            "t_sim": t_sim,
            "estimates": estimates,
        })
        self._append_jsonl(self._stage3_path, rec)

    def emit_stage3_episode_end(
        self,
        episode_uid: str,
        episode_idx: int,
        t_end: float,
        total_steps: int,
        term_reason: str,
        final_stats: Dict[str, Any],
        per_agent_stats: List[Dict[str, Any]] | None = None,
    ) -> None:
        """Write episode end summary."""
        if not episode_uid or not isinstance(episode_uid, str):
            raise ValueError("episode_uid must be non-empty str")

        rec = self._base_record("episode_end", episode_uid)
        rec.update({
            "episode_idx": episode_idx,
            "t_end": t_end,
            "total_steps": total_steps,
            "term_reason": term_reason,
            "final_stats": final_stats,
        })
        if per_agent_stats:
            rec["per_agent_stats"] = per_agent_stats

        self._append_jsonl(self._stage3_path, rec)

    # =========================================================================
    # Stage-4: Event records
    # =========================================================================

    def emit_stage4_event(
        self,
        episode_uid: str | None = None,
        event: Dict[str, Any] | str | None = None,
        *,
        stage3_episode_uid: str | None = None,
        stage3_episode_idx: int | None = None,
        payload: Dict[str, Any] | None = None,
    ) -> None:
        """Write a stage4 event.

        Supports two calling conventions:
          1. emit_stage4_event(episode_uid, event_dict)
          2. emit_stage4_event(stage3_episode_uid=..., event=..., payload=...)
        """
        uid = episode_uid or stage3_episode_uid
        if not uid or not isinstance(uid, str):
            raise ValueError("episode_uid must be non-empty str")

        # Resolve event content
        if isinstance(event, dict):
            event_data = event
        elif isinstance(event, str):
            event_data = {"event_type": event}
            if payload and isinstance(payload, dict):
                event_data.update(payload)
        elif event is None and payload is not None:
            event_data = payload
        else:
            raise TypeError("event must be dict or str, or provide payload")

        rec = self._base_record("event", uid)
        rec["event"] = event_data
        if stage3_episode_idx is not None:
            rec["episode_idx"] = stage3_episode_idx

        self._append_jsonl(self._stage4_path, rec)

    def emit_stage4_episode_start(
        self,
        episode_uid: str,
        episode_idx: int,
        t0: float,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        """Emit episode start event."""
        payload = {"event_type": "episode_start", "t0": t0}
        if extra:
            payload.update(extra)
        self.emit_stage4_event(
            stage3_episode_uid=episode_uid,
            stage3_episode_idx=episode_idx,
            payload=payload,
        )

    def emit_stage4_episode_end(
        self,
        episode_uid: str,
        episode_idx: int,
        t_end: float,
        term_reason: str,
        stats: Dict[str, Any],
    ) -> None:
        """Emit episode end event."""
        payload = {
            "event_type": "episode_end",
            "t_end": t_end,
            "term_reason": term_reason,
            **stats,
        }
        self.emit_stage4_event(
            stage3_episode_uid=episode_uid,
            stage3_episode_idx=episode_idx,
            payload=payload,
        )

    def emit_stage4_collision(
        self,
        episode_uid: str,
        episode_idx: int,
        t: float,
        ship_pair: tuple[int, int],
        positions: Dict[int, tuple[float, float]],
    ) -> None:
        """Emit collision event."""
        payload = {
            "event_type": "collision",
            "t": t,
            "ship_pair": list(ship_pair),
            "positions": {str(k): list(v) for k, v in positions.items()},
        }
        self.emit_stage4_event(
            stage3_episode_uid=episode_uid,
            stage3_episode_idx=episode_idx,
            payload=payload,
        )

    def emit_stage4_arrival(
        self,
        episode_uid: str,
        episode_idx: int,
        t: float,
        ship_id: int,
        position: tuple[float, float],
        goal: tuple[float, float],
    ) -> None:
        """Emit ship arrival event."""
        payload = {
            "event_type": "arrival",
            "t": t,
            "ship_id": ship_id,
            "position": list(position),
            "goal": list(goal),
        }
        self.emit_stage4_event(
            stage3_episode_uid=episode_uid,
            stage3_episode_idx=episode_idx,
            payload=payload,
        )

    def close(self) -> None:
        """Close recorder (no-op, kept for lifecycle symmetry)."""
        pass


# Backward-compatible alias
StagingRecorder = StageRecorder
