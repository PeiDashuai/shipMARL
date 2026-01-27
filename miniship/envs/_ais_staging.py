from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from miniship.envs._ais_identity import EpisodeIdentity, RunIdentity

from staging.recorder import RunIdentity as StagingRunIdentity
from staging.recorder import StageRecorder


@dataclass(slots=True)
class StagingSink:
    """
    Env-side staging interface (Phase 1/2).

    This wrapper holds the StageRecorder (writer) and exposes strict emit APIs.
    """
    run: RunIdentity
    rec: StageRecorder

    @staticmethod
    def from_run(run: RunIdentity) -> "StagingSink":
        srun = StagingRunIdentity(
            run_uuid=run.run_uuid,
            out_dir=run.out_dir,
            mode=run.mode,
            worker_index=run.worker_index,
            vector_index=run.vector_index,
        )
        return StagingSink(run=run, rec=StageRecorder(srun))

    def close(self) -> None:
        self.rec.close()

    def emit_stage3_episode(self, ep: EpisodeIdentity, payload: Dict[str, Any]) -> None:
        self.rec.emit_stage3_episode(episode_uid=ep.episode_uid, episode_idx=ep.episode_idx, payload=payload)

    def emit_stage3_comm_stats(self, ep: EpisodeIdentity, step_idx: int, payload: Dict[str, Any]) -> None:
        self.rec.emit_stage3_comm_stats(episode_uid=ep.episode_uid, step_idx=step_idx, payload=payload)

    def emit_stage4_event(self, ep: EpisodeIdentity, event: str, payload: Dict[str, Any]) -> None:
        self.rec.emit_stage4_event(
            stage3_episode_uid=ep.episode_uid,
            stage3_episode_idx=ep.episode_idx,
            event=event,
            payload=payload,
        )
