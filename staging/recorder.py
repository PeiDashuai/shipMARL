from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable
import json
import os
import time


@dataclass(frozen=True)
class StagingIdentity:
    """Identity contract for Stage-3/4 recording (Phase-2).

    Required:
      - run_uuid: unique run id
      - mode: "train" | "eval" | ... (used under stage3/<mode>/)
      - out_dir: base output directory
      - worker_index: RLlib worker index
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


# Alias: env-side code may import RunIdentity (same semantics)
RunIdentity = StagingIdentity


class StageRecorder:
    """Strict jsonl recorder for stage3 + stage4.

    Design constraints (NO silent fallback):
      - Never overwrite existing shard files.
      - Duplicate episode_uid in stage3 is an error.
      - stage4 events must include episode_uid.
    """

    def __init__(self, ident: StagingIdentity):
        if not isinstance(ident, StagingIdentity):
            raise TypeError(f"ident must be StagingIdentity, got {type(ident)}")
        if not ident.run_uuid or not isinstance(ident.run_uuid, str):
            raise ValueError("ident.run_uuid must be non-empty str")
        if not ident.mode or not isinstance(ident.mode, str):
            raise ValueError("ident.mode must be non-empty str")
        if not ident.out_dir or not isinstance(ident.out_dir, str):
            raise ValueError("ident.out_dir must be non-empty str")

        self.ident = ident
        self._pid = os.getpid()

        out = Path(ident.out_dir)
        self._stage3_dir = out / "stage3" / ident.mode
        self._stage4_dir = out / "stage4" / ident.mode

        self._stage3_dir.mkdir(parents=True, exist_ok=True)
        self._stage4_dir.mkdir(parents=True, exist_ok=True)

        shard = f"run{ident.run_uuid}.w{ident.worker_index}.v{ident.vector_index}.pid{self._pid}"
        self._stage3_path = self._stage3_dir / f"episodes.{shard}.jsonl"
        self._stage4_path = self._stage4_dir / f"events.{shard}.jsonl"

        # Strict: no overwrite
        if self._stage3_path.exists():
            raise FileExistsError(f"[staging] stage3 shard already exists: {self._stage3_path}")
        if self._stage4_path.exists():
            raise FileExistsError(f"[staging] stage4 shard already exists: {self._stage4_path}")

        self._seen_episode_uids: set[str] = set()

    @property
    def stage3_path(self) -> Path:
        return self._stage3_path

    @property
    def stage4_path(self) -> Path:
        return self._stage4_path

    def _append_jsonl(self, path: Path, obj: Dict[str, Any]) -> None:
        if not isinstance(obj, dict):
            raise TypeError(f"jsonl record must be dict, got {type(obj)}")
        line = json.dumps(obj, ensure_ascii=False)
        # O_APPEND is atomic on POSIX for small writes; we also flush+fsync for safety.
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def emit_stage3_episode(
        self,
        episode_uid: str,
        payload: Dict[str, Any],
        episode_idx: int | None = None,
    ) -> None:
        """Write exactly one stage3 record per episode_uid."""
        if not episode_uid or not isinstance(episode_uid, str):
            raise ValueError("episode_uid must be non-empty str")
        if episode_uid in self._seen_episode_uids:
            raise ValueError(f"[staging] duplicate episode_uid in stage3: {episode_uid}")
        self._seen_episode_uids.add(episode_uid)

        rec = {
            "type": "episode",
            "ts_wall": time.time(),
            "episode_uid": episode_uid,
            **self.ident.as_dict(),
        }
        if episode_idx is not None:
            rec["episode_idx"] = episode_idx
        if payload:
            if not isinstance(payload, dict):
                raise TypeError("payload must be dict")
            rec["payload"] = payload

        self._append_jsonl(self._stage3_path, rec)

    def emit_stage3_comm_stats(
        self,
        episode_uid: str,
        step_idx: int,
        payload: Dict[str, Any],
    ) -> None:
        """Write a stage3 communication stats record for a step."""
        if not episode_uid or not isinstance(episode_uid, str):
            raise ValueError("episode_uid must be non-empty str")
        if not isinstance(payload, dict):
            raise TypeError("payload must be dict")

        rec = {
            "type": "comm_stats",
            "ts_wall": time.time(),
            "episode_uid": episode_uid,
            "step_idx": step_idx,
            **self.ident.as_dict(),
            "payload": payload,
        }
        self._append_jsonl(self._stage3_path, rec)

    def emit_stage4_event(
        self,
        episode_uid: str | None = None,
        event: Dict[str, Any] | str | None = None,
        *,
        stage3_episode_uid: str | None = None,
        stage3_episode_idx: int | None = None,
        payload: Dict[str, Any] | None = None,
    ) -> None:
        """Write a stage4 event; must contain episode_uid.

        Supports two calling conventions:
          1. emit_stage4_event(episode_uid, event_dict)  # original
          2. emit_stage4_event(stage3_episode_uid=..., stage3_episode_idx=..., event=..., payload=...)  # env-side
        """
        # Resolve episode_uid (support both positional and keyword)
        uid = episode_uid or stage3_episode_uid
        if not uid or not isinstance(uid, str):
            raise ValueError("episode_uid (or stage3_episode_uid) must be non-empty str")

        # Resolve event content
        if isinstance(event, dict):
            event_data = event
        elif isinstance(event, str):
            event_data = {"event_type": event}
            if payload and isinstance(payload, dict):
                event_data["payload"] = payload
        elif event is None and payload is not None:
            event_data = payload
        else:
            raise TypeError("event must be dict or str, or provide payload")

        rec = {
            "type": "event",
            "ts_wall": time.time(),
            "episode_uid": uid,
            **self.ident.as_dict(),
            "event": event_data,
        }
        if stage3_episode_idx is not None:
            rec["episode_idx"] = stage3_episode_idx

        self._append_jsonl(self._stage4_path, rec)

    def close(self) -> None:
        # No persistent handles; method kept for symmetry/explicit lifecycle.
        return


# Backward-compatible alias (some code imports StagingRecorder)
StagingRecorder = StageRecorder
