"""
staging/validate.py — Validation for Stage-3/4 Recording System

Validates:
  1. Directory structure exists
  2. All JSONL files are parseable
  3. All records have required fields
  4. Stage-4 events map to Stage-3 episodes
  5. No duplicate episode_uids in episode records

Record Types Accepted:
  Stage-3: shard_header, episode_init, episode, step, comm_stats, pf_estimates, episode_end
  Stage-4: shard_header, event
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Set


# Valid record types for each stage
VALID_STAGE3_TYPES: Set[str] = {
    "shard_header",     # Shard metadata
    "episode_init",     # Comprehensive episode init (new)
    "episode",          # Legacy episode init (backward compat)
    "step",             # Per-step comprehensive data
    "comm_stats",       # Per-step comm stats (legacy)
    "pf_estimates",     # Per-step PF estimates
    "episode_end",      # Episode end summary
}

VALID_STAGE4_TYPES: Set[str] = {
    "shard_header",     # Shard metadata
    "event",            # All events (start, end, collision, arrival, etc.)
}

# Types that count as episode records (for uniqueness check)
EPISODE_RECORD_TYPES: Set[str] = {"episode_init", "episode"}


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Iterate over JSONL file, yielding parsed records."""
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise RuntimeError(f"[validate] json parse error: {path}:{ln}: {e}") from e


def validate_out_dir(out_dir: str, mode: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Validate staging output directory structure and data integrity.

    Args:
        out_dir: Base output directory
        mode: Mode subdirectory (train/eval)
        verbose: Print detailed progress

    Returns:
        Dictionary of statistics

    Raises:
        FileNotFoundError: If required directories/files missing
        RuntimeError: If data validation fails
    """
    out = Path(out_dir)
    stage3_dir = out / "stage3" / mode
    stage4_dir = out / "stage4" / mode

    if not stage3_dir.is_dir():
        raise FileNotFoundError(f"[validate] stage3 dir not found: {stage3_dir}")
    if not stage4_dir.is_dir():
        raise FileNotFoundError(f"[validate] stage4 dir not found: {stage4_dir}")

    s3_files = sorted(stage3_dir.glob("episodes.*.jsonl"))
    s4_files = sorted(stage4_dir.glob("events.*.jsonl"))

    if not s3_files:
        raise FileNotFoundError(f"[validate] no stage3 episode shards found in: {stage3_dir}")
    if not s4_files:
        raise FileNotFoundError(f"[validate] no stage4 event shards found in: {stage4_dir}")

    # ---- Stage-3 validation ----
    s3_stats = {
        "episode_records": 0,
        "step_records": 0,
        "comm_stats_records": 0,
        "pf_estimates_records": 0,
        "episode_end_records": 0,
        "shard_header_records": 0,
        # New: count embedded data in step records
        "steps_with_comm_stats": 0,
        "steps_with_pf_estimates": 0,
    }
    s3_uid_counts: Dict[str, int] = {}

    for fp in s3_files:
        if verbose:
            print(f"  Validating: {fp}")
        for rec in _iter_jsonl(fp):
            rec_type = rec.get("type")

            # Validate record type
            if rec_type not in VALID_STAGE3_TYPES:
                raise RuntimeError(f"[validate] invalid stage3 record type in {fp}: {rec_type}")

            # Skip header for episode_uid check
            if rec_type == "shard_header":
                s3_stats["shard_header_records"] += 1
                continue

            # All non-header records must have episode_uid
            uid = rec.get("episode_uid", None)
            if not isinstance(uid, str) or not uid:
                raise RuntimeError(f"[validate] stage3 record missing episode_uid in {fp}: type={rec_type}")

            # Count by type
            if rec_type in EPISODE_RECORD_TYPES:
                s3_uid_counts[uid] = s3_uid_counts.get(uid, 0) + 1
                s3_stats["episode_records"] += 1
            elif rec_type == "step":
                s3_stats["step_records"] += 1
                # Count embedded data in step records
                if rec.get("comm_stats"):
                    s3_stats["steps_with_comm_stats"] += 1
                if rec.get("pf_estimates"):
                    s3_stats["steps_with_pf_estimates"] += 1
            elif rec_type == "comm_stats":
                s3_stats["comm_stats_records"] += 1
            elif rec_type == "pf_estimates":
                s3_stats["pf_estimates_records"] += 1
            elif rec_type == "episode_end":
                s3_stats["episode_end_records"] += 1

    # Check for duplicate episode_uids
    dup_uids = [u for u, c in s3_uid_counts.items() if c != 1]
    if dup_uids:
        raise RuntimeError(f"[validate] stage3 duplicate episode_uid(s): {dup_uids[:10]} (showing first 10)")

    # ---- Stage-4 validation ----
    s4_stats = {
        "event_records": 0,
        "shard_header_records": 0,
    }
    missing_uids = []

    for fp in s4_files:
        if verbose:
            print(f"  Validating: {fp}")
        for rec in _iter_jsonl(fp):
            rec_type = rec.get("type")

            # Validate record type
            if rec_type not in VALID_STAGE4_TYPES:
                raise RuntimeError(f"[validate] invalid stage4 record type in {fp}: {rec_type}")

            # Skip header for episode_uid check
            if rec_type == "shard_header":
                s4_stats["shard_header_records"] += 1
                continue

            # All non-header records must have episode_uid
            uid = rec.get("episode_uid", None)
            if not isinstance(uid, str) or not uid:
                raise RuntimeError(f"[validate] stage4 record missing episode_uid in {fp}")

            # Check that episode_uid exists in stage3
            if uid not in s3_uid_counts:
                missing_uids.append(uid)

            s4_stats["event_records"] += 1

    # Report missing UIDs
    if missing_uids:
        uniq = sorted(set(missing_uids))
        raise RuntimeError(f"[validate] stage4 episode_uid missing in stage3: {uniq[:10]} (showing first 10)")

    # Final checks
    if s3_stats["episode_records"] == 0:
        raise RuntimeError("[validate] stage3 has zero episode records (smoke did not produce episodes)")
    if s4_stats["event_records"] == 0:
        raise RuntimeError("[validate] stage4 has zero event records (env did not emit any events)")

    # Build summary
    stats = {
        "stage3_dir": str(stage3_dir),
        "stage4_dir": str(stage4_dir),
        "stage3_files": len(s3_files),
        "stage4_files": len(s4_files),
        "stage3_episode_records": s3_stats["episode_records"],
        "stage3_unique_episode_uids": len(s3_uid_counts),
        "stage3_step_records": s3_stats["step_records"],
        "stage3_comm_stats_records": s3_stats["comm_stats_records"],
        "stage3_pf_estimates_records": s3_stats["pf_estimates_records"],
        "stage3_steps_with_comm_stats": s3_stats["steps_with_comm_stats"],
        "stage3_steps_with_pf_estimates": s3_stats["steps_with_pf_estimates"],
        "stage3_episode_end_records": s3_stats["episode_end_records"],
        "stage4_event_records": s4_stats["event_records"],
    }
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description="Validate staging out_dir (Phase-1/2 acceptance).")
    p.add_argument("--out", required=True, help="Output directory to validate")
    p.add_argument("--mode", default="train", help="Mode subdirectory (train/eval)")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = p.parse_args()

    stats = validate_out_dir(args.out, args.mode, verbose=args.verbose)
    print("[validate] OK")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
