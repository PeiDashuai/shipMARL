from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise RuntimeError(f"[validate] json parse error: {path}:{ln}: {e}") from e


def validate_out_dir(out_dir: str, mode: str) -> Dict[str, Any]:
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

    # stage3: episode_uid -> count (only count "episode" records, skip "comm_stats")
    s3_count = 0
    s3_comm_stats_count = 0
    s3_uid_counts: Dict[str, int] = {}
    valid_s3_types = {"episode", "comm_stats"}
    for fp in s3_files:
        for rec in _iter_jsonl(fp):
            rec_type = rec.get("type")
            if rec_type not in valid_s3_types:
                raise RuntimeError(f"[validate] invalid stage3 record type in {fp}: {rec_type}")
            uid = rec.get("episode_uid", None)
            if not isinstance(uid, str) or not uid:
                raise RuntimeError(f"[validate] stage3 record missing episode_uid in {fp}")
            if rec_type == "episode":
                s3_uid_counts[uid] = s3_uid_counts.get(uid, 0) + 1
                s3_count += 1
            elif rec_type == "comm_stats":
                s3_comm_stats_count += 1

    dup_uids = [u for u, c in s3_uid_counts.items() if c != 1]
    if dup_uids:
        raise RuntimeError(f"[validate] stage3 duplicate episode_uid(s): {dup_uids[:10]} (showing first 10)")

    # stage4: every event must map to exactly one stage3 record
    s4_count = 0
    missing = []
    for fp in s4_files:
        for rec in _iter_jsonl(fp):
            if rec.get("type") != "event":
                raise RuntimeError(f"[validate] invalid stage4 record type in {fp}: {rec.get('type')}")
            uid = rec.get("episode_uid", None)
            if not isinstance(uid, str) or not uid:
                raise RuntimeError(f"[validate] stage4 record missing episode_uid in {fp}")
            if uid not in s3_uid_counts:
                missing.append(uid)
            s4_count += 1

    if missing:
        uniq = sorted(set(missing))
        raise RuntimeError(f"[validate] stage4 episode_uid missing in stage3: {uniq[:10]} (showing first 10)")

    if s3_count == 0:
        raise RuntimeError("[validate] stage3 has zero records (smoke did not produce episodes)")
    if s4_count == 0:
        raise RuntimeError("[validate] stage4 has zero records (env did not emit any events)")

    stats = {
        "stage3_episode_records": s3_count,
        "stage3_unique_episode_uids": len(s3_uid_counts),
        "stage4_event_records": s4_count,
        "stage3_dir": str(stage3_dir),
        "stage4_dir": str(stage4_dir),
    }
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description="Validate staging out_dir (Phase-1/2 acceptance).")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", default="train")
    args = p.parse_args()

    stats = validate_out_dir(args.out, args.mode)
    print("[validate] OK")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
