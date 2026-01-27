from __future__ import annotations

import argparse

from staging.smoke import smoke
from staging.validate import validate_out_dir


def main() -> int:
    p = argparse.ArgumentParser(description="Run staging smoke then validate.")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", default="train")
    p.add_argument("--ais-cfg-path", default=None)
    p.add_argument("--max-episodes", type=int, default=1)

    args = p.parse_args()

    # 1) smoke: produce stage3/stage4 shards
    smoke(
        out_dir=args.out,
        mode=args.mode,
        ais_cfg_path=args.ais_cfg_path,
        max_episodes=args.max_episodes,
    )

    # 2) validate: identity join stage3<->stage4
    stats = validate_out_dir(args.out, args.mode)

    print("\n[acceptance] PASS")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
