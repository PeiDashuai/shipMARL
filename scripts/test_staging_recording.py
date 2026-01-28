#!/usr/bin/env python3
"""
Simple staging recording test script.

This script runs the MiniShipAISCommsEnv with random actions to test
the complete data recording chain: AIS → PF → RL staging.

Usage:
    python scripts/test_staging_recording.py --out TEST_STAGING --episodes 3
"""

import argparse
import shutil
import sys
from pathlib import Path


def run_test(out_dir: str, mode: str, ais_cfg: str, episodes: int, max_steps: int):
    """Run staging recording test."""

    # Clean output directory
    out = Path(out_dir)
    if out.exists():
        print(f"[test] Removing existing output: {out}")
        shutil.rmtree(out)

    # Import after path setup
    from miniship.envs.miniship_ais_comms_env import MiniShipAISCommsEnv
    import uuid

    run_uuid = uuid.uuid4().hex
    print(f"[test] Starting run: {run_uuid}")
    print(f"[test] Output: {out_dir}")
    print(f"[test] Episodes: {episodes}, Max steps per episode: {max_steps}")

    cfg = {
        "N": 2,
        "dt": 0.5,
        "T_max": float(max_steps * 0.5),  # Convert to simulation time
        "mode": mode,
        "out_dir": out_dir,
        "run_uuid": run_uuid,
        "worker_index": 0,
        "vector_index": 0,
        "ais_cfg_path": ais_cfg,
        "staging_enable": True,
        "staging_record_steps": True,
        "staging_record_pf": True,
    }

    env = MiniShipAISCommsEnv(cfg)

    total_steps = 0
    for ep in range(episodes):
        obs, infos = env.reset()
        ep_uid = infos.get("__common__", {}).get("episode_uid", "unknown")
        print(f"\n[test] Episode {ep+1}/{episodes} started (uid={ep_uid[:8]}...)")

        step = 0
        while step < max_steps:
            # Random actions
            action_dict = {aid: env.action_space(aid).sample() for aid in env.agents}
            obs, rewards, terms, truncs, infos = env.step(action_dict)
            step += 1
            total_steps += 1

            # Check termination
            done = bool(terms.get("__all__", False)) or bool(truncs.get("__all__", False))
            if not done:
                done = all(terms.get(a, False) or truncs.get(a, False) for a in env.agents)

            if done:
                term_reason = "unknown"
                for info in infos.values():
                    if isinstance(info, dict) and "term_reason" in info:
                        term_reason = info["term_reason"]
                        break
                print(f"[test] Episode {ep+1} ended at step {step}: {term_reason}")
                break

        if step >= max_steps:
            print(f"[test] Episode {ep+1} reached max steps: {step}")

    env.close()
    print(f"\n[test] Completed: {episodes} episodes, {total_steps} total steps")

    # Validate output
    print("\n[test] Validating output...")
    from staging.validate import validate_out_dir
    try:
        stats = validate_out_dir(out_dir, mode, verbose=True)
        print("\n[test] Validation PASSED!")
        print("  Summary:")
        for k, v in sorted(stats.items()):
            print(f"    {k}: {v}")
        return True
    except Exception as e:
        print(f"\n[test] Validation FAILED: {e}")
        return False


def main():
    p = argparse.ArgumentParser(description="Test staging recording with 2-ship scenario")
    p.add_argument("--out", default="TEST_STAGING", help="Output directory")
    p.add_argument("--mode", default="train", help="Mode (train/eval)")
    p.add_argument("--ais-cfg", default="ais_comms/ais_config_pf_perfect.yaml",
                   help="AIS config path")
    p.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    p.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    args = p.parse_args()

    success = run_test(
        out_dir=args.out,
        mode=args.mode,
        ais_cfg=args.ais_cfg,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
