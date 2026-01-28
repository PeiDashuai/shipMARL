from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Any, Dict


def _extract_episode_uid(infos: Any) -> str:
    """Extract episode_uid from env returned infos.

    Phase-2 contract: env.reset() must expose episode_uid.

    Accepted locations:
      - infos.get("__common__", {}).get("episode_uid")
      - any per-agent info dict containing "episode_uid"
    """
    if not isinstance(infos, dict):
        raise RuntimeError(f"[smoke] env.reset infos must be dict, got {type(infos)}")

    common = infos.get("__common__", None)
    if isinstance(common, dict):
        uid = common.get("episode_uid", None)
        if isinstance(uid, str) and uid:
            return uid

    for _, v in infos.items():
        if isinstance(v, dict):
            uid = v.get("episode_uid", None)
            if isinstance(uid, str) and uid:
                return uid

    raise RuntimeError("[smoke] missing episode_uid in infos (Phase-2 contract not satisfied)")


def _ensure_out_dir_clean(out_dir: str) -> None:
    out = Path(out_dir)
    if out.exists():
        # Strict: do not reuse existing out_dir (avoids silent mixing of shards)
        # User should delete it explicitly.
        raise FileExistsError(
            f"[smoke] out_dir already exists: {out_dir}. "
            f"Delete it (or choose a new --out) to avoid mixing shards."
        )
    out.mkdir(parents=True, exist_ok=False)


def _run_one_episode(env) -> None:
    obs, infos = env.reset()
    _ = _extract_episode_uid(infos)  # hard requirement

    # sample actions until done
    while True:
        # build action_dict for parallel env
        agents = []
        if hasattr(env, "agents") and isinstance(env.agents, (list, tuple)):
            agents = list(env.agents)
        elif hasattr(env, "possible_agents") and isinstance(env.possible_agents, (list, tuple)):
            agents = list(env.possible_agents)

        action_dict: Dict[str, Any] = {}
        for aid in agents:
            try:
                space = env.action_space(aid)
                action_dict[aid] = space.sample()
            except Exception:
                # If env does not expose per-agent spaces, it's a real interface violation.
                raise RuntimeError(f"[smoke] cannot sample action for aid={aid}")

        obs, rewards, terms, truncs, infos = env.step(action_dict)

        # Check for episode termination (either __all__ or all agents done)
        done_all = False
        if isinstance(terms, dict) and bool(terms.get("__all__", False)):
            done_all = True
        if isinstance(truncs, dict) and bool(truncs.get("__all__", False)):
            done_all = True
        if not done_all and agents:
            # Check if all agents are terminated or truncated
            done_all = all(
                bool(terms.get(a, False)) or bool(truncs.get(a, False))
                for a in agents
            )
        if done_all:
            break


def smoke(out_dir: str, mode: str, ais_cfg_path: str | None, max_episodes: int = 1) -> None:
    """Run a minimal env rollout and assert stage3/stage4 shards are produced by env."""
    _ensure_out_dir_clean(out_dir)

    run_uuid = uuid.uuid4().hex

    # Let env create and manage its own staging recorder
    from miniship.envs.miniship_ais_comms_env import MiniShipAISCommsEnv

    cfg: Dict[str, Any] = {
        "N": 2,
        "dt": 0.5,
        "T_max": 50.0,  # Short timeout for smoke test (100 steps max)
        "mode": mode,
        "out_dir": out_dir,
        "run_uuid": run_uuid,
        "worker_index": 0,
        "vector_index": 0,
        "ais_cfg_path": ais_cfg_path,
        "staging_enable": True,
        "staging_record_steps": True,
        "staging_record_pf": True,
    }

    env = MiniShipAISCommsEnv(cfg)
    try:
        for _ in range(max_episodes):
            _run_one_episode(env)
    finally:
        try:
            env.close()
        except Exception:
            pass

    # Check output existence (Phase-1 acceptance)
    from pathlib import Path
    stage3_dir = Path(out_dir) / "stage3" / mode
    stage4_dir = Path(out_dir) / "stage4" / mode

    stage3_files = list(stage3_dir.glob("episodes.*.jsonl"))
    stage4_files = list(stage4_dir.glob("events.*.jsonl"))

    if not stage3_files:
        raise RuntimeError(f"[smoke] no stage3 shards found in: {stage3_dir}")
    if not stage4_files:
        raise RuntimeError(f"[smoke] no stage4 shards found in: {stage4_dir}")

    for fp in stage3_files:
        if fp.stat().st_size == 0:
            raise RuntimeError(f"[smoke] stage3 shard empty: {fp}")
    for fp in stage4_files:
        if fp.stat().st_size == 0:
            raise RuntimeError(f"[smoke] stage4 shard empty: {fp}")

    print("[smoke] OK")
    print(f"  stage3: {stage3_files[0]}")
    print(f"  stage4: {stage4_files[0]}")


def main() -> int:
    p = argparse.ArgumentParser(description="Staging smoke test (Phase-1/2).")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", default="train")
    p.add_argument("--ais-cfg-path", default=None)
    p.add_argument("--max-episodes", type=int, default=1)
    args = p.parse_args()

    smoke(
        out_dir=args.out,
        mode=args.mode,
        ais_cfg_path=args.ais_cfg_path,
        max_episodes=args.max_episodes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
