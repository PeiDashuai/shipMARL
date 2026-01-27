from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from ais_comms.ais_comms import AISCommsSim
from ais_comms.datatypes import AgentId, ShipId, TrueState
from ais_comms.track_manager_pf import AISTrackManagerPF

from miniship.envs._ais_identity import EpisodeIdAllocator, EpisodeIdentity, RunIdentity
from miniship.envs._ais_staging import StagingSink
from miniship.wrappers.lagrangian_pz_env import MiniShipLagrangianParallelEnv


class MiniShipAISCommsEnv:
    """
    MiniShip PettingZoo-parallel env wrapper with:
      - AIS communication simulation (AISCommsSim)
      - per-agent PF tracking (AISTrackManagerPF)
      - staging writer ownership (Phase 1)
      - identity contract (Phase 2)

    This file intentionally keeps ONLY env orchestration logic.
    AIS/PF algorithms and stage writing live elsewhere.
    """

    metadata = {"name": "MiniShipAISCommsEnv"}

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = dict(cfg)

        # ---- Phase 2 identity (fail-fast; no fallback) ----
        self._run = RunIdentity.from_cfg(self.cfg)

        # ---- Underlying core env (physics/reward/termination) ----
        self._core = MiniShipLagrangianParallelEnv(self.cfg)

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
            self._staging = StagingSink.from_run(self._run)

        # episode identity allocator
        self._ep_alloc = EpisodeIdAllocator(self._run)
        self._ep: Optional[EpisodeIdentity] = None
        self._step_idx: int = 0

        # mappings
        self._int_agents: List[AgentId] = []
        self._ship_of_agent: Dict[AgentId, ShipId] = {}
        self._agent_of_ship: Dict[ShipId, AgentId] = {}

    # ---------------- PettingZoo parallel API delegation ----------------

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
        return t, out

    # ---------------- Reset/Step with AIS+PF+Staging ----------------

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, infos = self._core.reset(seed=seed, options=options)

        # episode identity (Phase 2)
        self._ep = self._ep_alloc.next()
        self._step_idx = 0

        # derive agent mappings from current obs keys (core env agents)
        agents = [a for a in obs.keys() if a not in ("__all__", "__common__")]
        self._refresh_mappings(agents)

        # init true states
        t0, true_states = self._get_true_states()
        ship_ids = sorted(true_states.keys())

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

        # staging: stage3 episode record (Phase 1)
        if self._staging is not None:
            self._staging.emit_stage3_episode(self._ep, payload={
                "seed": seed,
                "N": int(self.cfg.get("N", len(ship_ids))),
                "ais_cfg_path": str(self.cfg.get("ais_cfg_path")),
            })
            self._staging.emit_stage4_event(self._ep, event="episode_start", payload={"t0": t0})

        infos = dict(infos) if isinstance(infos, dict) else {}
        infos["__common__"] = {
            **(infos.get("__common__", {}) if isinstance(infos.get("__common__", {}), dict) else {}),
            "episode_uid": self._ep.episode_uid,
            "episode_idx": self._ep.episode_idx,
        }
        return obs, infos

    def step(self, actions: Dict[str, Any]):
        obs, rews, terms, truncs, infos = self._core.step(actions)

        if self._ep is None:
            raise RuntimeError("[MiniShipAISCommsEnv] BUG: step called before reset")

        # Update AIS and PF based on current truth
        t, true_states = self._get_true_states()
        ready = self._ais.step(t=t, true_states=true_states)
        self._track_mgr.ingest(t=t, ready=ready, true_states=true_states)

        if self._staging is not None:
            self._staging.emit_stage3_comm_stats(self._ep, step_idx=self._step_idx, payload=self._ais.metrics_snapshot())
        self._step_idx += 1

        done_all = bool(terms.get("__all__", False)) or bool(truncs.get("__all__", False))
        if done_all and self._staging is not None:
            self._staging.emit_stage4_event(self._ep, event="episode_end", payload={
                "t_end": t,
                "core_done": True,
                "ais_metrics_final": self._ais.metrics_snapshot(),
            })

        infos = dict(infos) if isinstance(infos, dict) else {}
        infos["__common__"] = {
            **(infos.get("__common__", {}) if isinstance(infos.get("__common__", {}), dict) else {}),
            "episode_uid": self._ep.episode_uid,
            "episode_idx": self._ep.episode_idx,
            "step_idx": self._step_idx,
        }
        return obs, rews, terms, truncs, infos
