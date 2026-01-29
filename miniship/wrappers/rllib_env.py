# miniship/wrappers/rllib_env.py
from __future__ import annotations

from typing import Any, Dict, Tuple, Set, Optional
import os
import re
import copy
import math

import numpy as np
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from miniship.wrappers.lagrangian_pz_env import MiniShipLagrangianParallelEnv
from miniship.envs.miniship_ais_comms_env import MiniShipAISCommsEnv


"""
================================================================================
MiniShip RLlib Wrapper — Phase 1–2 CONTRACT ENFORCER (NO STAGE WRITES HERE)
================================================================================

This wrapper is responsible for:
  1) RLlib context injection:
      - worker_index, vector_index must be injected from EnvContext (attr-first).
  2) Identity contract alignment (Phase 2):
      - run_uuid / mode / out_dir should be provided by the TRAINING SCRIPT.
      - this file will normalize out_dir from "out" if needed, and enforce stability.
  3) Probe/Rollout separation:
      - "space-probe" env (used only to infer spaces) must NOT write any stage files.
        We force staging_enable=False for the temporary env created in register_miniship_env().
  4) Key canonicalization:
      - obs/reward/terminated/truncated/infos keys are remapped to a canonical agent-id set.
      - infos are normalized to per-agent dicts and may include "__common__".

Hard rule:
  - Do NOT create/write stage3/stage4 files here.
  - All staging writes must happen inside env via staging.recorder (Phase 1).
"""


# -----------------------------
# small utilities
# -----------------------------
def _to_int(v: Any, default: int) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _is_truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def _safe_deepcopy(d: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return copy.deepcopy(d)
    except Exception:
        return dict(d)


# -----------------------------
# Wrapper
# -----------------------------
class MiniShipMultiAgentWrapper(MultiAgentEnv):
    """
    Adapt a PettingZoo ParallelEnv-like env to RLlib MultiAgentEnv.

    Key requirements:
      - agent set is derived from returned dict keys (obs/rewards/terms/truncs)
      - ensure "__all__" exists in terminated/truncated
      - normalize infos to {aid: dict, ..., "__common__": dict?}
      - apply RLlib parallel context: worker_index/vector_index into cfg
      - do NOT write stage files in this wrapper
    """

    def __init__(self, cfg: Dict[str, Any], use_pf_ais_env: bool):
        super().__init__()

        # Make cfg a plain dict (EnvContext is dict-like but may carry attrs).
        self._cfg: Dict[str, Any] = dict(cfg) if hasattr(cfg, "items") else dict(cfg)

        # Debug knobs (default OFF; keep this file quiet in multi-worker runs)
        self._dbg_step_print_n = _to_int(self._cfg.get("debug_step_print_n", 0), 0)
        self._dbg_step_print_every = max(1, _to_int(self._cfg.get("debug_step_print_every", 1), 1))
        self._dbg_step_tag = str(self._cfg.get("debug_step_print_tag", "miniship"))

        self._dbg_obs_state_check = bool(self._cfg.get("debug_obs_state_check", False))
        self._dbg_obs_state_check_n = _to_int(self._cfg.get("debug_obs_state_check_n", 20), 20)
        self._dbg_obs_state_check_every = max(1, _to_int(self._cfg.get("debug_obs_state_check_every", 1), 1))
        self._dbg_obs_state_check_tol = float(self._cfg.get("debug_obs_state_check_tol", 1e-4))
        self._dbg_obs_state_check_assert = bool(self._cfg.get("debug_obs_state_check_assert", False))
        self._dbg_obschk_i = 0

        # Reward debug (default OFF)
        self._rdbg_enable = bool(self._cfg.get("debug_rllib_reward", False)) or _is_truthy(os.environ.get("MINISHIP_RLLIB_REWDBG", "0"))
        self._rdbg_n = _to_int(self._cfg.get("debug_rllib_reward_n", 0), 0)
        self._rdbg_every = max(1, _to_int(self._cfg.get("debug_rllib_reward_every", 1), 1))
        self._rdbg_ep_total = 0.0
        self._rdbg_ep_total_firstn = 0.0
        self._rdbg_ep_total_by_agent: Dict[str, float] = {}
        self._rdbg_ep_total_firstn_by_agent: Dict[str, float] = {}

        # Dual snapshot plumbing
        self._dual_snapshot: Optional[Dict[str, Any]] = None

        # Episode/step counters (wrapper-local)
        self._ep_i = 0
        self._step_i = 0
        self._terminal_printed = False

        # Validate/normalize identity contract (Phase 2)
        # NOTE: training script must inject run_uuid/mode/out_dir for real rollouts.
        self._normalize_identity_in_cfg()

        # Create underlying env
        self._env = MiniShipAISCommsEnv(self._cfg) if use_pf_ais_env else MiniShipLagrangianParallelEnv(self._cfg)

        # Canonical agent id set from possible_agents/agents (stable contract outward)
        sample_ids = None
        if getattr(self._env, "possible_agents", None):
            sample_ids = list(self._env.possible_agents)
        elif getattr(self._env, "agents", None):
            sample_ids = list(self._env.agents)

        if not sample_ids:
            raise RuntimeError("Underlying env has no 'agents' or 'possible_agents' at init().")

        self._canon_agents = [str(a) for a in sample_ids]
        self._canon_set = set(self._canon_agents)

        any_agent = self._canon_agents[0]
        self.observation_space = self._env.observation_space(any_agent)
        self.action_space = self._env.action_space(any_agent)

        # current active agents (derived from returned keys)
        self._agent_ids: Set[str] = set()

        # terminal info cache
        self._last_ep_infos: Dict[str, Any] = {}

        # If cfg already carries dual snapshot, apply immediately.
        if isinstance(self._cfg.get("dual_snapshot", None), dict):
            self.set_dual_snapshot(self._cfg.get("dual_snapshot"))

    # ------------------------------------------------------------------
    # Identity contract helpers
    # ------------------------------------------------------------------
    def _normalize_identity_in_cfg(self) -> None:
        """
        Phase 2 identity contract normalization:
          - out_dir := out_dir if present else out (common CLI flag)
          - mode default 'train' (unless explicitly set)
          - staging_enable default: True for rollout, False for probe
          - For staging_enable=True, require: run_uuid + out_dir + mode
        """
        # 1) normalize out_dir
        if "out_dir" not in self._cfg or not self._cfg.get("out_dir"):
            if self._cfg.get("out"):
                self._cfg["out_dir"] = self._cfg.get("out")

        # 2) normalize mode
        if "mode" not in self._cfg or not self._cfg.get("mode"):
            self._cfg["mode"] = "train"

        # 3) staging_enable default logic (probe should disable)
        # Users may pass explicit staging_enable; otherwise infer from mode.
        if "staging_enable" not in self._cfg:
            self._cfg["staging_enable"] = (str(self._cfg.get("mode")).lower() != "probe")

        staging_enable = bool(self._cfg.get("staging_enable", True))

        # 4) enforce presence for real rollout
        if staging_enable:
            missing = []
            if not self._cfg.get("run_uuid"):
                missing.append("run_uuid")
            if not self._cfg.get("out_dir"):
                missing.append("out_dir(or out)")
            if not self._cfg.get("mode"):
                missing.append("mode")
            if missing:
                raise ValueError(
                    f"[rllib_env] identity contract missing fields: {missing}. "
                    f"Training script must inject run_uuid/mode/out_dir for staging rollouts."
                )

        # snapshot the identity fields (prevent drift by accidental mutation)
        self._id_run_uuid = self._cfg.get("run_uuid")
        self._id_mode = self._cfg.get("mode")
        self._id_out_dir = self._cfg.get("out_dir")
        self._id_worker_index = self._cfg.get("worker_index")
        self._id_vector_index = self._cfg.get("vector_index")

    def _assert_identity_stable(self) -> None:
        """
        Guard against identity drift inside a single env instance.
        """
        if self._cfg.get("run_uuid") != self._id_run_uuid:
            raise RuntimeError("[rllib_env] run_uuid drift detected in cfg (forbidden).")
        if self._cfg.get("mode") != self._id_mode:
            raise RuntimeError("[rllib_env] mode drift detected in cfg (forbidden).")
        if self._cfg.get("out_dir") != self._id_out_dir:
            raise RuntimeError("[rllib_env] out_dir drift detected in cfg (forbidden).")
        if self._cfg.get("worker_index") != self._id_worker_index:
            raise RuntimeError("[rllib_env] worker_index drift detected in cfg (forbidden).")
        if self._cfg.get("vector_index") != self._id_vector_index:
            raise RuntimeError("[rllib_env] vector_index drift detected in cfg (forbidden).")

    # ------------------------------------------------------------------
    # Key canonicalization
    # ------------------------------------------------------------------
    def _to_canon_key(self, k: Any) -> Optional[str]:
        if k is None:
            return None
        s = str(k)
        if s in ("__all__", "__common__"):
            return s
        if s in self._canon_set:
            return s

        # ship_1 <-> 1 mapping support
        m = re.match(r"^ship_(\d+)$", s)
        if m:
            d = m.group(1)
            if d in self._canon_set:
                return d
            cand = f"ship_{d}"
            if cand in self._canon_set:
                return cand

        if s.isdigit():
            cand = f"ship_{s}"
            if cand in self._canon_set:
                return cand

        return None

    def _remap_dict_keys(self, d: Any, *, keep_common: bool) -> Any:
        if not isinstance(d, dict):
            return d
        out: Dict[str, Any] = {}
        for k, v in d.items():
            if k == "__common__" and keep_common:
                out["__common__"] = v
                continue
            nk = self._to_canon_key(k)
            if nk is None:
                continue
            out[nk] = v
        return out

    def _remap_action_dict(self, action_dict: Any) -> Any:
        if not isinstance(action_dict, dict):
            return action_dict
        out: Dict[str, Any] = {}
        for k, v in action_dict.items():
            nk = self._to_canon_key(k)
            if nk is None or nk in ("__all__", "__common__"):
                continue
            out[nk] = v
        return out

    # ------------------------------------------------------------------
    # infos normalization: prevent nested bundle and enforce per-agent dict shape
    # ------------------------------------------------------------------
    def _looks_like_agent_bundle(self, d: Any) -> bool:
        if not isinstance(d, dict) or not d:
            return False
        child_keys = [k for k in d.keys() if str(k) in self._canon_set]
        if len(child_keys) < 2:
            return False
        if len(child_keys) < max(2, int(0.8 * len(d))):
            return False
        for k in child_keys:
            if not isinstance(d.get(k), dict):
                return False
        return True

    def _unwrap_nested_infos(self, infos: Any) -> Any:
        if not isinstance(infos, dict) or not infos:
            return infos
        out = dict(infos)
        bundle_stored = False
        for aid, v in list(out.items()):
            if aid in ("__all__", "__common__"):
                continue
            if isinstance(v, dict) and self._looks_like_agent_bundle(v):
                picked = v.get(aid) if isinstance(v.get(aid), dict) else None
                if picked is None:
                    for k in v.keys():
                        if str(k) in self._canon_set and isinstance(v.get(k), dict):
                            picked = v[k]
                            break
                out[aid] = picked if isinstance(picked, dict) else {}

                if not bundle_stored:
                    out.setdefault("__common__", {})
                    if isinstance(out["__common__"], dict) and "_all_agents_info" not in out["__common__"]:
                        out["__common__"]["_all_agents_info"] = v
                    bundle_stored = True
        return out

    def _normalize_infos(self, agent_ids: Set[str], infos: Any) -> Dict[str, Any]:
        if not isinstance(infos, dict):
            infos = {}
        common = infos.get("__common__", {})
        common = common if isinstance(common, dict) else {}

        out: Dict[str, Any] = {}
        for aid in agent_ids:
            v = infos.get(aid, {})
            if isinstance(v, dict):
                if any((k in agent_ids) and isinstance(v.get(k), dict) for k in v.keys()):
                    v = v.get(aid, {}) if isinstance(v.get(aid, {}), dict) else {}
            else:
                v = {}

            if common:
                vv = dict(v)
                for ck, cv in common.items():
                    if ck not in vv:
                        vv[ck] = cv
                out[aid] = vv
            else:
                out[aid] = v

        if common:
            out["__common__"] = common
        return out

    def _filter_infos(self, allowed_keys: Set[str], infos: Any) -> Any:
        if not isinstance(infos, dict):
            return infos
        valid = set(allowed_keys)
        valid.add("__common__")
        return {k: v for k, v in infos.items() if k in valid}

    # ------------------------------------------------------------------
    # Dual snapshot plumbing
    # ------------------------------------------------------------------
    def _apply_dual_to_obj(self, obj: Any, snap: Dict[str, Any]) -> bool:
        if obj is None or not isinstance(snap, dict):
            return False
        seen = set()
        cur = obj
        for _ in range(16):
            if cur is None or id(cur) in seen:
                return False
            seen.add(id(cur))

            for name in ("set_dual_snapshot", "set_dual"):
                f = getattr(cur, name, None)
                if callable(f):
                    try:
                        f(snap)
                        return True
                    except Exception:
                        pass

            if hasattr(cur, "core_env") and getattr(cur, "core_env") is not None and getattr(cur, "core_env") is not cur:
                cur = getattr(cur, "core_env")
                continue

            nxt = None
            for attr in ("unwrapped", "env", "_env", "wrapped_env", "_wrapped_env"):
                if hasattr(cur, attr):
                    cand = getattr(cur, attr, None)
                    if cand is not None and cand is not cur:
                        nxt = cand
                        break
            if nxt is None:
                return False
            cur = nxt
        return False

    def set_dual_snapshot(self, snap: Dict[str, Any]) -> None:
        if not isinstance(snap, dict):
            return
        self._dual_snapshot = _safe_deepcopy(snap)
        _ = self._apply_dual_to_obj(self._env, self._dual_snapshot)
        try:
            cfg = getattr(self._env, "cfg", None)
            if isinstance(cfg, dict):
                cfg["dual_snapshot"] = self._dual_snapshot
        except Exception:
            pass

    def set_dual(self, snap: Dict[str, Any]) -> None:
        self.set_dual_snapshot(snap)

    def get_dual_snapshot(self) -> Optional[Dict[str, Any]]:
        for name in ("get_dual_snapshot", "get_dual"):
            f = getattr(self._env, name, None)
            if callable(f):
                try:
                    v = f()
                    if isinstance(v, dict):
                        return v
                except Exception:
                    pass
        return self._dual_snapshot

    # RLlib sometimes reads env.agents
    @property
    def agents(self):
        return list(self._agent_ids)

    @property
    def core_env(self):
        """Expose core env for callbacks to access _last_infos through wrapper chain."""
        return getattr(self._env, "core_env", None) or getattr(self._env, "core", None) or self._env

    # ------------------------------------------------------------------
    # reset/step
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        # identity must not drift
        self._assert_identity_stable()

        obs, infos = self._env.reset(seed=seed, options=options)

        obs = self._remap_dict_keys(obs, keep_common=False)
        infos = self._remap_dict_keys(infos, keep_common=True)
        infos = self._unwrap_nested_infos(infos)

        agent_ids = {k for k in obs.keys() if k not in ("__all__", "__common__")}
        self._agent_ids = set(agent_ids)

        infos = self._normalize_infos(agent_ids, infos)
        infos = self._filter_infos(agent_ids, infos)

        # reset counters/caches
        self._ep_i += 1
        self._step_i = 0
        self._terminal_printed = False
        self._last_ep_infos = {}

        if self._rdbg_enable:
            self._rdbg_ep_total = 0.0
            self._rdbg_ep_total_firstn = 0.0
            self._rdbg_ep_total_by_agent = {}
            self._rdbg_ep_total_firstn_by_agent = {}

        return obs, infos

    def step(self, action_dict: Dict[str, Any]):
        # identity must not drift
        self._assert_identity_stable()

        action_dict = self._remap_action_dict(action_dict)
        obs, rewards, terminateds, truncateds, infos = self._env.step(action_dict)

        obs = self._remap_dict_keys(obs, keep_common=False)
        rewards = self._remap_dict_keys(rewards, keep_common=False)
        terminateds = self._remap_dict_keys(terminateds, keep_common=False)
        truncateds = self._remap_dict_keys(truncateds, keep_common=False)
        infos = self._remap_dict_keys(infos, keep_common=True)
        infos = self._unwrap_nested_infos(infos)

        # derive current agent set from returned keys
        agent_ids: Set[str] = set()
        if isinstance(obs, dict):
            agent_ids |= {k for k in obs.keys() if k not in ("__all__", "__common__")}
        if isinstance(rewards, dict):
            agent_ids |= set(rewards.keys())
        if isinstance(terminateds, dict):
            agent_ids |= {k for k in terminateds.keys() if k not in ("__all__", "__common__")}
        if isinstance(truncateds, dict):
            agent_ids |= {k for k in truncateds.keys() if k not in ("__all__", "__common__")}

        # fallback if everything is empty at terminal
        if not agent_ids:
            agent_ids = set(self._agent_ids) if self._agent_ids else set(self._canon_agents)

        self._agent_ids = set(agent_ids)

        # ensure "__all__"
        terminateds = dict(terminateds) if isinstance(terminateds, dict) else {}
        truncateds = dict(truncateds) if isinstance(truncateds, dict) else {}

        term_all = any(bool(terminateds.get(a, False)) for a in agent_ids)
        trunc_all = any(bool(truncateds.get(a, False)) for a in agent_ids)

        if "__all__" not in terminateds:
            terminateds["__all__"] = term_all
        if "__all__" not in truncateds:
            truncateds["__all__"] = trunc_all

        infos = self._normalize_infos(agent_ids, infos)
        infos = self._filter_infos(agent_ids, infos)

        # reward accumulation (final rewards returned to RLlib)
        if self._rdbg_enable and isinstance(rewards, dict):
            step_sum = 0.0
            for aid, rv in rewards.items():
                try:
                    r = float(rv)
                except Exception:
                    r = float("nan")
                if not math.isfinite(r):
                    r = 0.0
                step_sum += r
                self._rdbg_ep_total_by_agent[aid] = self._rdbg_ep_total_by_agent.get(aid, 0.0) + r
                if self._step_i < int(self._rdbg_n):
                    self._rdbg_ep_total_firstn_by_agent[aid] = self._rdbg_ep_total_firstn_by_agent.get(aid, 0.0) + r

            self._rdbg_ep_total += step_sum
            if self._step_i < int(self._rdbg_n):
                self._rdbg_ep_total_firstn += step_sum

        # optional obs-vs-info consistency checks
        self._debug_check_obs_vs_info(obs, infos, agent_ids, self._step_i)

        # terminal cache
        done_all = bool(terminateds.get("__all__", False)) or bool(truncateds.get("__all__", False))
        if done_all and isinstance(infos, dict):
            last: Dict[str, Any] = {}
            for aid in agent_ids:
                v = infos.get(aid, {})
                last[aid] = v if isinstance(v, dict) else {}
            if "__common__" in infos and isinstance(infos.get("__common__"), dict):
                last["__common__"] = infos["__common__"]
            self._last_ep_infos = last

        # throttled step prints (default OFF)
        if self._dbg_step_print_n > 0:
            if (self._step_i < self._dbg_step_print_n) and (self._step_i % self._dbg_step_print_every == 0):
                pid = os.getpid()
                print(
                    f"[rllib_env][StepDbg] tag={self._dbg_step_tag} ep={self._ep_i} step={self._step_i+1} pid={pid} "
                    f"obs_keys={list(obs.keys()) if isinstance(obs, dict) else type(obs)} "
                    f"rew_keys={list(rewards.keys()) if isinstance(rewards, dict) else type(rewards)} "
                    f"term_all={bool(terminateds.get('__all__', False))} trunc_all={bool(truncateds.get('__all__', False))}"
                )

        self._step_i += 1
        return obs, rewards, terminateds, truncateds, infos

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()

    # ------------------------------------------------------------------
    # Debug helper: obs vs info check (optional)
    # ------------------------------------------------------------------
    def _debug_check_obs_vs_info(self, obs: Any, infos: Any, agent_ids: Set[str], step_i: int) -> None:
        if not self._dbg_obs_state_check:
            return
        if self._dbg_obschk_i >= self._dbg_obs_state_check_n:
            return
        if (step_i % self._dbg_obs_state_check_every) != 0:
            return
        if not isinstance(obs, dict) or not isinstance(infos, dict):
            return

        idx_map = self._cfg.get("debug_obs_state_check_idx_map", {"x": 0, "y": 1, "psi": 2, "v": 3, "gx": 4, "gy": 5})
        tol = float(self._dbg_obs_state_check_tol)
        hard_assert = bool(self._dbg_obs_state_check_assert)

        def wrap_pi(a: float) -> float:
            return (a + math.pi) % (2 * math.pi) - math.pi

        printed_any = False
        for aid in sorted(agent_ids):
            o = obs.get(aid, None)
            info = infos.get(aid, None)
            if o is None or not isinstance(info, dict):
                continue

            try:
                ov = np.asarray(o, dtype=float).reshape(-1)
            except Exception:
                continue

            if ov.size > 0 and (not np.all(np.isfinite(ov))):
                msg = f"[rllib_env][OBSCHK][BAD_OBS] ep={self._ep_i} step={step_i} aid={aid} obs_has_nan_inf=True"
                print(msg)
                if hard_assert:
                    raise AssertionError(msg)

            diffs = {}
            for k, idx in idx_map.items():
                if k not in info:
                    continue
                if not isinstance(idx, int) or idx < 0 or idx >= ov.size:
                    continue
                try:
                    obs_val = float(ov[idx])
                    info_val = float(info.get(k))
                except Exception:
                    continue

                if not (math.isfinite(obs_val) and math.isfinite(info_val)):
                    continue

                d = abs(wrap_pi(obs_val - info_val)) if k == "psi" else abs(obs_val - info_val)
                if d > tol:
                    diffs[k] = (obs_val, info_val, d)

            if diffs:
                printed_any = True
                parts = " ".join([f"{k}:obs={v[0]:+.6f} info={v[1]:+.6f} |d|={v[2]:.3e}" for k, v in diffs.items()])
                msg = f"[rllib_env][OBSCHK][MISMATCH] ep={self._ep_i} step={step_i} aid={aid} tol={tol:.1e} {parts}"
                print(msg)
                if hard_assert:
                    raise AssertionError(msg)

        if printed_any:
            self._dbg_obschk_i += 1


# -----------------------------
# RLlib env registration
# -----------------------------
def register_miniship_env(base_env_cfg: Dict[str, Any]) -> Tuple[str, Any, Any]:
    """
    Register MiniShip env into RLlib.

    Important:
      - env_creator will receive EnvContext; we must inject worker_index/vector_index into cfg.
      - A temporary env is created to infer spaces; that env MUST NOT write stage files
        (probe/rollout separation). We force staging_enable=False for the temporary env.
    """
    env_name = "MiniShipLagrangianPZ-v0"

    def env_creator(env_config: Dict[str, Any]):
        cfg = dict(base_env_cfg)
        if env_config:
            # EnvContext is dict-like; update will read keys, while attrs are handled below
            cfg.update(dict(env_config) if hasattr(env_config, "items") else dict(env_config))

        use_pf_ais_env = bool(cfg.get("use_pf_ais_env", False))

        # Inject RLlib context: attr-first
        wi_attr = getattr(env_config, "worker_index", None)
        vi_attr = getattr(env_config, "vector_index", None)

        wi_key = None
        vi_key = None
        if hasattr(env_config, "get"):
            try:
                wi_key = env_config.get("worker_index", None)
                vi_key = env_config.get("vector_index", None)
            except Exception:
                wi_key, vi_key = None, None

        wi = _to_int(wi_attr if wi_attr is not None else wi_key, _to_int(cfg.get("worker_index", None), 0))
        vi = _to_int(vi_attr if vi_attr is not None else vi_key, _to_int(cfg.get("vector_index", None), 0))

        cfg["worker_index"] = wi
        cfg["vector_index"] = vi

        return MiniShipMultiAgentWrapper(cfg, use_pf_ais_env=use_pf_ais_env)

    register_env(env_name, env_creator)

    # Space probe env (must not write stage files)
    tmp_cfg = dict(base_env_cfg)
    tmp_cfg.setdefault("mode", "probe")
    tmp_cfg["staging_enable"] = False
    tmp_cfg.setdefault("worker_index", 0)
    tmp_cfg.setdefault("vector_index", 0)

    use_pf_ais_env_tmp = bool(tmp_cfg.get("use_pf_ais_env", False))
    tmp_env = MiniShipMultiAgentWrapper(tmp_cfg, use_pf_ais_env=use_pf_ais_env_tmp)
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    tmp_env.close()

    return env_name, obs_space, act_space
