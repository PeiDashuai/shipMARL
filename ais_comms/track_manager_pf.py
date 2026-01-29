# ais_comms/track_manager_pf.py
from __future__ import annotations
from collections import deque
import csv
from collections.abc import Mapping
import os
import glob
from datetime import datetime, timezone
"""
================================================================================
AISTrackManagerPF (Per-Agent PF) — SEMANTICS CONTRACT (DO NOT CHANGE)
================================================================================

This file enforces:
  1) single-point conversion + end-to-end naming unification (angles)
  2) single time-axis for PF + end-to-end time naming unification (times)

If you change any definition below, you MUST update every dependent module
consistently (ais_preproc.py / pf_ctrv.py / obs builder / logs).

================================================================================
A) ANGLE SEMANTICS CONTRACT (DO NOT CHANGE)
================================================================================

We use TWO distinct angle conventions in this project:

+(A1) Incoming raw COG direction (incoming from RxMsg.reported_cog):
+    - Name in code:   raw_cog_rad  (a.k.a. course_enu_ccw_rad)
+    - Meaning:        0 rad along +X (EAST), positive COUNTER-CLOCKWISE
+                      i.e., standard math angle used by atan2(vy, vx)
+    - Unit:           radians

(A2) Internal yaw (used everywhere inside tracker/PF/obs/debug):
+    - Name in code:   yaw_sim_rad  (a.k.a. yaw_enu_rad)
+    - Meaning:        0 rad along +X axis, positive COUNTER-CLOCKWISE
+                      i.e., standard math angle used by atan2(vy, vx)
+    - Unit:           radians
+    - Range:          (-pi, pi]

+Single-point normalization (the ONLY allowed operation in this file):
+    yaw_sim_rad = wrap_pi(raw_cog_rad)
+
+We DO NOT use nautical AIS COG semantics (North=0, CW+) anywhere in the PF
+pipeline. If you ever need a North/CW angle for reporting, derive it only as a
+debug/aux value (never as PF input).

Legacy policy:
- Some historical interfaces used key name "cog" to store **yaw**.
- In THIS file, "cog" is NEVER treated as AIS COG.
- When calling legacy PF interfaces, we pass alias key "cog" == yaw_sim_rad
  ONLY for backward compatibility.

Ego yaw (debug-only):
- ego_yaw_sim_rad uses internal yaw convention.
- it comes from TrueState heading/yaw/psi if present, else atan2(vy, vx).
- used ONLY for debugging prints (never for warm-start cheating).

RxMsg.reported_cog is raw_cog_rad == yaw_sim_rad (ENU, +X=0, CCW+, rad, (-pi,pi]).

PF input uses: meas.yaw = wrap_pi(RxMsg.reported_cog) and additionally sets legacy alias meas.cog = meas.yaw.

Any cog_north_cw value is derived for debug only and must never be used as PF input.

================================================================================
C) STAGE-3 / STAGE-4 EPISODE JOIN KEY CONTRACT (DO NOT CHANGE)
================================================================================

All Stage-3 and Stage-4 logs produced by this module (CSV + JSONL) MUST contain
the following *required* columns/keys in every row:

    run_uuid, run_id, pid, worker_index, vector_index,
    stage3_episode_uid, stage3_episode_idx, agent_id

Episode-level join key (PRIMARY KEY for cross-stage joins):

    (run_uuid, worker_index, vector_index, stage3_episode_uid, stage3_episode_idx)

Notes:
  - run_uuid: Stage-3 authoritative run identifier, stable across processes.
  - run_id: trial/experiment identifier (aux; NOT part of the episode-level PK).
  - pid: process id (aux; NOT part of the PK; used to disambiguate multi-process writers).
  - agent_id: required for per-agent rows. If you ever write purely episode-level
    rows (no per-agent semantics), set agent_id="__episode__" explicitly.

Hard rule:
  - This module MUST NOT invent random IDs for cross-stage join.
  - The environment must inject Stage-3 identity into this tracker via:
        set_stage4_context(...) / stage4_set_episode_context(...)
    before any Stage-3/Stage-4 rows are written.

Enforcement:
  - On every write (Stage-3 CSV / Stage-4 CSV / Stage-4 JSONL), we assert that
    the required fields above are present and non-empty (unless explicitly
    disabled by PF_STAGE34_JOIN_ASSERT=0).


================================================================================
B) TIME SEMANTICS CONTRACT (DO NOT CHANGE)
================================================================================

We define THREE time concepts:

(T0) t_env / now:
    - Name in code:   t_env, now, ts_now
    - Meaning:        environment simulation time at current RL step boundary
    - Source:         ingest(t, ...) receives t == current env time
    - Unit:           seconds

(T1) reported_ts (message timestamp):
    - Name in code:   ts_rep
    - Meaning:        timestamp embedded in AIS message (Tx-time on sender axis)
    - Source:         RxMsg.reported_ts  (after comms simulation)
    - Unit:           seconds

(T2) arrival_time (network delivery time):
    - Name in code:   ts_arr
    - Meaning:        simulated time when packet arrives at receiver
    - Source:         RxMsg.arrival_time
    - Constraint:     ts_arr <= t_env (packet may arrive within the step interval)
    - Unit:           seconds
    - IMPORTANT:      In THIS FILE, ts_arr is **metadata only** (comms stats/debug).
                      PF does NOT run on arrival_time axis.

-------------------------------
PF SINGLE TIME AXIS (HARD RULE)
-------------------------------
PF internal time axis is ALWAYS the environment step time t_env.

That means:
- Every PF fusion/update is performed at time:   t_pf = t_env
- PF internal cursor: pf.last_ts is always aligned with env time after sync/predict.

We represent delayed measurements by projecting them from reported_ts to t_env:

Let:
    dt_comm  = max(0, ts_arr - ts_rep)     # pure network delay (stats/debug)
    dt_hold  = max(0, t_env  - ts_arr)     # "held until step boundary" (scheduler artifact)
    dt_proj  = max(0, t_env  - ts_rep)     # effective staleness to be compensated/uncertainty

We "use" a measurement position at t_env by forward projecting:
    x_use = x_rep + sog * dt_proj * cos(yaw_sim_rad)
    y_use = y_rep + sog * dt_proj * sin(yaw_sim_rad)

Age semantics:
- age_rep (or age_meas) = dt_proj = t_env - ts_rep
  This is the ONLY "age" fed into PF gating/robust update.

Track timestamp fields (write-on-fusion):
- PFTrack.last_reported_ts : last ts_rep that got fused (message timestamp)
- PFTrack.last_arrival_ts  : last ts_arr of fused packet (metadata)
- PFTrack.last_update_ts   : last t_env when fusion happened (PF axis == env axis)
- PFTrack.last_meas_x/y    : last (x_use,y_use) fused at t_env (projected-to-env)

Validity / staleness used by query:
- info_age  = t_env - last_reported_ts   (message staleness)
- silence   = t_env - last_update_ts     (how long PF hasn't been fused)
- tr.valid  = (silence <= max_age)

================================================================================
"""

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Dict, List, Optional, Tuple
from typing import Any
import zlib

import math
import json
import numpy as np
from typing import Iterable
from typing import IO
import re
from typing import Any, Dict, Optional
import inspect
from .ais_preproc import AISPreprocessor, SimpleMeas
from .datatypes import TrueState, ShipId, AgentId, Ts, RxMsg
from .pf_ctrv import PFConfig, PFNoiseConfig, PFAgeGatingConfig, ParticleCTRVFilter

# ============================================================
# STAGE-3/4 JOIN CONTRACT (hard assertions on write)
# ============================================================

PF_STAGE34_JOIN_ASSERT = (os.environ.get("PF_STAGE34_JOIN_ASSERT", "1").strip() != "0")

STAGE34_REQUIRED_FIELDS: tuple[str, ...] = (
    "run_uuid",
    "run_id",
    "pid",
    "worker_index",
    "vector_index",
    "stage3_episode_uid",
    "stage3_episode_idx",
    "agent_id",
)

STAGE34_EPISODE_PK_FIELDS: tuple[str, ...] = (
    "run_uuid",
    "worker_index",
    "vector_index",
    "stage3_episode_uid",
    "stage3_episode_idx",
)

# ============================================================
# Stage-4 ctor ctx freeze contract
#   - "ctor ctx" = stage4_ctx passed into __init__ (or PF_STAGE4_CTX_JSON)
#   - Freeze run-level identifiers/metadata once set (do NOT allow overwrite).
#   - Episode keys MUST remain mutable across resets.
# ============================================================

STAGE4_EPISODE_MUTABLE_KEYS: set[str] = {
    "stage3_episode_uid", "stage3_episode_idx",
    "episode_uid", "episode_idx", "episode_id",
}

# Run-level keys we freeze when they are provided via ctor ctx (or on first set).
STAGE4_RUN_FROZEN_KEYS: set[str] = {
    # join identity
    "run_uuid", "stage3_run_uuid", "stage3_run_id",
    "run_id",
    "worker_index", "vector_index", "env_index",
    # run/log placement
    "run_dir", "out_dir", "trial_dir", "log_dir", "out", "logdir", "result_dir",
    # metadata that should not drift inside a run
    "ais_cfg_hash", "pf_cfg_hash", "code_commit",
    "env_dt", "N",
    "scenario_id", "encounter_type",
    "seed", "worker", "exp",
}

# ============================================================
# Stage-3 stage4_ctx loader (Plan A for Stage-4 join recovery)
#   - Stage-4 startup loads Stage-3 "stage4_ctx*.jsonl"
#   - Build index by: (worker_index, vector_index, stage3_episode_idx)
#   - Fill missing keys (run_uuid, stage3_episode_uid, ...) by lookup.
#
# HARD RULE:
#   - MUST NOT invent run/episode IDs for cross-stage join.
#   - Only consume env-injected fields OR load them from Stage-3 ctx logs.
# ============================================================
PF_STAGE4_LOAD_STAGE3_CTX = (os.environ.get("PF_STAGE4_LOAD_STAGE3_CTX", "1").strip() != "0")
PF_STAGE3_STAGE4_CTX_GLOB = os.environ.get("PF_STAGE3_STAGE4_CTX_GLOB", "").strip()
PF_STAGE3_STAGE4_CTX_RECURSIVE = (os.environ.get("PF_STAGE3_STAGE4_CTX_RECURSIVE", "1").strip() != "0")
PF_STAGE3_STAGE4_CTX_MAX_FILES = int(os.environ.get("PF_STAGE3_STAGE4_CTX_MAX_FILES", "256"))
PF_STAGE3_STAGE4_CTX_MAX_LINES = int(os.environ.get("PF_STAGE3_STAGE4_CTX_MAX_LINES", "500000"))
PF_STAGE3_STAGE4_CTX_WARN = (os.environ.get("PF_STAGE3_STAGE4_CTX_WARN", "0").strip() == "1")


def _is_empty_stage34(v: object) -> bool:
    if v is None:
        return True
    # allow 0 for index-like fields
    if isinstance(v, (int, np.integer)):
        return False
    s = str(v).strip()
    return (s == "") or (s.lower() in ("none", "null", "na", "nan"))

def _assert_stage34_join_contract(row: Mapping[str, object], *, where: str) -> None:
    """
    Hard assert that Stage-3/Stage-4 join contract fields exist and are non-empty.
    This is intentionally strict to prevent silent schema drift / broken joins.
    Disable ONLY for emergency by setting: PF_STAGE34_JOIN_ASSERT=0
    """
    if not PF_STAGE34_JOIN_ASSERT:
        return
    missing: list[str] = []
    for k in STAGE34_REQUIRED_FIELDS:
        if (k not in row) or _is_empty_stage34(row.get(k)):
            missing.append(k)
    if missing:
        preview = {}
        try:
            for kk in STAGE34_REQUIRED_FIELDS:
                preview[kk] = row.get(kk, None)
        except Exception:
            preview = {"__preview__": "failed"}
        raise AssertionError(
            f"[STAGE34-JOIN-CONTRACT-FAIL] where={where} missing_or_empty={missing} "
            f"required={list(STAGE34_REQUIRED_FIELDS)} preview={preview}"
        )


# ===================== VERSION / PATH DEBUG =====================
def _print_loaded_paths_once():
    try:
        import ais_comms
        import ais_comms.track_manager_pf as _tm
        import ais_comms.pf_ctrv as _pf
        import ais_comms.ais_preproc as _pp
        import ais_comms.datatypes as _dt
        print("[PATHDBG] ais_comms package:", getattr(ais_comms, "__file__", None))
        print("[PATHDBG] track_manager_pf:", getattr(_tm, "__file__", None))
        print("[PATHDBG] pf_ctrv:", getattr(_pf, "__file__", None))
        print("[PATHDBG] ais_preproc:", getattr(_pp, "__file__", None))
        print("[PATHDBG] datatypes:", getattr(_dt, "__file__", None))
    except Exception as e:
        print("[PATHDBG] failed to print module paths:", repr(e))


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_sim_rad_to_cog_north_cw_rad_debug(yaw_sim_rad: float) -> float:
    """
    DEBUG-ONLY: derive nautical COG for reporting.
    DO NOT feed this into PF. PF uses yaw_sim_rad (ENU, +X=0, CCW+) everywhere.
    """
    cog = (math.pi / 2.0) - float(yaw_sim_rad)
    return (cog % (2.0 * math.pi) + 2.0 * math.pi) % (2.0 * math.pi)


# ===================== PF DEBUG SETTINGS =====================
DEBUG_PF = bool(int(os.getenv("PF_DEBUG", "0")))
DEBUG_PF_SID = os.getenv("PF_DEBUG_SID", None)
DEBUG_PF_SID = int(DEBUG_PF_SID) if (DEBUG_PF_SID is not None and DEBUG_PF_SID != "") else None
DEBUG_PF_ERR_THRESH = float(os.getenv("PF_DEBUG_ERR_THRESH", "150.0"))
DEBUG_PF_RX_DETAIL = bool(int(os.getenv("PF_DEBUG_RX_DETAIL", "0")))

try:
    if os.environ.get("PF_DEBUG", "").strip() != "":
        DEBUG_PF = (os.environ.get("PF_DEBUG", "0") == "1")
    if os.environ.get("PF_DEBUG_SID", "").strip() != "":
        _sid = os.environ.get("PF_DEBUG_SID", "").strip()
        DEBUG_PF_SID = None if _sid.lower() in ("none", "") else int(_sid)
    if os.environ.get("PF_DEBUG_RX_DETAIL", "").strip() != "":
        DEBUG_PF_RX_DETAIL = (os.environ.get("PF_DEBUG_RX_DETAIL", "0") == "1")
except Exception:
    pass

_pf_log_f = None


def _get_pf_log_f():
    global _pf_log_f
    if _pf_log_f is not None:
        return _pf_log_f
    fn = os.environ.get("PF_DEBUG_LOG", "").strip()
    if not fn:
        fn = f"pf_debug.{os.getpid()}.log"
    _pf_log_f = open(fn, "a", buffering=1)
    return _pf_log_f

# ===================== PF STATS (CSV) =====================
PF_STATS = bool(int(os.getenv("PF_STATS", "0")))
_pf_stats_f = None
_pf_stats_header_written = False


# ===================== PF STATS (CSV) SCHEMA =====================
# Goal:
#   - Stable columns across the whole run (do NOT depend on first row keys).
#   - Preserve unknown keys via extra_json to avoid silent field loss.
PF_STATS_COLS: List[str] = [
    # ---- REQUIRED: Stage-3/4 contract fields (must exist in every row) ----
    # Columns: run_uuid, run_id, pid, worker_index, vector_index,
    #          stage3_episode_uid, stage3_episode_idx, agent_id
    "run_uuid", "run_id", "pid", "worker_index", "vector_index",
    "stage3_episode_uid", "stage3_episode_idx", "episode_join_key", "agent_id",


    # ---- local ctx versioning ----
    "ctx_version",

    # legacy / optional
    "seed", "worker", "exp",
    "scenario_id", "encounter_type", "env_dt", "N",
    "ais_cfg_hash", "pf_cfg_hash", "code_commit",

    # ---- canonical identifiers ----
    "sid", "msg_id", "t_env", "ts_rep", "ts_arr",
    "join_agent_id", "join_sid", "join_msg_id", "join_t_env", "join_ts_rep", "join_ts_arr", "join_key",

    # ---- time decomposition ----
    "dt_comm", "dt_hold", "age_meas",

    # ---- measurement / pf residuals ----
    "mx_rep", "my_rep", "mx_use", "my_use",
    "px_pre", "py_pre", "dist_pre",
    "relock",

    # ---- PF update stats (flat) ----
    "sigma_pos", "sigma_sog", "sigma_yaw",
    "neff", "resampled", "collapsed", "ret",

    # ---- Stage-4 instrumentation (flat) ----
    "s4_dt_gap", "s4_soft_thr", "s4_hard_thr",
    "s4_meas_yaw", "s4_pyaw_pre", "s4_yawd_pre", "s4_yaw_innov_abs",
    "s4_age_band",
    "s4_use_q_gap", "s4_use_r_cont",
    "s4_q_sigma_pos", "s4_q_sigma_sog", "s4_q_sigma_yaw", "s4_q_sigma_yawd",
    "s4_yaw_gate_hit", "s4_yaw_soft_pass", "s4_yawd_soft_pass",
    "s4_like_min", "s4_like_max",
    "s4_px_post", "s4_py_post", "s4_pyaw_post", "s4_yawd_post",

    # ---- catch-all ----
    "extra_json",
]

# PF yaw contract debug (meas injection point)
# Enable via: PF_DEBUG_YAW=1
# Optional:
#   PF_DEBUG_YAW_SID=<int>   (only this target sid)
#   PF_DEBUG_YAW_TOL=<float> (default 1e-4, in rad, wrapped to (-pi,pi])
#   PF_DEBUG_YAW_MAX=<int>   (max prints; default 20)
PF_DEBUG_YAW = bool(int(os.getenv("PF_DEBUG_YAW", "0")))
_pf_yaw_sid_s = os.getenv("PF_DEBUG_YAW_SID", "").strip()
PF_DEBUG_YAW_SID = int(_pf_yaw_sid_s) if _pf_yaw_sid_s else None
PF_DEBUG_YAW_TOL = float(os.getenv("PF_DEBUG_YAW_TOL", "1e-4"))
PF_DEBUG_YAW_MAX = int(os.getenv("PF_DEBUG_YAW_MAX", "20"))
_pf_yaw_dbg_n = 0

# ------------------------------------------------------------
# IMPORTANT (anti-cheating guard):
#   Truth-derived fields (e.g., ego yaw from TrueState) MUST NOT
#   enter the preprocessor/PF path by default.
#   Enable ONLY when you explicitly want debug instrumentation.
# ------------------------------------------------------------
PF_ATTACH_EGO_YAW = bool(int(os.getenv("PF_ATTACH_EGO_YAW", "0")))

def _get_pf_stats_f():
    global _pf_stats_f
    if _pf_stats_f is not None:
        return _pf_stats_f
    fn = os.environ.get("PF_STATS_LOG", "").strip()
    if not fn:
        # 每个进程独立文件，避免多 worker 争用同一文件
        fn = f"pf_stats.{os.getpid()}.csv"
    _pf_stats_f = open(fn, "a", buffering=1)
    return _pf_stats_f

def _csv_file_is_empty(path: str) -> bool:
    try:
        return (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    except Exception:
        return False

def _csv_write_schema_and_header_if_needed(
    f: IO,
    *,
    path: str,
    schema_comment: Optional[str],
    cols: List[str],
) -> None:
    """
    Robust header logic:
      - ONLY write comment+header if file is empty (size==0).
      - This prevents repeated header lines under re-open / re-init.
    """
    try:
        if not _csv_file_is_empty(path):
            return
        if schema_comment:
            f.write(schema_comment.rstrip("\n") + "\n")
        w = csv.writer(f)
        w.writerow(cols)
    except Exception:
        # never fail training due to logging
        return

def _csv_write_row(
    f: IO,
    *,
    cols: List[str],
    row: Dict[str, Any],
) -> None:
    """
    CSV row writer using csv.writer (proper quoting).
    Unknown keys must already be packed into extra_json.
    """
    vals: List[Any] = []
    for k in cols:
        v = row.get(k, "")
        if v is None:
            v = ""
        vals.append(v)
    w = csv.writer(f)
    w.writerow(vals)

def pf_stats_write(row: dict):
    """
    Lightweight CSV logger. Default OFF.
    Enable by: PF_STATS=1
    Optional: PF_STATS_LOG=path/to/file.csv
    """
    global _pf_stats_header_written
    if not PF_STATS:
        return
    try:

        f = _get_pf_stats_f()
        cols = PF_STATS_COLS
        # P0: robust header (size-based). Use the *actual* opened file path (f.name),
        # not an env/default recomputation that can diverge.
        path_hint = os.environ.get("PF_STATS_LOG", "").strip() or f"pf_stats.{os.getpid()}.csv"
        real_path = getattr(f, "name", "") or path_hint
        _csv_write_schema_and_header_if_needed(
            f,
            path=real_path,
            schema_comment="# stage3/pf_stats schema=PF_STATS_COLS_v2",
            cols=cols,
        )

        # unknown keys -> extra_json (keep information, avoid schema drift loss)
        extra = {}
        try:
            for k, v in (row.items() if isinstance(row, dict) else []):
                if str(k) not in cols:
                    extra[str(k)] = v
        except Exception:
            extra = {}
        row2 = dict(row) if isinstance(row, dict) else {}
        if "extra_json" not in row2:
            try:
                # P1: keep strict JSON (do NOT replace commas)
                row2["extra_json"] = json.dumps(_stage4_json_sanitize(extra), ensure_ascii=False, separators=(",", ":"))
            except Exception:
                row2["extra_json"] = str(extra)
        _assert_stage34_join_contract(row2, where="stage3/pf_stats_write")
        _csv_write_row(f, cols=cols, row=row2)

    except Exception as e:
        # stats 不应影响训练
        if os.environ.get("PF_STATS_WARN", "0") == "1":
            print("[PF_STATS][WARN]", repr(e))

def pf_log(msg: str, sid: int | None = None):
    if not DEBUG_PF:
        return
    if (DEBUG_PF_SID is not None) and (sid is not None) and (sid != DEBUG_PF_SID):
        return
    f = _get_pf_log_f()
    print(msg, file=f)

 
# ===================== STAGE-4 OUTPUT (JSONL) =====================
# Stage-4 is "instrumentation output" around robust PF update/relock decisions.
# Enable via:
#   PF_STAGE4=1
# Optional:
#   PF_STAGE4_LOG=path/to/pf_stage4.{pid}.jsonl   (supports "{pid}" placeholder)
PF_STAGE4 = bool(int(os.getenv("PF_STAGE4", "0")))

_pf_stage4_f: Optional[IO] = None
_pf_stage4_path: Optional[str] = None


# PF_STATS_STAGE4 default: ON when PF_STAGE4=1 unless explicitly disabled.
try:
    PF_STATS_STAGE4 = bool(PF_STAGE4) and (os.environ.get("PF_STATS_STAGE4", "1").strip() != "0")
except Exception:
    PF_STATS_STAGE4 = bool(PF_STAGE4)

def _default_stage4_path() -> str:
    """
    Backward-compatible default path resolver.
    Priority:
      1) PF_STAGE4_LOG (file path; if relative, will be relative to PF_STAGE4_DIR or cwd)
      2) PF_STAGE4_DIR (dir) + stage4/pf_stage4.{pid}.jsonl
      3) cwd/stage4/pf_stage4.{pid}.jsonl
    """
    pid = os.getpid()
    fn = os.environ.get("PF_STAGE4_LOG", "").strip()
    base_dir = os.environ.get("PF_STAGE4_DIR", "").strip()

    # If PF_STAGE4_DIR is not provided, infer Stage-4 base dir from Stage-3 run_dir
    # so that default Stage-4 artifacts land at "<stage3_run_dir>/stage4/" instead of CWD.
    if not base_dir:
        try:
            base_dir = _infer_stage4_base_dir_from_stage3_env() or ""
        except Exception:
            base_dir = ""

    if fn:
        if "{pid}" in fn:
            fn = fn.format(pid=pid)
        else:
            root, ext = os.path.splitext(fn)
            ext = ext if ext else ".jsonl"
            fn = f"{root}.{pid}{ext}"
        # If relative path, anchor it to inferred base_dir (Stage-3 run_dir) when available.
        if not os.path.isabs(fn) and base_dir:
            fn = os.path.join(base_dir, fn)
        return fn

    # directory mode
    if not base_dir:
        base_dir = os.getcwd()
    return os.path.join(base_dir, "stage4", f"pf_stage4.{pid}.jsonl")

 
def _infer_stage4_base_dir_from_stage3_env() -> Optional[str]:
    """
    Infer Stage-4 base directory from Stage-3 run directory hints.
    Goal: default Stage-4 outputs go to "<stage3_run_dir>/stage4/" (sibling of "stage3/"),
    NOT the current working directory.

    Heuristics:
      - If a hint points to a run_dir that contains "stage3/" as a subdir, treat it as run_dir.
      - If a hint points to ".../stage3" or ".../stage4", use its parent.
      - If a hint points to a file/dir under ".../stage3/...", use the parent before "stage3".
    """
    def _abspath(p: str) -> str:
        p = os.path.abspath(p)
        try:
            if os.path.isfile(p):
                p = os.path.dirname(p)
        except Exception:
            pass
        return p

    def _pick_run_dir(p_in: str) -> Optional[str]:
        if not isinstance(p_in, str) or not p_in.strip():
            return None
        p = _abspath(p_in.strip())

        # If p looks like run_dir and contains stage3/ -> accept directly.
        try:
            if os.path.isdir(p) and os.path.isdir(os.path.join(p, "stage3")):
                return p
        except Exception:
            pass

        # If p ends at stage3/ or stage4/ -> use parent as run_dir.
        b = os.path.basename(p.rstrip(os.sep))
        if b in ("stage3", "stage4"):
            return os.path.dirname(p.rstrip(os.sep))

        # If p contains ".../stage3/..." -> take parent before stage3.
        parts = p.split(os.sep)
        if "stage3" in parts:
            idx = parts.index("stage3")
            if idx > 0:
                return os.sep.join(parts[:idx])
            return os.sep

        return None

    # Stage-3 first, then common Ray/Tune run dir hints as fallback.
    env_keys = (
        "STAGE3_RUN_DIR", "STAGE3_DIR", "STAGE3_OUT_DIR", "STAGE3_LOG_DIR", "STAGE3_PATH",
        "PF_STAGE3_RUN_DIR", "PF_STAGE3_DIR", "PF_STAGE3_PATH",
        "PF_STATS_LOG", "PF_COMM_STATS_LOG", "PF_EPISODES_LOG",
        "TUNE_TRIAL_DIR", "TUNE_LOGDIR", "TUNE_RESULT_DIR", "RAY_RESULTS_DIR", "RAY_AIR_LOCAL_CACHE_DIR",
    )
    for k in env_keys:
        v = os.environ.get(k, "").strip()
        if not v:
            continue
        rd = _pick_run_dir(v)
        if rd:
            return rd


    # -------------------------------------------------------------
    # LAST-RESORT (no env hints):
    # If Stage-3 wrote stage4_ctx*.jsonl somewhere under CWD (typical runs/*),
    # infer run_dir from the newest ctx file path.
    # This prevents Stage-4 default logs from falling back to CWD/stage4.
    # -------------------------------------------------------------
    try:
        cwd = os.getcwd()
        cand = []
        cand += glob.glob(os.path.join(cwd, "**", "stage3", "stage4_ctx*.jsonl"), recursive=True)
        cand += glob.glob(os.path.join(cwd, "**", "stage4_ctx*.jsonl"), recursive=True)
        cand = [os.path.abspath(p) for p in cand if os.path.isfile(p)]
        if cand:
            newest = max(cand, key=lambda p: os.path.getmtime(p))
            rd = _pick_run_dir(newest)
            if rd:
                return rd
    except Exception:
        pass

    return None


def _get_pf_stage4_f(path: Optional[str] = None):
    global _pf_stage4_f, _pf_stage4_path
    if path is None or str(path).strip() == "":
        path = _default_stage4_path()
    path = str(path)

    if _pf_stage4_f is not None and _pf_stage4_path == path:
        return _pf_stage4_f

    # path changed -> rotate
    try:
        if _pf_stage4_f is not None:
            _pf_stage4_f.flush()
            _pf_stage4_f.close()
    except Exception:
        pass
    _pf_stage4_f = None
    _pf_stage4_path = path

    d = os.path.dirname(path)
    if d:
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass

    _pf_stage4_f = open(path, "a", buffering=1)
    return _pf_stage4_f



def _stage4_json_sanitize(v):
    # numpy scalars -> python scalars
    try:
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            v = float(v)
    except Exception:
        pass

    # NaN/Inf -> None
    try:
        if isinstance(v, float) and (not math.isfinite(v)):
            return None
    except Exception:
        pass

    # arrays -> list
    try:
        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:
        pass

    # dict/list recursion
    if isinstance(v, dict):
        return {str(k): _stage4_json_sanitize(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_stage4_json_sanitize(x) for x in v]

    # keep json-serializable primitives as-is; fallback to string
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


def _crc32_hex_of_obj(obj: Any) -> str:
    """
    Stable-ish config hash for join. Best-effort only.
    """
    try:
        # dataclass -> dict
        if is_dataclass(obj):
            obj = asdict(obj)
        s = json.dumps(_stage4_json_sanitize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        s = str(obj)
    try:
        return f"{(zlib.crc32(s.encode('utf-8')) & 0xffffffff):08x}"
    except Exception:
        return "00000000"

def pf_stage4_write_jsonl(row: dict, *, path: Optional[str] = None) -> None:

    """
    Write one JSONL line for Stage-4 diagnostics.
    Pure logging; must never affect training.
    """
    if not PF_STAGE4:
        return
    try:
        _assert_stage34_join_contract(row, where="stage4/pf_stage4_write_jsonl")

        f = _get_pf_stage4_f(path)
        safe = _stage4_json_sanitize(row)
        f.write(json.dumps(safe, ensure_ascii=False) + "\n")
    except Exception as e:
        # never fail the training loop
        if os.environ.get("PF_STAGE4_WARN", "0") == "1":
            print("[PF_STAGE4][WARN]", repr(e))

# ===================== STAGE-4 STATS (CSV) =====================
# Goal:
#   - Produce pf_stats_stage4.csv (per-process by default) with the SAME Stage-3 join keys:
#       run_id/worker_index/vector_index/episode_uid/episode_idx
#   - Do NOT generate any new run/episode IDs here. Only consume env-injected stage3_* fields.
#
# Enable:
#   PF_STAGE4=1  (stage4 instrumentation)
#   PF_STATS_STAGE4=1   (optional; default "1" when PF_STAGE4=1)
#
# Optional path:
#   PF_STATS_STAGE4_LOG=/abs/or/rel/path.csv
#
# NOTE: PF_STATS_STAGE4 already defined above. DO NOT overwrite it here.
_pf_stats_stage4_f: Optional[IO] = None
_pf_stats_stage4_path: Optional[str] = None

def _get_pf_stats_stage4_f(path: Optional[str] = None):
    global _pf_stats_stage4_f, _pf_stats_stage4_path
    if path is None or str(path).strip() == "":
        # fallback: per-process in cwd (caller should pass run-scoped path)
        path = os.environ.get("PF_STATS_STAGE4_LOG", "").strip()
        pid = os.getpid()
        if path:
            # If relative, anchor to inferred base_dir (Stage-3 run_dir) when available.
            if not os.path.isabs(path):
                base_dir = os.environ.get("PF_STAGE4_DIR", "").strip()
                if not base_dir:
                    base_dir = _infer_stage4_base_dir_from_stage3_env() or ""
                if base_dir:
                    path = os.path.join(base_dir, path)
        else:
            base_dir = os.environ.get("PF_STAGE4_DIR", "").strip()
            if not base_dir:
                base_dir = _infer_stage4_base_dir_from_stage3_env() or os.getcwd()
            path = os.path.join(base_dir, "stage4", f"pf_stats_stage4.{pid}.csv")

    path = str(path)
    if _pf_stats_stage4_f is not None and _pf_stats_stage4_path == path:
        return _pf_stats_stage4_f

    # rotate on path change
    try:
        if _pf_stats_stage4_f is not None:
            _pf_stats_stage4_f.flush()
            _pf_stats_stage4_f.close()
    except Exception:
        pass
    _pf_stats_stage4_f = None
    _pf_stats_stage4_path = path

    d = os.path.dirname(path)
    if d:
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass
    _pf_stats_stage4_f = open(path, "a", buffering=1)
    return _pf_stats_stage4_f

def pf_stats_stage4_write(row: dict, *, path: Optional[str] = None):
    """
    Stage-4 CSV logger.
    Writes PF_STATS_COLS in fixed order.
    Unknown keys are preserved into extra_json.
    """

    if not PF_STATS_STAGE4:
        return
    try:
        f = _get_pf_stats_stage4_f(path)
        # Use the *actual* opened file path for empty-check/header logic.
        path_hint = str(path) if (path is not None and str(path).strip() != "") else (_pf_stats_stage4_path or "")
        real_path = getattr(f, "name", "") or path_hint

        cols = PF_STATS_COLS
        # P0: write header ONLY if file is empty (size-based)
        _csv_write_schema_and_header_if_needed(
            f,
            path=real_path,
            schema_comment="# stage4/pf_stats_stage4 schema=PF_STATS_COLS_v2",

            cols=cols,
        )

        extra = {}
        try:
            for k, v in (row.items() if isinstance(row, dict) else []):
                if str(k) not in cols:
                    extra[str(k)] = v
        except Exception:
            extra = {}
        row2 = dict(row) if isinstance(row, dict) else {}
        if "extra_json" not in row2:
            try:
                # P1: keep strict JSON
                row2["extra_json"] = json.dumps(_stage4_json_sanitize(extra), ensure_ascii=False, separators=(",", ":"))

            except Exception:
                row2["extra_json"] = str(extra)

        # Enforce Stage-3/4 join contract on Stage-4 CSV as well.
        _assert_stage34_join_contract(row2, where="stage4/pf_stats_stage4_write")

        _csv_write_row(f, cols=cols, row=row2)
    except Exception as e:
        if os.environ.get("PF_STATS_STAGE4_WARN", "0") == "1":
            print("[PF_STATS_STAGE4][WARN]", repr(e))



# ============================================================
# PF track per target ship
# ============================================================

@dataclass
class PFTrack:
    pf: ParticleCTRVFilter

    # ----- timestamps (see TIME CONTRACT) -----
    last_meas_ts: Optional[float] = None          # DEPRECATED alias; kept for backward compat (same as last_reported_ts)
    last_reported_ts: Optional[float] = None      # last ts_rep fused (message timestamp)
    last_update_ts: Optional[float] = None        # last t_env fused (PF axis == env axis)
    last_arrival_ts: Optional[float] = None       # last ts_arr of fused packet (metadata only)

    # ----- last fused measurement (projected to t_env) -----
    last_meas_x: Optional[float] = None
    last_meas_y: Optional[float] = None
    last_fused_msg_id: Optional[str] = None
    recent_fused_msg_ids: deque[str] = field(default_factory=lambda: deque(maxlen=64))
    valid: bool = True



@dataclass
class AgentPFState:
    preproc: AISPreprocessor
    tracks: Dict[int, PFTrack]

    debug_msg_count: int = 0
    debug_update_count: int = 0
    dbg_err_pos_sum: float = 0.0
    dbg_err_pos_max: float = 0.0
    dbg_err_pos_cnt: int = 0

    # debug timestamp maps (optional)
    last_pf_ts: Dict[int, float] = field(default_factory=dict)
    last_rep_ts: Dict[int, float] = field(default_factory=dict)
    last_upd_ts: Dict[int, float] = field(default_factory=dict)
    last_arr_ts: Dict[int, float] = field(default_factory=dict)

    # ego yaw cache (debug-only)
    ego_yaw_sim_rad: float = float("nan")
    last_ego_yaw_sim_rad: float = float("nan")

    # >>> ADD: truth cache for debug-only (never used by PF)
    true_by_sid: Optional[Dict[int, TrueState]] = None

class AISTrackManagerPF:
    """
    Per-Agent PF tracker:

    - ingest(t_env, ready): per-agent inbox -> preproc -> PF update/create @ t_env
    - all_estimates(agent_id, t_env): return that agent's local belief world at t_env
    """

    def __init__(
        self,
        max_age: float = 20.0,
        # Stage-4 ctor ctx (run identity + logging join keys). Frozen once accepted.
        stage4_ctx: Optional[Dict[str, Any]] = None,
        freeze_ctor_ctx: bool = True,

        # preproc params
        window_sec: float = 60.0,
        max_buffer: int = 256,
        max_pos_jump: float = 80.0,
        max_cog_jump_deg: float = 90.0,
        init_avg_K: int = 1,
        lowpass_alpha: float = 1.0,

        # PF noise params
        meas_std_pos: float = 8.0,
        meas_std_sog: float = 0.2,
        meas_std_cog_deg: float = 8.0,  # legacy name; treated as yaw std inside PF
        process_std_a: float = 0.5,
        process_std_yaw_deg: float = 10.0,

        # PF algorithm params
        num_particles: int = 256,
        resample_threshold_ratio: float = 0.5,
        pf_seed_base: int = 2025,

        # relock
        soft_relock_dist: float = 20.0,
        hard_relock_dist: float = 40.0,
        soft_relock_beta: float = 0.3,
        stage4_ctor_ctx: Optional[Dict[str, Any]] = None,
    ):
        self.max_age = float(max_age)

        # --- preproc params ---
        self.preproc_window_sec = float(window_sec)
        self.preproc_max_buffer = int(max_buffer)
        self.preproc_max_pos_jump = float(max_pos_jump)
        self.preproc_max_cog_jump_deg = float(max_cog_jump_deg)
        self.preproc_init_avg_K = int(init_avg_K)
        self.preproc_lowpass_alpha = float(lowpass_alpha)

        # --- PF noise + age gating ---
        yaw_std = math.radians(process_std_yaw_deg)
        self.pf_noise = PFNoiseConfig(
            meas_std_pos=meas_std_pos,
            meas_std_sog=meas_std_sog,
            meas_std_cog_deg=meas_std_cog_deg,  # legacy field name; semantically yaw std
            process_std_a=process_std_a,
            process_std_yaw=yaw_std,
            age=PFAgeGatingConfig(
                age_fresh=1.0,
                age_stale=8.0,
                age_very_stale=20.0,
                age_max=float(max_age),
                scale_fresh=1.0,
                scale_stale=3.0,
                scale_very_stale=6.0,
                pos_scale_factor=1.0,
                vel_scale_factor=1.0,
                cog_scale_factor=2.5,  # legacy field; semantically yaw scale
            ),
        )

        self.num_particles = int(num_particles)
        self.resample_threshold_ratio = float(resample_threshold_ratio)
        self.pf_seed_base = int(pf_seed_base)

        # per-agent PF states
        self.agent_states: Dict[AgentId, AgentPFState] = {}

        # mmsi -> ship_id
        self.mmsi_to_sid: Dict[int, ShipId] = {}

        # episode time origin
        self.t0 = 0.0
        self.last_t = 0.0  # last env time t_env

        # relock params
        self.soft_relock_dist = float(soft_relock_dist)
        self.hard_relock_dist = float(hard_relock_dist)
        self.soft_relock_beta = float(soft_relock_beta)

        # agent_id -> ego ship_id (for self filtering)
        self.agent_to_ego_sid: Dict[AgentId, ShipId] = {}


        # === debug toggles (env) ===
        self._dbg_sog_check = (os.environ.get("AIS_DBG_SOG_CHECK", "0") == "1")
        self._dbg_hash_push = (os.environ.get("AIS_DBG_HASH_PUSH", "0") == "1")

        # === truth probe (print only once per (agent,sid,kind)) ===
        self._truth_probe_once = set()

        # >>> ADD: global truth cache (debug-only)
        self._truth_by_sid_global: Optional[Dict[int, TrueState]] = None

        # ============================================================
        # Stage-3 ctx index (Plan A):
        #   - loaded once per process, used to recover Stage-3 identity in Stage-4.
        # ============================================================
        self._stage3_ctx_loaded: bool = False
        self._stage3_ctx_sources: List[str] = []
        self._stage3_ctx_index: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        self._stage3_ctx_miss_once: set[Tuple[str, Tuple[int, int, int], str]] = set()

        # ============================================================
        # Stage-4 context (process-level) + autofill
        #   - Use set_stage4_context(...) at episode reset if desired.
        #   - Optional bootstrap from env PF_STAGE4_CTX_JSON='{"run_id": "...", ...}'
        # ============================================================
        self._stage4_ctx: Dict[str, Any] = {}
        self._stage4_frozen_keys: set[str] = set()
        self._stage4_allow_override: bool = os.environ.get("PF_STAGE4_CTX_ALLOW_OVERRIDE", "0").strip() == "1"
        self._stage4_ctor_keys: set[str] = set()
        self._stage4_autofreeze_keys: set[str] = {
            "run_uuid", "run_id", "worker_index", "vector_index",
            "run_dir", "out_dir", "trial_dir", "log_dir", "out", "logdir",
            "ais_cfg_hash", "pf_cfg_hash", "code_commit",
            "stage3_run_uuid", "stage3_run_id",
        }
        self._stage4_ctx_version: int = 1
        # freeze control

        self._freeze_ctor_ctx: bool = bool(freeze_ctor_ctx)

        self._stage4_episode_seq: int = 0
        self._stage4_current_episode_id: Optional[int] = None  # legacy alias; maps to episode_idx if present
        self._stage4_current_episode_uid: Optional[str] = None
        self._stage4_path_dbg_printed: bool = False
 

        # optional: caller can provide a stable run_dir so stage4 always lands under it
        # accepted keys: run_dir/out_dir/trial_dir/log_dir
        # or env: PF_RUN_DIR / PF_STAGE4_DIR

        _ctx_json = os.environ.get("PF_STAGE4_CTX_JSON", "").strip()
        if _ctx_json:
            try:
                _ctx = json.loads(_ctx_json)
                if isinstance(_ctx, dict):
                    self._stage4_ctx.update(_ctx)
                    self._stage4_ctor_keys.update(_ctx.keys())
            except Exception:
                pass

        if isinstance(stage4_ctor_ctx, dict) and stage4_ctor_ctx:
            self._stage4_ctx.update(stage4_ctor_ctx)
            self._stage4_ctor_keys.update(stage4_ctor_ctx.keys())

        # ctor ctx (highest priority): force accept
        if isinstance(stage4_ctx, dict) and len(stage4_ctx) > 0:
            try:
                self._stage4_ctx.update(stage4_ctx)
            except Exception:
                pass

        # Autofill pf_cfg_hash if upstream didn't provide it (join-friendly).
        # Pure metadata; must NOT affect PF logic.
        try:
            if "pf_cfg_hash" not in self._stage4_ctx:
                self._stage4_ctx["pf_cfg_hash"] = _crc32_hex_of_obj({
                    "pf_noise": self.pf_noise,
                    "num_particles": self.num_particles,
                    "resample_threshold_ratio": self.resample_threshold_ratio,
                    "max_age": self.max_age,
                    "soft_relock_dist": self.soft_relock_dist,
                    "hard_relock_dist": self.hard_relock_dist,
                    "soft_relock_beta": self.soft_relock_beta,
                })
            self._stage4_ctor_keys.add("pf_cfg_hash")
        except Exception:
            pass

        # Normalize required Stage-3/4 keys if already available in env
        # (does NOT generate any new IDs; only maps aliases to canonical fields).
        try:
            self.set_stage4_context({})
        except Exception:
            pass

        # Freeze ctor-injected keys (except episode-scoped keys)
        try:
            for _k in list(self._stage4_ctor_keys):
                if _k in ("stage3_episode_uid", "stage3_episode_idx"):
                    continue
                if not _is_empty_stage34(self._stage4_ctx.get(_k)):
                    self._stage4_frozen_keys.add(_k)
        except Exception:
            pass

        # Freeze ctor ctx run-level fields (do NOT freeze episode keys).
        # This prevents accidental drift/overwrite later in the run.
        try:
            if self._freeze_ctor_ctx:
                for k in list(self._stage4_ctx.keys()):
                    if (k in STAGE4_RUN_FROZEN_KEYS) and (k not in STAGE4_EPISODE_MUTABLE_KEYS) and (not _is_empty_stage34(self._stage4_ctx.get(k))):
                        self._stage4_frozen_keys.add(str(k))
        except Exception:
            pass

        # Plan A: best-effort load Stage-3 ctx index at startup (Stage-4 joins).
        try:
            if PF_STAGE4_LOAD_STAGE3_CTX:
                self._load_stage3_stage4_ctx_index()
        except Exception:
            pass

        _print_loaded_paths_once()


    def _export_stage4_dir_env(self, run_dir: str) -> None:
        """
        Logging-only side effect: if this process has resolved a run_dir,
        export it to env vars so any *default* Stage-4 writers won't fall back to CWD.
        Never overwrite non-empty env vars.
        """
        try:
            rd = str(run_dir).strip()
            if not rd:
                return
            if os.environ.get("PF_STAGE4_DIR", "").strip() == "":
               os.environ["PF_STAGE4_DIR"] = rd
            if os.environ.get("PF_RUN_DIR", "").strip() == "":
                os.environ["PF_RUN_DIR"] = rd
            # optional convenience for other heuristics (only if it truly looks like a run_dir)
            if os.path.isdir(os.path.join(rd, "stage3")) and os.environ.get("STAGE3_RUN_DIR", "").strip() == "":
                os.environ["STAGE3_RUN_DIR"] = rd
        except Exception:
            pass

    def _stage4_put(self, k: str, v: Any) -> None:
        if v is None:
            return
        k = str(k)
        if isinstance(v, str) and v.strip() == "":
            return
        cur = self._stage4_ctx.get(k)
        if (
            (not self._stage4_allow_override)
            and (k in self._stage4_frozen_keys)
            and (not _is_empty_stage34(cur))
            and str(cur) != str(v)
        ):
            return
        self._stage4_ctx[k] = v
        if (k in self._stage4_autofreeze_keys) or (k in self._stage4_ctor_keys):
            self._stage4_frozen_keys.add(k)
    # -------------------------------------------------------
    # Stage4 run dir / identity helpers
    # -------------------------------------------------------

    def _stage4_ctx_set_kv(self, k: str, v: Any, *, freeze_on_first: bool = False) -> None:
        """
        Centralized setter to enforce frozen semantics.
        - If key is frozen and current value is non-empty:
            - allow ONLY "same value" overwrite for normalization (e.g., "3" -> 3)
            - otherwise ignore.
        - If key is not set (empty) and incoming is non-empty:
            - accept
            - optionally freeze_on_first.
        """
        kk = str(k)
        cur = self._stage4_ctx.get(kk, None)
        cur_empty = _is_empty_stage34(cur)
        new_empty = _is_empty_stage34(v)

        if (kk in self._stage4_frozen_keys) and (not cur_empty):
            if new_empty:
                return
            # allow same-value normalization
            try:
                if str(cur).strip() == str(v).strip():
                    self._stage4_ctx[kk] = v
            except Exception:
                pass
            return

        if not new_empty:
            self._stage4_ctx[kk] = v
            if freeze_on_first and (kk in STAGE4_RUN_FROZEN_KEYS) and (kk not in STAGE4_EPISODE_MUTABLE_KEYS):
                self._stage4_frozen_keys.add(kk)
        else:
            # ignore empty writes
            return

    def _infer_run_dir_from_stage3_path(self, p_in: str) -> Optional[str]:
        """
        Best-effort: derive <run_dir> from any path that points to stage3 artifacts.
        Examples:
          - /.../<run_dir>/stage3/train/xxx.jsonl  -> <run_dir>
          - /.../<run_dir>/stage3                 -> <run_dir>
          - /.../<run_dir>/stage3_train/...       -> <run_dir> (fallback: return dirname)
        """
        if not isinstance(p_in, str) or not p_in.strip():
            return None
        p = os.path.abspath(p_in.strip())
        try:
            if os.path.isfile(p):
                p = os.path.dirname(p)
        except Exception:
            pass

        # If caller already passes run_dir (not a stage3 subpath), and it contains stage3/ -> accept directly.
        try:
            if os.path.isdir(p) and os.path.isdir(os.path.join(p, "stage3")):
                return p
        except Exception:
            pass

        # If someone passes ".../<run_dir>/stage4" (Stage-4 side), parent is the run_dir.
        try:
            if os.path.basename(p.rstrip(os.sep)) == "stage4":
                return os.path.dirname(p.rstrip(os.sep))
        except Exception:
            pass

        # canonical: contains ".../stage3/..."
        parts = p.split(os.sep)
        if "stage3" in parts:
            idx = parts.index("stage3")
            if idx > 0:
                return os.sep.join(parts[:idx])
            return os.sep

        # ends with ".../stage3"
        if os.path.basename(p.rstrip(os.sep)) == "stage3":
            return os.path.dirname(p.rstrip(os.sep))

        # weak fallback: if someone passes ".../stage3/train" but the segment is missing due to naming,
        # just return its parent dir so stage4 doesn't land in cwd.
        return os.path.dirname(p)

    def _infer_run_dir_from_ctx(self) -> Optional[str]:
        """
        Best-effort inference to align Stage4 outputs with Stage3 run directory.
        Accept common keys:
          - run_dir/out_dir/trial_dir/log_dir/out/logdir/result_dir
          - stage3_dir (or any path containing '/stage3/')
        """
        # direct keys
        for k in ("run_dir", "out_dir", "trial_dir", "log_dir", "out", "logdir", "result_dir",
                  "stage3_run_dir", "stage3_out_dir", "stage3_log_dir"):
            v = self._stage4_ctx.get(k, None)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # stage3_dir-like keys
        for k in ("stage3_dir", "stage3_path", "stage3_root", "stage3_train_dir", "stage3_file"):
            v = self._stage4_ctx.get(k, None)
            if not (isinstance(v, str) and v.strip()):
                continue
            p = os.path.abspath(v.strip())
            try:
                if os.path.isfile(p):
                    p = os.path.dirname(p)
            except Exception:
                pass
            # find 'stage3' segment and take its parent as run_dir
            parts = p.split(os.sep)
            if "stage3" in parts:
                idx = parts.index("stage3")
                if idx > 0:
                    return os.sep.join(parts[:idx])
                return os.sep
            # if path itself ends with stage3, take parent
            if os.path.basename(p.rstrip(os.sep)) == "stage3":
                return os.path.dirname(p.rstrip(os.sep))
            return p

        return None

    def _resolve_run_dir(self) -> str:
        """
        Determine run directory for Stage4 outputs.
        Priority:
          1) self._stage4_ctx['run_dir'/'out_dir'/'trial_dir'/'log_dir'/'out'/'logdir'] or inferred from stage3_dir
          2) env PF_RUN_DIR
          3) env PF_STAGE4_DIR
          4) env TUNE_TRIAL_DIR / TUNE_LOGDIR / TUNE_RESULT_DIR / RAY_RESULTS_DIR (best-effort)
          5) cwd
        """
        for k in ("run_dir", "out_dir", "trial_dir", "log_dir", "out", "logdir"):
            v = self._stage4_ctx.get(k, None)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # infer from stage3 paths if provided
        inf = self._infer_run_dir_from_ctx()
        if isinstance(inf, str) and inf.strip():
            # cache for consistency
            self._stage4_ctx["run_dir"] = inf.strip()
            # ensure default stage4 writers won't land in CWD
            self._export_stage4_dir_env(inf.strip())
            return inf.strip()

        # -------------------------------------------------------------
        # NEW: infer run_dir from Stage-3 path-like environment hints
        # (does NOT affect PF logic; logging destination only)
        # -------------------------------------------------------------
        for k in (
            # common user-defined
            "STAGE3_DIR", "STAGE3_RUN_DIR", "STAGE3_OUT_DIR", "STAGE3_LOG_DIR", "STAGE3_PATH",
            # your Stage-3 / stats writers sometimes expose file paths
            "PF_STATS_LOG", "PF_COMM_STATS_LOG", "PF_EPISODES_LOG",
        ):
            v = os.environ.get(k, "").strip()
            if not v:
                continue
            inf2 = self._infer_run_dir_from_stage3_path(v)
            if isinstance(inf2, str) and inf2.strip():
                # cache so all later stage4 paths are consistent within process
                self._stage4_ctx["run_dir"] = inf2.strip()
                return inf2.strip()

        v = os.environ.get("PF_RUN_DIR", "").strip()
        if v:
            return v
        v = os.environ.get("PF_STAGE4_DIR", "").strip()
        if v:
            return v
        # common Ray/Tune hints (best-effort)
        for k in ("TUNE_TRIAL_DIR", "TUNE_LOGDIR", "TUNE_RESULT_DIR", "RAY_RESULTS_DIR", "RAY_AIR_LOCAL_CACHE_DIR"):

            v = os.environ.get(k, "").strip()
            if v:
                return v
        return os.getcwd()

    # -------------------------------------------------------
    # Plan A: load Stage-3 stage4_ctx*.jsonl and index by:
    #   (worker_index, vector_index, stage3_episode_idx)
    # -------------------------------------------------------
    def _split_ctx_patterns(self, s: str) -> List[str]:
        if not isinstance(s, str) or not s.strip():
            return []
        # allow comma/semicolon separated patterns
        toks = [t.strip() for t in re.split(r"[;,]+", s.strip()) if t.strip()]
        return toks

    def _stage3_stage4_ctx_patterns(self) -> List[str]:
        """
        Collect glob patterns for Stage-3 ctx files.
        Priority:
          1) stage4_ctx passed via stage4_ctx dict keys
          2) env PF_STAGE3_STAGE4_CTX_GLOB
          3) inferred from run_dir (best-effort)
        """
        pats: List[str] = []

        # 1) ctx-provided hints
        try:
            for k in (
                "stage3_stage4_ctx_glob", "stage3_ctx_glob", "stage4_ctx_glob",
                "stage3_stage4_ctx_path", "stage3_ctx_path",
            ):
                v = self._stage4_ctx.get(k, None)
                if isinstance(v, str) and v.strip():
                    pats.extend(self._split_ctx_patterns(v.strip()))
        except Exception:
            pass

        # 2) env glob
        if PF_STAGE3_STAGE4_CTX_GLOB:
            pats.extend(self._split_ctx_patterns(PF_STAGE3_STAGE4_CTX_GLOB))

        # 3) default inference from run_dir
        if not pats:
            run_dir = ""
            try:
                run_dir = self._resolve_run_dir()
            except Exception:
                run_dir = ""
            if isinstance(run_dir, str) and run_dir.strip():
                rd = run_dir.strip()
                # common locations
                pats.append(os.path.join(rd, "stage3", "stage4_ctx*.jsonl"))
                pats.append(os.path.join(rd, "stage3", "**", "stage4_ctx*.jsonl"))
                pats.append(os.path.join(rd, "stage4", "stage4_ctx*.jsonl"))
                pats.append(os.path.join(rd, "**", "stage4_ctx*.jsonl"))

        # de-dup while preserving order
        out: List[str] = []
        seen = set()
        for p in pats:
            if not p or p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out

    def _load_stage3_stage4_ctx_index(self, *, force: bool = False) -> None:
        """
        Load Stage-3 ctx jsonl files and build index:
          key = (worker_index, vector_index, stage3_episode_idx)
        value includes (best-effort):
          run_uuid, run_id, stage3_episode_uid, stage3_episode_idx, worker_index, vector_index, run_dir
        """
        if self._stage3_ctx_loaded and (not force):
            return

        self._stage3_ctx_loaded = True  # mark early to avoid recursion loops
        self._stage3_ctx_sources = []
        self._stage3_ctx_index = {}

        pats = self._stage3_stage4_ctx_patterns()
        files: List[str] = []
        for pat in pats:
            try:
                files.extend(glob.glob(pat, recursive=PF_STAGE3_STAGE4_CTX_RECURSIVE))
            except Exception:
                continue

        # filter to existing files, stable order
        fset: List[str] = []
        for fp in sorted(set([os.path.abspath(x) for x in files])):
            try:
                if os.path.isfile(fp):
                    fset.append(fp)
            except Exception:
                continue
            if len(fset) >= PF_STAGE3_STAGE4_CTX_MAX_FILES:
                break

        self._stage3_ctx_sources = list(fset)
        if not fset:
            return

        # -------------------------------------------------------------
        # CRITICAL: bootstrap run_dir from Stage-3 ctx FILE PATH early.
        # This does NOT invent any IDs; it only anchors Stage-4 outputs
        # to the same <run_dir> that contains "stage3/".
        # -------------------------------------------------------------
        try:
            if _is_empty_stage34(self._stage4_ctx.get("run_dir", None)):
                newest_fp = max(fset, key=lambda p: os.path.getmtime(p))
                rd_fp = self._infer_run_dir_from_stage3_path(newest_fp)
                if isinstance(rd_fp, str) and rd_fp.strip():
                    self._stage4_ctx_set_kv("run_dir", rd_fp.strip(), freeze_on_first=True)
                    self._export_stage4_dir_env(rd_fp.strip())
        except Exception:
            pass
  
        # read + index
        n_lines = 0
        stop = False
        for fp in fset:
            if stop:
                break
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        if stop:
                            break
                        if not line or not line.strip():
                            continue
                        n_lines += 1
                        if n_lines > PF_STAGE3_STAGE4_CTX_MAX_LINES:
                            stop = True
                            break
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        if not isinstance(obj, dict):
                            continue

                        w = obj.get("worker_index", obj.get("stage3_worker_index", obj.get("worker", None)))
                        v = obj.get("vector_index", obj.get("stage3_vector_index", obj.get("env_index", None)))
                        ei = obj.get("stage3_episode_idx", obj.get("episode_idx", obj.get("episode_id", None)))
                        uid = obj.get("stage3_episode_uid", obj.get("episode_uid", None))
                        ru = obj.get("run_uuid", obj.get("stage3_run_uuid", obj.get("stage3_run_id", None)))
                        rid = obj.get("run_id", obj.get("trial_id", obj.get("tune_trial_id", None)))

                        if _is_empty_stage34(w) or _is_empty_stage34(v) or _is_empty_stage34(ei):
                            continue
                        try:
                            w_i = int(w)
                            v_i = int(v)
                            ei_i = int(ei)
                        except Exception:
                            continue

                        rec: Dict[str, Any] = {
                            "worker_index": w_i,
                            "vector_index": v_i,
                            "stage3_episode_idx": ei_i,
                        }
                        if (uid is not None) and (str(uid).strip() != ""):
                            rec["stage3_episode_uid"] = str(uid).strip()
                        if (ru is not None) and (str(ru).strip() != ""):
                            rec["run_uuid"] = str(ru).strip()
                        if (rid is not None) and (str(rid).strip() != ""):
                            rec["run_id"] = str(rid).strip()

                        # best-effort run_dir carry-through
                        for kk in ("run_dir", "out_dir", "trial_dir", "log_dir", "out", "logdir", "result_dir"):
                            vv = obj.get(kk, None)
                            if isinstance(vv, str) and vv.strip():
                                rec["run_dir"] = vv.strip()
                                break

                        # IMPORTANT FIX:
                        # If Stage-3 ctx rows do NOT contain run_dir, infer it from the ctx FILE PATH.
                        # This is the missing link that caused Stage-4 logs to fall back to CWD (shipRL).
                        if _is_empty_stage34(rec.get("run_dir", None)):
                            rd_fp = self._infer_run_dir_from_stage3_path(fp)
                            if isinstance(rd_fp, str) and rd_fp.strip():
                                rec["run_dir"] = rd_fp.strip()

                        key = (w_i, v_i, ei_i)
                        if key not in self._stage3_ctx_index:
                            self._stage3_ctx_index[key] = rec
                        else:
                            # merge missing fields only (do NOT overwrite non-empty)
                            old = self._stage3_ctx_index[key]
                            for kk, vv in rec.items():
                                if (kk not in old) or _is_empty_stage34(old.get(kk)):
                                    old[kk] = vv
            except Exception:
                continue

    def _stage4_fill_from_stage3_ctx_index(self, *, where: str = "") -> None:
        """
        If current stage4 ctx has (worker_index, vector_index, stage3_episode_idx),
        fill missing run_uuid / stage3_episode_uid / run_dir by looking up Stage-3 ctx index.
        """
        if not PF_STAGE4_LOAD_STAGE3_CTX:
            return
        # load once
        try:
            if not self._stage3_ctx_loaded:
                self._load_stage3_stage4_ctx_index()
        except Exception:
            return

        if not isinstance(self._stage3_ctx_index, dict) or (len(self._stage3_ctx_index) == 0):
            return

        w = self._stage4_ctx.get("worker_index", self._stage4_ctx.get("stage3_worker_index", None))
        v = self._stage4_ctx.get("vector_index", self._stage4_ctx.get("stage3_vector_index", self._stage4_ctx.get("env_index", None)))
        ei = self._stage4_ctx.get("stage3_episode_idx", self._stage4_ctx.get("episode_idx", self._stage4_ctx.get("episode_id", None)))

        if _is_empty_stage34(w) or _is_empty_stage34(v) or _is_empty_stage34(ei):
            return
        try:
            key = (int(w), int(v), int(ei))
        except Exception:
            return

        rec = self._stage3_ctx_index.get(key, None)
        if not isinstance(rec, dict):
            if PF_STAGE3_STAGE4_CTX_WARN:
                miss = ("MISS", key, str(where))
                if miss not in self._stage3_ctx_miss_once:
                    self._stage3_ctx_miss_once.add(miss)
                    print(f"[STAGE34-CTX-MISS] where={where} key={key} loaded_files={len(self._stage3_ctx_sources)}")
            return

        # run_uuid / run_id (freeze-on-first)
        if not _is_empty_stage34(rec.get("run_uuid", None)):
            ru_s = str(rec["run_uuid"]).strip()
            self._stage4_ctx_set_kv("run_uuid", ru_s, freeze_on_first=True)
            self._stage4_ctx_set_kv("stage3_run_uuid", ru_s, freeze_on_first=True)
            self._stage4_ctx_set_kv("stage3_run_id", ru_s, freeze_on_first=True)
        if not _is_empty_stage34(rec.get("run_id", None)):
            self._stage4_ctx_set_kv("run_id", str(rec["run_id"]).strip(), freeze_on_first=True)

        # worker/vector (freeze-on-first)
        self._stage4_ctx_set_kv("worker_index", int(rec.get("worker_index", key[0])), freeze_on_first=True)
        self._stage4_ctx_set_kv("vector_index", int(rec.get("vector_index", key[1])), freeze_on_first=True)
        self._stage4_ctx_set_kv("env_index", int(rec.get("vector_index", key[1])), freeze_on_first=True)

        # episode keys MUST remain mutable (direct set)
        try:
            ei_i = int(rec.get("stage3_episode_idx", key[2]))
            self._stage4_ctx["stage3_episode_idx"] = ei_i
            self._stage4_ctx["episode_idx"] = ei_i
            self._stage4_ctx["episode_id"] = ei_i
        except Exception:
            pass
        if not _is_empty_stage34(rec.get("stage3_episode_uid", None)):
            uid_s = str(rec["stage3_episode_uid"]).strip()
            self._stage4_ctx["stage3_episode_uid"] = uid_s
            self._stage4_ctx["episode_uid"] = uid_s

        # run_dir hint (freeze-on-first)
        if isinstance(rec.get("run_dir", None), str) and rec["run_dir"].strip():
            self._stage4_ctx_set_kv("run_dir", rec["run_dir"].strip(), freeze_on_first=True)
            # ensure default stage4 writers won't land in CWD
            self._export_stage4_dir_env(rec["run_dir"].strip())

    def _ensure_run_uuid(self) -> str:
        """
        Backward-compatible accessor, but MUST NOT generate new IDs.
        We consume env-injected Stage-3 run identity for cross-stage join:
          - stage3_run_uuid (preferred) / run_uuid / stage3_run_id (legacy alias)

        Returns "" if unavailable.
        """
        ru = (
            self._stage4_ctx.get("stage3_run_uuid", None)
            or self._stage4_ctx.get("run_uuid", None)
            or self._stage4_ctx.get("stage3_run_id", None)
        )

        if ru is None or str(ru).strip() == "":
            return ""
        try:
            # store as run_uuid for join; DO NOT overwrite run_id (trial id)
            self._stage4_ctx["run_uuid"] = str(ru).strip()

        except Exception:
            pass

        try:

            self._stage4_ctx["stage3_run_uuid"] = str(ru).strip()
        except Exception:
            pass

        return str(ru).strip()

    def _on_episode_reset_autofill(self) -> None:
        """
        MUST NOT generate new IDs.
        We only snapshot env-injected Stage-3 episode keys when present:
          - stage3_episode_uid / stage3_episode_idx

        """
        self._stage4_ctx_version = int(getattr(self, "_stage4_ctx_version", 0)) + 1

        ep_uid = self._stage4_ctx.get("stage3_episode_uid", None) or self._stage4_ctx.get("episode_uid", None)
        ep_idx = self._stage4_ctx.get("stage3_episode_idx", None) or self._stage4_ctx.get("episode_idx", None)

        if ep_uid is not None and str(ep_uid).strip() != "":
            self._stage4_current_episode_uid = str(ep_uid).strip()
            self._stage4_ctx["episode_uid"] = self._stage4_current_episode_uid
        if ep_idx is not None and str(ep_idx).strip() != "":
            try:
                self._stage4_current_episode_id = int(ep_idx)
                self._stage4_ctx["episode_idx"] = int(ep_idx)
                # legacy alias (same value)
                self._stage4_ctx["episode_id"] = int(ep_idx)
            except Exception:
                pass

    def _stage4_log_path(self) -> str:
        """
        Run-scoped stage4 jsonl path:
          <run_dir>/stage4/pf_stage4.<run_id_short>.w<worker>.v<vector>.pid<pid>.jsonl
        NOTE: run_id/worker/vector are from env-injected stage3_* only.
        """
        # CRITICAL: fill run_dir from Stage-3 ctx index BEFORE resolving path
        try:
            self._ensure_stage34_join_ctx_ready(where="_stage4_log_path")
        except Exception:
            pass

        run_dir = self._resolve_run_dir()
        ru = self._ensure_run_uuid()
        ru_short = "".join([c for c in ru if c.isalnum()])[:16] if isinstance(ru, str) and ru else "runNA"

        pid = os.getpid()
        w = self._stage4_ctx.get("worker_index", "")
        v = self._stage4_ctx.get("vector_index", "")
        w_s = f"w{int(w)}" if str(w).strip() != "" and str(w).lower() != "none" else "wNA"
        v_s = f"v{int(v)}" if str(v).strip() != "" and str(v).lower() != "none" else "vNA"
        fn = f"pf_stage4.{ru_short}.{w_s}.{v_s}.pid{pid}.jsonl"

        return os.path.join(run_dir, "stage4", fn)


    # ============================================================
    # Stage-4 writers MUST use instance-resolved run-scoped paths.
    # This is the missing link that kept Stage-4 landing in CWD.
    # ============================================================
    def _stage4_write_jsonl(self, row: dict, jsonl_path: str = "", **kwargs) -> None:
        # Backward/forward compatible:
        # - old call sites: _stage4_write_jsonl(row, jsonl_path)
        # - new call sites: _stage4_write_jsonl(row, path=...)
        if (not jsonl_path) and ("path" in kwargs):
            jsonl_path = kwargs.get("path") or ""
        if not jsonl_path:
            return
        if not PF_STAGE4:
            return
        try:
            p = self._stage4_log_path()
            if (os.environ.get("PF_STAGE4_PATH_DBG", "0") == "1") and (not self._stage4_path_dbg_printed):
                self._stage4_path_dbg_printed = True
                print(f"[PF-STAGE4-PATH] pid={os.getpid()} path={p}")
            pf_stage4_write_jsonl(row, path=p)
        except Exception:
            # logging must never affect training
            return

    def _stage4_write_csv(self, row: Dict[str, Any]) -> None:
        if not PF_STATS_STAGE4:
            return
        try:
            p = self._stage4_stats_csv_path()
            pf_stats_stage4_write(row, path=p)
        except Exception:
            return

    def _stage4_stats_csv_path(self) -> str:
        """
        Run-scoped stage4 CSV path:
          <run_dir>/stage4/pf_stats_stage4.<run_id_short>.w<worker>.v<vector>.pid<pid>.csv
        """

        # CRITICAL: fill run_dir from Stage-3 ctx index BEFORE resolving path
        try:
            self._ensure_stage34_join_ctx_ready(where="_stage4_stats_csv_path")
        except Exception:
            pass

        run_dir = self._resolve_run_dir()
        ru = self._ensure_run_uuid()
        ru_short = "".join([c for c in ru if c.isalnum()])[:16] if isinstance(ru, str) and ru else "runNA"
        pid = os.getpid()
        w = self._stage4_ctx.get("worker_index", "")
        v = self._stage4_ctx.get("vector_index", "")
        w_s = f"w{int(w)}" if str(w).strip() != "" and str(w).lower() != "none" else "wNA"
        v_s = f"v{int(v)}" if str(v).strip() != "" and str(v).lower() != "none" else "vNA"
        fn = f"pf_stats_stage4.{ru_short}.{w_s}.{v_s}.pid{pid}.csv"
        return os.path.join(run_dir, "stage4", fn)

    # -------------------------------------------------------
    # helpers
    # -------------------------------------------------------

    def _new_preproc(self) -> AISPreprocessor:
        pre_dbg = (os.environ.get("PF_PREPROC_DBG", "0") == "1")
        return AISPreprocessor(
            window_sec=self.preproc_window_sec,
            max_buffer=self.preproc_max_buffer,
            base_pos_jump=self.preproc_max_pos_jump,
            max_cog_jump_deg=self.preproc_max_cog_jump_deg,
            init_avg_K=self.preproc_init_avg_K,
            lowpass_alpha=self.preproc_lowpass_alpha,
            debug=pre_dbg,
            init_policy="passthrough",
        )

    def _sync_all_tracks_to_now(self, now: float, eps: float = 1e-6) -> None:
        """
        Pure-predict all PFs to env-step now (align PF outputs with RL time axis).
        PF time axis is env time, so this is the ONLY allowed global sync.
        """
        now = float(now)
        for _rx_agent, st in self.agent_states.items():
            for _sid, tr in st.tracks.items():
                pf = getattr(tr, "pf", None)
                if pf is None or getattr(pf, "last_ts", None) is None:
                    continue
                dt = now - float(pf.last_ts)
                if abs(dt) <= eps:
                    pf.last_ts = now
                    continue
                if dt > 0.0:
                    pf.predict(dt)

    def _agent_seed_offset(self, agent_id: AgentId) -> int:
        return zlib.crc32(str(agent_id).encode("utf-8")) % 10_000

    def _ego_sid_of_agent(self, agent_id: AgentId) -> Optional[int]:
        """Resolve ego ship_id for this agent (best-effort)."""
        ego_sid = self.agent_to_ego_sid.get(agent_id, None)
        if ego_sid is None and isinstance(agent_id, str):
            try:
                if agent_id.startswith("ship_"):
                    ego_sid = int(agent_id.split("_")[-1])
            except Exception:
                ego_sid = None
        return int(ego_sid) if ego_sid is not None else None

    def _try_setattr(self, obj: Any, k: str, v: Any) -> None:
        """Best-effort setattr (TrueState may be a dataclass with/without slots)."""
        try:
            setattr(obj, k, v)
        except Exception:
            pass


    def set_agent_ship_map(self, agent_of_ship: Dict[ShipId, AgentId]):
        """Let PF know agent->ego ship_id map (used to filter self tracks)."""
        self.agent_to_ego_sid = {a: int(sid) for sid, a in agent_of_ship.items()}

    def _new_pf_for_ship(self, sid: int, agent_id: AgentId) -> ParticleCTRVFilter:
        cfg = PFConfig(
            noise=self.pf_noise,
            num_particles=self.num_particles,
            resample_threshold_ratio=self.resample_threshold_ratio,
            seed=self.pf_seed_base + int(sid) + 100000 * self._agent_seed_offset(agent_id),
        )
        return ParticleCTRVFilter(cfg)

    def _ensure_agent_state(self, agent_id: AgentId) -> AgentPFState:
        st = self.agent_states.get(agent_id, None)
        if st is None:
            st = AgentPFState(preproc=self._new_preproc(), tracks={})
            self.agent_states[agent_id] = st
        # ensure maps (backward safety)
        if not hasattr(st, "last_pf_ts"):
            st.last_pf_ts = {}
        if not hasattr(st, "last_rep_ts"):
            st.last_rep_ts = {}
        if not hasattr(st, "last_upd_ts"):
            st.last_upd_ts = {}
        if not hasattr(st, "last_arr_ts"):
            st.last_arr_ts = {}
        return st

    def _get_pf_time(self, tr: PFTrack) -> float:
        """PF internal time cursor (source of truth: pf.last_ts)."""
        if tr is None:
            return -1.0
        pf = getattr(tr, "pf", None)
        if pf is not None and hasattr(pf, "last_ts"):
            try:
                return float(getattr(pf, "last_ts"))
            except Exception:
                pass
        return -1.0

    def _predict_x_readonly(self, pf: ParticleCTRVFilter, t_query: float) -> tuple[np.ndarray, float]:
        """Query-time deterministic CTRV propagation on point-estimate pf.x without mutating PF."""
        x0 = np.array(getattr(pf, "x", np.zeros(5, dtype=float)), dtype=float)
        pf_last = getattr(pf, "last_ts", None)
        if pf_last is None:
            return x0, float("nan")
        pf_ts = float(pf_last)
        dt = float(t_query - pf_ts)
        if dt <= 1e-9:
            return x0, pf_ts
        x_pred = pf._ctrv_step(x0.reshape(1, 5), dt)[0]
        x_pred[3] = _wrap_pi(float(x_pred[3]))
        return x_pred, pf_ts


    # -------------------------------------------------------
    # Truth debug helpers (print-only, NEVER used by PF logic)
    # -------------------------------------------------------

    def _parse_int_key(self, k) -> Optional[int]:
        """Best-effort parse keys like 1, np.int64(1), '1', 'ship_1', 'mmsi_123'."""
        if k is None:
            return None
        try:
            return int(k)
        except Exception:
            pass
        try:
            s = str(k).strip()
            if s.startswith("ship_"):
                return int(s.split("_")[-1])
            if s.startswith("mmsi_"):
                return int(s.split("_")[-1])
            return int(s)
        except Exception:
            return None

    def _normalize_truth_dict(self, true_states) -> Optional[Dict[int, TrueState]]:
        """
        Normalize any upstream truth container to: {sid(int) -> TrueState}.
        Accept:
          - {sid -> TrueState}
          - {mmsi -> TrueState}
          - {agent_id -> {sid -> TrueState}}  (we flatten by sid)
          - mixed key types: int/np.int/str('ship_1')/etc.
        Also uses self.mmsi_to_sid and TrueState.ship_id/mmsi when present.
        """
        if not isinstance(true_states, dict) or len(true_states) == 0:
            return None

        tb: Dict[int, TrueState] = {}

        def _insert_sid(sid_i: int, v: TrueState):
            try:
                tb[int(sid_i)] = v
            except Exception:
                pass

        # flatten one level if needed (per-agent dict)
        items = list(true_states.items())
        if len(items) > 0:
            k0, v0 = items[0]
            if isinstance(v0, dict):
                # pattern: true_states[agent] = {sid: TrueState}
                for _ak, sub in true_states.items():
                    if isinstance(sub, dict):
                        for kk, vv in sub.items():
                            sid_k = self._parse_int_key(kk)
                            if sid_k is not None:
                                _insert_sid(sid_k, vv)
                # still continue to try mapping using fields below
            else:
                # normal pattern: {sid/mmsi/...: TrueState}
                for kk, vv in true_states.items():
                    sid_k = self._parse_int_key(kk)
                    if sid_k is not None:
                        _insert_sid(sid_k, vv)

        # augment mapping using fields inside TrueState
        for _k, v in true_states.items():
            if v is None:
                continue

            # ship_id / sid / id
            for attr in ["ship_id", "sid", "id"]:
                try:
                    if hasattr(v, attr):
                        sv = getattr(v, attr)
                        sid_v = self._parse_int_key(sv)
                        if sid_v is not None:
                            _insert_sid(sid_v, v)
                except Exception:
                    pass

            # mmsi -> sid via map
            try:
                mv = getattr(v, "mmsi", None)
                mi = self._parse_int_key(mv)
                if mi is not None:
                    sid_from_m = self.mmsi_to_sid.get(int(mi), None)
                    if sid_from_m is not None:
                        _insert_sid(int(sid_from_m), v)
            except Exception:
                pass

        # also interpret keys as MMSI if possible
        for kk, vv in true_states.items():
            mi = self._parse_int_key(kk)
            if mi is None:
                continue
            sid_from_m = self.mmsi_to_sid.get(int(mi), None)
            if sid_from_m is not None:
                _insert_sid(int(sid_from_m), vv)

        return tb if len(tb) > 0 else None

    def _truth_lookup(self, st: AgentPFState, sid: int, mmsi: Optional[int] = None) -> Optional[TrueState]:
        """
        Debug-only truth lookup.
        Priority:
          1) st.true_by_sid[sid]
          2) global cache self._truth_by_sid_global[sid]
          3) fallback: try mmsi->sid mapping
        """
        sid_i = int(sid)

        tb = getattr(st, "true_by_sid", None)
        if isinstance(tb, dict):
            ts = tb.get(sid_i, None)
            if ts is not None:
                return ts

        gb = getattr(self, "_truth_by_sid_global", None)
        if isinstance(gb, dict):
            ts = gb.get(sid_i, None)
            if ts is not None:
                return ts

        if mmsi is not None:
            try:
                sid2 = self.mmsi_to_sid.get(int(mmsi), None)
                if sid2 is not None:
                    if isinstance(tb, dict) and tb.get(int(sid2), None) is not None:
                        return tb.get(int(sid2), None)
                    if isinstance(gb, dict) and gb.get(int(sid2), None) is not None:
                        return gb.get(int(sid2), None)
            except Exception:
                pass

        return None

    def _truth_unpack(self, ts_true: Optional[TrueState]) -> tuple[float, float, float, float, float, float]:
        """
        Debug-only: robustly extract (x,y,yaw,sog,vx,vy) from TrueState-like objects.
        yaw uses internal yaw convention if available; otherwise derive from vx/vy.
        """
        if ts_true is None:
            return (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

        def _fattr(obj, names, default=float("nan")):
            for n in names:
                if hasattr(obj, n):
                    v = getattr(obj, n)
                    if v is None:
                        continue
                    try:
                        return float(v)
                    except Exception:
                        pass
            return float(default)

        x = _fattr(ts_true, ["x", "px", "pos_x"])
        y = _fattr(ts_true, ["y", "py", "pos_y"])
        vx = _fattr(ts_true, ["vx", "vel_x"])
        vy = _fattr(ts_true, ["vy", "vel_y"])

        sog = _fattr(ts_true, ["sog", "sog_mps", "speed", "v"])
        if not (sog == sog):  # NaN
            if (vx == vx) and (vy == vy):
                sog = float(math.hypot(vx, vy))

        yaw = _fattr(ts_true, ["yaw_sim_rad", "yaw", "psi", "heading", "hdg_rad"])
        if not (yaw == yaw):  # NaN
            if (vx == vx) and (vy == vy) and (vx * vx + vy * vy > 1e-8):
                yaw = _wrap_pi(math.atan2(vy, vx))

        yaw = _wrap_pi(yaw) if (yaw == yaw) else yaw
        return x, y, yaw, sog, vx, vy


    # -------------------------------------------------------
    # public API
    # -------------------------------------------------------
    # ============================================================
    # Stage-4 context setter + autofill (also used by Stage-3 PF_STATS)
    # ============================================================

    def set_stage4_context(self, ctx: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """
        Set/merge stage4 context (process-level).
        Typical usage at episode reset:
            tm.set_stage4_context(run_uuid=..., episode_uid=..., episode_id=..., worker_index=..., env_index=..., run_dir=...)
        """
        # 1) merge caller ctx/kwargs with SAFE freeze semantics:
        #    - NEVER write empty-string/None placeholders (they freeze as empty and break join).
        #    - Episode keys MUST remain mutable (no freeze).
        EP_MUTABLE = {
            "stage3_episode_uid", "stage3_episode_idx",
            "episode_uid", "episode_idx", "episode_id",
        }

        def _nonempty(v: Any) -> bool:
            if v is None:
                return False
            if isinstance(v, str):
                return (v.strip() != "")
            return True

        if isinstance(ctx, dict):
            for k, v in ctx.items():
                kk = str(k)
                if not _nonempty(v):
                    continue
                self._stage4_ctx_set_kv(kk, v, freeze_on_first=(kk not in EP_MUTABLE))
        for k, v in kwargs.items():
            kk = str(k)
            if not _nonempty(v):
                continue
            self._stage4_ctx_set_kv(kk, v, freeze_on_first=(kk not in EP_MUTABLE))
 

        # ------------------------------------------------------------------
        # Stage-3 authoritative join keys (ONLY consume env-injected stage3_*)
        # Required contract fields:
        #   run_uuid, run_id, pid, worker_index, vector_index,
        #   stage3_episode_uid, stage3_episode_idx, agent_id
        #
        # Episode-level PK:
        #   (run_uuid, worker_index, vector_index, stage3_episode_uid, stage3_episode_idx)
        # ------------------------------------------------------------------

        try:

            # ---- run_uuid (stable) ----
            s_run_uuid = (
                self._stage4_ctx.get("run_uuid", None)
                or self._stage4_ctx.get("stage3_run_uuid", None)
                or self._stage4_ctx.get("stage3_run_id", None)   # legacy alias of run_uuid
            )
            if s_run_uuid is not None and str(s_run_uuid).strip() != "":
                self._stage4_ctx_set_kv("run_uuid", str(s_run_uuid).strip(), freeze_on_first=True)
                self._stage4_ctx_set_kv("stage3_run_uuid", str(s_run_uuid).strip(), freeze_on_first=True)
                self._stage4_ctx_set_kv("stage3_run_id", str(s_run_uuid).strip(), freeze_on_first=True)

            # ---- run_id (trial/exp; NOT part of join PK) ----
            s_run_id = (
                self._stage4_ctx.get("run_id", None)
                or self._stage4_ctx.get("trial_id", None)
                or self._stage4_ctx.get("tune_trial_id", None)
            )
            if s_run_id is not None and str(s_run_id).strip() != "":
                self._stage4_ctx_set_kv("run_id", str(s_run_id).strip(), freeze_on_first=True)

            # If run_uuid exists but run_id is missing/empty, default run_id := run_uuid
            if (
                isinstance(self._stage4_ctx.get("run_uuid", None), str)
                and self._stage4_ctx.get("run_uuid", "").strip()
                and not (isinstance(self._stage4_ctx.get("run_id", None), str) and self._stage4_ctx.get("run_id", "").strip())
            ):
                self._stage4_ctx_set_kv("run_id", str(self._stage4_ctx["run_uuid"]).strip(), freeze_on_first=True) 

            # ---- worker/vector ----
            s_w = self._stage4_ctx.get("stage3_worker_index", None) or self._stage4_ctx.get("worker_index", None)
            s_v = self._stage4_ctx.get("stage3_vector_index", None) or self._stage4_ctx.get("vector_index", None)
            if s_w is not None and str(s_w).strip() != "":
                self._stage4_ctx_set_kv("worker_index", int(s_w), freeze_on_first=True)

            if s_v is not None and str(s_v).strip() != "":
                self._stage4_ctx_set_kv("vector_index", int(s_v), freeze_on_first=True)
                self._stage4_ctx_set_kv("env_index", int(s_v), freeze_on_first=True)


            # ---- episode uid/idx (Stage-3 authoritative names) ----
            s_uid = (
                self._stage4_ctx.get("stage3_episode_uid", None)
                or self._stage4_ctx.get("episode_uid", None)
            )
            s_ei = (
                self._stage4_ctx.get("stage3_episode_idx", None)
                or self._stage4_ctx.get("episode_idx", None)
            )
            if s_uid is not None and str(s_uid).strip() != "":
                # episode keys stay mutable (no freeze)
                self._stage4_ctx["stage3_episode_uid"] = str(s_uid).strip()
                self._stage4_ctx["episode_uid"] = str(s_uid).strip()
            if s_ei is not None and str(s_ei).strip() != "":
                self._stage4_ctx["stage3_episode_idx"] = int(s_ei)
                self._stage4_ctx["episode_idx"] = int(s_ei)
                self._stage4_ctx["episode_id"] = int(s_ei)

            # ---- path alignment: if caller provides stage3_dir, infer run_dir ----

            if not any(isinstance(self._stage4_ctx.get(k, None), str) and self._stage4_ctx.get(k).strip()
                       for k in ("run_dir", "out_dir", "trial_dir", "log_dir", "out", "logdir")):
                inf = self._infer_run_dir_from_ctx()
                if isinstance(inf, str) and inf.strip():
                    self._stage4_ctx_set_kv("run_dir", inf.strip(), freeze_on_first=True)

        except Exception:
            pass


    # ------------------------------------------------------------------
    # Stage3/4 join contract hardening:
    # Ensure required join fields exist BEFORE any Stage3/4 logging/assert.
    # This never overwrites existing non-empty values.
    # ------------------------------------------------------------------

    def _ensure_stage34_join_ctx_ready(self, *, where: str = "unknown") -> None:
        """
        HARD RULE:
          - MUST NOT invent run/episode IDs for cross-stage join.
        This method may ONLY:
          (i) normalize aliases already present in self._stage4_ctx, and/or
          (ii) fill missing values by loading Stage-3 stage4_ctx*.jsonl (Plan A).

        Assertions still happen at write-time via _assert_stage34_join_contract.
        """
        # Plan A fill (best-effort)
        try:
            self._stage4_fill_from_stage3_ctx_index(where=str(where))
        except Exception:
            pass

        # Alias normalization (no generation)
        try:
            ru = (
                self._stage4_ctx.get("run_uuid", None)
                or self._stage4_ctx.get("stage3_run_uuid", None)
                or self._stage4_ctx.get("stage3_run_id", None)
            )
            if (ru is not None) and str(ru).strip():
                ru_s = str(ru).strip()
                self._stage4_ctx["run_uuid"] = ru_s
                self._stage4_ctx["stage3_run_uuid"] = ru_s
                self._stage4_ctx["stage3_run_id"] = ru_s

            # worker/vector aliases
            if _is_empty_stage34(self._stage4_ctx.get("worker_index", None)) and (self._stage4_ctx.get("stage3_worker_index", None) is not None):
                try:
                    self._stage4_ctx["worker_index"] = int(self._stage4_ctx["stage3_worker_index"])
                except Exception:
                    pass
            if _is_empty_stage34(self._stage4_ctx.get("vector_index", None)):
                vv = self._stage4_ctx.get("stage3_vector_index", None)
                if vv is None:
                    vv = self._stage4_ctx.get("env_index", None)
                if vv is not None and (not _is_empty_stage34(vv)):
                    try:
                        self._stage4_ctx["vector_index"] = int(vv)
                        self._stage4_ctx["env_index"] = int(vv)
                    except Exception:
                        pass

            # episode aliases (mutable; normalize int/string only)
            uid = self._stage4_ctx.get("stage3_episode_uid", None) or self._stage4_ctx.get("episode_uid", None)
            if (uid is not None) and str(uid).strip():
                uid_s = str(uid).strip()
                self._stage4_ctx["stage3_episode_uid"] = uid_s
                self._stage4_ctx["episode_uid"] = uid_s

            ei = self._stage4_ctx.get("stage3_episode_idx", None) or self._stage4_ctx.get("episode_idx", None) or self._stage4_ctx.get("episode_id", None)
            if (ei is not None) and (not _is_empty_stage34(ei)):
                try:
                    ei_i = int(ei)
                    self._stage4_ctx["stage3_episode_idx"] = ei_i
                    self._stage4_ctx["episode_idx"] = ei_i
                    self._stage4_ctx["episode_id"] = ei_i
                except Exception:
                    pass
        except Exception:
            pass

    # -------------------------------------------------------
    # Env-expected wrappers (Stage4 episode context injection)
    # -------------------------------------------------------

    # NOTE:
    #   Your file previously had DUPLICATE definitions of:
    #     - stage4_set_episode_context
    #     - set_stage4_episode_context
    #   The later definitions override the earlier ones, silently disabling
    #   episode-boundary detection. We keep ONLY ONE canonical implementation below.

    def stage4_set_episode_context(self, episode_ctx: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """
        Environment compatibility wrapper (SINGLE SOURCE OF TRUTH).
        Accepts:
          - positional dict: episode_ctx
          - alternate kw dict names: ctx / context / episode_context / episode_ctx

        Behavior:
          - merges into set_stage4_context()
          - detects episode boundary (uid/idx change) and triggers _on_episode_reset_autofill()
        """
        prev_uid = str(getattr(self, "_stage4_current_episode_uid", "") or "")
        prev_idx = str(getattr(self, "_stage4_current_episode_id", "") or "")

        # Allow callers to pass the dict via alternate kw names.
        alt = None
        for k in ("episode_context", "context", "ctx", "episode_ctx"):
            if k in kwargs and isinstance(kwargs.get(k), dict):
                alt = kwargs.pop(k)
                break

        # Merge precedence: positional episode_ctx overrides alt on key conflicts
        ctx2 = None
        if isinstance(alt, dict) and isinstance(episode_ctx, dict):
            try:
                ctx2 = dict(alt)
                ctx2.update(episode_ctx)
            except Exception:
                ctx2 = episode_ctx
        elif isinstance(episode_ctx, dict):
            ctx2 = episode_ctx
        elif isinstance(alt, dict):
            ctx2 = alt

        self.set_stage4_context(ctx2, **kwargs)
        # Plan A: fill missing join keys from Stage-3 ctx index (no ID invention).
        self._ensure_stage34_join_ctx_ready(where="stage4_set_episode_context")

        # Episode boundary detection (NO ID generation; just snapshot env-injected keys)
        uid = str(self._stage4_ctx.get("stage3_episode_uid", "") or self._stage4_ctx.get("episode_uid", "") or "")
        idx = str(self._stage4_ctx.get("stage3_episode_idx", "") or self._stage4_ctx.get("episode_idx", "") or "")

        if (uid and uid != prev_uid) or (idx and idx != prev_idx):
            self._on_episode_reset_autofill()

    def set_stage4_episode_context(self, episode_ctx: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Alias for env compatibility."""
        return self.stage4_set_episode_context(episode_ctx, **kwargs)
 

    def clear_stage4_context(self) -> None:
        """Clear stage4 context."""
        try:
            self._stage4_ctx.clear()
        except Exception:
            self._stage4_ctx = {}

    def _stage34_autofill_and_inject_join_keys(
        self,
        row: Dict[str, Any],
        *,
        agent_id: AgentId,
        sid: int,
        msg_id: str,
        t_env: float,
        ts_rep: Optional[float] = None,
        ts_arr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Autofill Stage-3/4 row with:
          - stage4 context (self._stage4_ctx)
          - normalized join keys (join_*)
          - canonical identifiers (agent_id/sid/msg_id)
        Also enforces: do NOT write legacy field 'agent' (use 'agent_id' only).
        """
        out: Dict[str, Any] = {}
        # Plan A: ensure ctx is best-effort filled from Stage-3 ctx index before copying.
        try:
            self._ensure_stage34_join_ctx_ready(where="stage34/autofill_and_inject")
        except Exception:
            pass

        # 1) user-provided context first
        try:
            if isinstance(self._stage4_ctx, dict):
                out.update(self._stage4_ctx)
        except Exception:
            pass

        # 2) deterministic base identifiers
        out["ctx_version"] = int(getattr(self, "_stage4_ctx_version", 1))
        out["pid"] = int(os.getpid())

        out["agent_id"] = str(agent_id) if (agent_id is not None and str(agent_id).strip() != "") else "__episode__"

        out["sid"] = int(sid)
        out["msg_id"] = str(msg_id)

        out["t_env"] = float(t_env)

        if ts_rep is not None:
            out["ts_rep"] = float(ts_rep)
        if ts_arr is not None:
            out["ts_arr"] = float(ts_arr)

        # 3) join keys (explicit & stable across files)
        out["join_agent_id"] = out["agent_id"]
        out["join_sid"] = out["sid"]
        out["join_msg_id"] = out["msg_id"]
        out["join_t_env"] = out["t_env"]
        if ts_rep is not None:
            out["join_ts_rep"] = float(ts_rep)
        if ts_arr is not None:
            out["join_ts_arr"] = float(ts_arr)

        # Optional: join key string (handy for grep/join)
        try:
            # P1: v2 join_key includes ts_rep/ts_arr for robustness under delay/reorder
            ts_rep_s = "NA"
            ts_arr_s = "NA"
            if ts_rep is not None:
                ts_rep_s = f"{float(ts_rep):.3f}"
            if ts_arr is not None:
                ts_arr_s = f"{float(ts_arr):.3f}"
            out["join_key"] = (
                f"{out['agent_id']}|{int(out['sid'])}|{out['msg_id']}|{float(out['t_env']):.3f}"
                f"|{ts_rep_s}|{ts_arr_s}"
            )
        except Exception:
            pass

        # 4) merge actual payload (payload can override context if needed)
        try:
            if isinstance(row, dict):
                out.update(row)
        except Exception:
            pass

        # 5) enforce schema: remove legacy 'agent'
        if "agent" in out:
            try:
                del out["agent"]
            except Exception:
                pass

        # 6) enforce final naming (in case payload used legacy)
        out["agent_id"] = str(agent_id)
        out["sid"] = int(sid)
        out["msg_id"] = str(msg_id)
        out["t_env"] = float(t_env)

        # Re-assert Stage-3/4 contract fields (cannot be removed by payload)

        try:
            out["run_uuid"] = str(self._stage4_ctx.get("run_uuid", "") or self._stage4_ctx.get("stage3_run_uuid", "") or self._stage4_ctx.get("stage3_run_id", ""))
            out["run_id"] = str(self._stage4_ctx.get("run_id", ""))

            out["worker_index"] = self._stage4_ctx.get("worker_index", "")
            out["vector_index"] = self._stage4_ctx.get("vector_index", "")

            out["stage3_episode_uid"] = str(self._stage4_ctx.get("stage3_episode_uid", "") or self._stage4_ctx.get("episode_uid", ""))
            out["stage3_episode_idx"] = self._stage4_ctx.get("stage3_episode_idx", "")

            # NEW: episode-level join key string (PRIMARY KEY helper)
            try:
                ru = str(out.get("run_uuid", "")).strip()
                wi = str(out.get("worker_index", "")).strip()
                vi = str(out.get("vector_index", "")).strip()
                eu = str(out.get("stage3_episode_uid", "")).strip()
                ei = str(out.get("stage3_episode_idx", "")).strip()
                out["episode_join_key"] = f"{ru}|w{wi}|v{vi}|{eu}|{ei}"
            except Exception:
                out["episode_join_key"] = ""
 
            # legacy aliases for other internal consumers (not required columns)
            if out.get("vector_index", "") != "":
                out["env_index"] = out["vector_index"]
            if out.get("stage3_episode_idx", "") != "":
                out["episode_id"] = out["stage3_episode_idx"]
                out["episode_idx"] = out["stage3_episode_idx"]
            if out.get("stage3_episode_uid", "") != "":
                out["episode_uid"] = out["stage3_episode_uid"]
            # legacy alias: stage3_run_id semantic == run_uuid
            out["stage3_run_id"] = out.get("run_uuid", "")
 
        except Exception:
            pass
        # Hard assert: required join contract fields must be present & non-empty.
        _assert_stage34_join_contract(out, where="stage34/autofill_and_inject")

        return out

    def set_mmsi_map(self, mmsi_of_ship: Dict[ShipId, int]):
        self.mmsi_to_sid = {int(m): int(sid) for sid, m in mmsi_of_ship.items()}

    def reset(
        self,
        ship_ids,
        t0,
        init_states: Optional[Dict[int, TrueState]] = None,
        agent_ids: Optional[List[AgentId]] = None,
    ):
        """Reset per-agent PF states (optional warm-start from init_states)."""
        # Stage4: mark new episode boundary (MUST NOT generate IDs)
        self._on_episode_reset_autofill()

        self.agent_states.clear()
        self.t0 = float(t0)
        self.last_t = float(t0)

        if agent_ids is not None:
            for a in agent_ids:
                self._ensure_agent_state(a)
        # NOTE:
        #   Do NOT assert Stage-3/Stage-4 join keys here. In RLlib/Env reset flow,
        #   the environment may inject episode context *after* tm.reset().
        #   Join contract is enforced at write-time by _assert_stage34_join_contract.


        # Warm-start uses internal yaw convention (atan2(vy, vx)).
        if isinstance(init_states, dict) and len(init_states) > 0:
            eps = 1e-12
            target_agents = list(self.agent_states.keys())

            for a in target_agents:
                st = self.agent_states[a]
                st.tracks.clear()
                st.preproc.reset()

                # --- NEW: resolve ego sid for this agent ---
                ego_sid = self.agent_to_ego_sid.get(a, None)
                if ego_sid is None and isinstance(a, str) and a.startswith("ship_"):
                    try:
                        ego_sid = int(a.split("_")[-1])
                    except Exception:
                        ego_sid = None

                for sid in ship_ids:
                    sid = int(sid)
                    ts = init_states.get(sid, None)

                    # --- NEW: PF manager should never create ego track ---
                    if ego_sid is not None and sid == int(ego_sid):
                        continue

                    ts = init_states.get(sid, None)

                    if ts is None:
                        continue

                    x = float(ts.x)
                    y = float(ts.y)
                    vx = float(getattr(ts, "vx", 0.0))
                    vy = float(getattr(ts, "vy", 0.0))

                    sog_mps = float(math.hypot(vx, vy))
                    yaw_sim_rad = float(math.atan2(vy, vx)) if sog_mps > eps else 0.0
                    yaw_sim_rad = _wrap_pi(yaw_sim_rad)

                    pf = self._new_pf_for_ship(sid, a)

                    cog_true = (math.pi/2.0) - yaw_sim_rad
                    cog_true = (cog_true % (2.0*math.pi) + 2.0*math.pi) % (2.0*math.pi)  # wrap to [0,2pi) DEBUG-only
                    # Legacy alias: PF may read key "cog". In THIS project, "cog" == yaw_sim_rad (ENU, rad).
                    meas_use = {"x": x, "y": y, "sog": sog_mps, "yaw": yaw_sim_rad, "cog": yaw_sim_rad}
                    pf.init_from_meas(self.t0, meas_use)

                    st.tracks[sid] = PFTrack(
                        pf=pf,
                        last_meas_ts=self.t0,
                        last_reported_ts=self.t0,
                        last_update_ts=self.t0,
                        last_arrival_ts=self.t0,
                        valid=True,
                        last_meas_x=x,
                        last_meas_y=y,
                    )

    # -------------------------------------------------------
    # Per-agent query API
    # -------------------------------------------------------

    def all_estimates(
        self,
        agent_id: AgentId,
        t: Ts,
        true_states: Optional[Dict[ShipId, TrueState]] = None,
    ) -> Dict[ShipId, TrueState]:
        """
        Return agent's local PF estimates at env time t_env.
        Output is always PF-based (never uses true_states except debug printing).
        """
        ego_sid = self._ego_sid_of_agent(agent_id)

        t_env = float(t)
        st = self.agent_states.get(agent_id, None)
        if st is None:
            return {}

        out: Dict[ShipId, TrueState] = {}

        # normalize constants for u_* (kept in [0,1])
        try:
            age_stale = float(getattr(self.pf_noise.age, "age_stale", 8.0))
        except Exception:
            age_stale = 8.0
        max_age = float(self.max_age)

        lifecycle_dbg = (os.environ.get("PF_LIFECYCLE_DBG", "0") == "1")
        if lifecycle_dbg:
            st1 = inspect.stack()[1]
            # direct caller
            caller1 = f"{os.path.basename(st1.filename)}:{st1.lineno} in {st1.function}"
            # optional deeper caller chain
            depth = int(os.environ.get("PF_LIFECYCLE_CALLER_DEPTH", "1"))
            chain = [caller1]
            if depth > 1:
                stk = inspect.stack()
                for k in range(2, min(depth + 1, len(stk))):
                    s = stk[k]
                    chain.append(f"{os.path.basename(s.filename)}:{s.lineno} in {s.function}")

            print(
                f"[PF-LIFECYCLE-EST] t={t_env:.2f} agent_id={agent_id} "
                f"track_sids={list(st.tracks.keys())} caller={' <- '.join(chain)}"
            )

        for sid, tr in list(st.tracks.items()):
            sid = int(sid)

            if ego_sid is not None and sid == ego_sid:
                if lifecycle_dbg:
                    print(
                        f"[PF-SELF-TRACK] t={t_env:.2f} agent={agent_id} sid={sid} "
                        f"(ego) present in tracks -> skip"
                    )
                if os.environ.get("PF_CLEAN_SELF", "0") == "1":
                    st.tracks.pop(sid, None)
                continue

            if tr is None or tr.pf is None:
                continue

            # staleness based on TIME CONTRACT
            last_rep = getattr(tr, "last_reported_ts", None)
            last_upd = getattr(tr, "last_update_ts", None)
            if last_rep is None and last_upd is None:
                tr.valid = False
                continue

            last_rep_f = float(last_rep) if last_rep is not None else None
            last_upd_f = float(last_upd) if last_upd is not None else None

            info_age = float("inf") if last_rep_f is None else max(0.0, t_env - last_rep_f)
            silence = float("inf") if last_upd_f is None else max(0.0, t_env - last_upd_f)

            tr.valid = (silence <= float(self.max_age))

            pf = tr.pf
            pf_last = getattr(pf, "last_ts", None)
            if pf_last is not None and abs(float(pf_last) - float(t_env)) <= 1e-6:
                x_pf = np.array(getattr(pf, "x", None), dtype=float)
            else:
                x_pf, _ = self._predict_x_readonly(pf, t_env)

            if x_pf is None or len(x_pf) < 4:
                continue

            px, py, v, yaw = float(x_pf[0]), float(x_pf[1]), float(x_pf[2]), float(x_pf[3])
            vx = v * math.cos(yaw)
            vy = v * math.sin(yaw)

            ts_out = TrueState(ship_id=sid, t=t_env, x=px, y=py, vx=vx, vy=vy, yaw_east_ccw_rad=yaw)
            # Attach per-agent AIS uncertainty to the estimate object (best-effort).
            # These fields are READ by obs builder if you pass this object downstream.
            u_stale = 1.0
            u_silence = 1.0
            if math.isfinite(info_age):
                u_stale = float(np.clip(info_age / max(age_stale, 1e-6), 0.0, 1.0))
            if math.isfinite(silence):
                u_silence = float(np.clip(silence / max(max_age, 1e-6), 0.0, 1.0))
            self._try_setattr(ts_out, "ais_u_stale", u_stale)
            self._try_setattr(ts_out, "ais_u_silence", u_silence)
            self._try_setattr(ts_out, "ais_valid", bool(tr.valid))
            self._try_setattr(ts_out, "ais_info_age", float(info_age))
            self._try_setattr(ts_out, "ais_silence", float(silence))

            out[sid] = ts_out

        return out


    def get_latest(self, agent_id: AgentId, sid: int):
        st = self.agent_states.get(agent_id, None)
        if st is None:
            return None

        tr = st.tracks.get(int(sid), None)
        if tr is None:
            return None

        x_pred, _ = self._predict_x_readonly(tr.pf, self.last_t)
        return x_pred.copy()


    def get_estimate(self, agent_id: AgentId, sid: int, ts: float):
        st = self.agent_states.get(agent_id, None)
        if st is None:
            return None

        sid = int(sid)
        tr = st.tracks.get(sid, None)
        if tr is None:
            return None

        pf = tr.pf
        if pf.last_ts is None:
            return None

        x_pred, _ = self._predict_x_readonly(pf, float(ts))
        return x_pred.copy()


    # ---- optional stats ----

    def get_debug_stats(self, agent_id: AgentId) -> Dict[str, float]:
        st = self.agent_states.get(agent_id, None)
        if st is None:
            return {"ais_msgs": 0, "pf_updates": 0, "mean_pos_err": 0.0, "max_pos_err": 0.0, "samples": 0}

        mean_err = (st.dbg_err_pos_sum / st.dbg_err_pos_cnt) if st.dbg_err_pos_cnt > 0 else 0.0
        return {
            "ais_msgs": int(st.debug_msg_count),
            "pf_updates": int(st.debug_update_count),
            "mean_pos_err": float(mean_err),
            "max_pos_err": float(st.dbg_err_pos_max),
            "samples": int(st.dbg_err_pos_cnt),
        }


    def reset_debug_stats(self, agent_id: Optional[AgentId] = None):
        if agent_id is None:
            for a in list(self.agent_states.keys()):
                self.reset_debug_stats(a)
            return

        st = self.agent_states.get(agent_id, None)
        if st is None:
            return

        st.debug_msg_count = 0
        st.debug_update_count = 0
        st.dbg_err_pos_sum = 0.0
        st.dbg_err_pos_max = 0.0
        st.dbg_err_pos_cnt = 0

    def get_pf_estimates_for_staging(
        self,
        t: Ts,
        true_states: Dict[ShipId, TrueState],
    ) -> List[Dict[str, Any]]:
        """
        Get PF estimates for all ships with error metrics for staging recording.

        Returns a list of dicts, one per ship, containing:
          - ship_id
          - Estimate: est_x, est_y, est_vx, est_vy, est_psi, est_sog
          - Truth: true_x, true_y, true_vx, true_vy, true_psi, true_sog
          - Errors: pos_error, vel_error, heading_error
          - Track quality: track_age, num_particles, eff_particles
        """
        import math
        results = []
        t_env = float(t)

        # Get all ship IDs from true_states
        for sid, true_st in true_states.items():
            sid = int(sid)

            # Find PF estimate for this ship (from any agent that has it)
            est_found = False
            est_x, est_y, est_vx, est_vy, est_psi = 0.0, 0.0, 0.0, 0.0, 0.0
            track_age = float("inf")
            num_particles = 0
            eff_particles = 0.0

            for agent_id, agent_st in self.agent_states.items():
                if agent_st is None:
                    continue
                tr = agent_st.tracks.get(sid, None)
                if tr is None:
                    continue

                pf = tr.pf
                if pf is None or pf.last_ts is None:
                    continue

                # Get estimate at current time
                try:
                    x_pred, _ = self._predict_x_readonly(pf, t_env)
                    if x_pred is not None:
                        est_x = float(x_pred[0])
                        est_y = float(x_pred[1])
                        est_vx = float(x_pred[2])
                        est_vy = float(x_pred[3])
                        est_psi = float(math.atan2(est_vy, est_vx))
                        track_age = t_env - float(pf.last_ts)
                        num_particles = int(getattr(pf, "N", 0))
                        # Effective sample size estimation
                        w = getattr(pf, "w", None)
                        if w is not None and len(w) > 0:
                            w_sum = float(sum(w))
                            if w_sum > 0:
                                w_norm = [wi / w_sum for wi in w]
                                eff_particles = 1.0 / sum(wi * wi for wi in w_norm) if sum(wi * wi for wi in w_norm) > 0 else 0.0
                        est_found = True
                        break
                except Exception:
                    continue

            # Compute truth values
            true_x = float(true_st.x)
            true_y = float(true_st.y)
            true_vx = float(true_st.vx)
            true_vy = float(true_st.vy)
            true_psi = float(true_st.yaw_east_ccw_rad)
            true_sog = float(true_st.sog)

            # Compute errors
            if est_found:
                est_sog = math.hypot(est_vx, est_vy)
                pos_error = math.hypot(est_x - true_x, est_y - true_y)
                vel_error = abs(est_sog - true_sog)
                heading_error = est_psi - true_psi
                # Wrap to [-pi, pi]
                while heading_error > math.pi:
                    heading_error -= 2 * math.pi
                while heading_error < -math.pi:
                    heading_error += 2 * math.pi
            else:
                # No estimate found - use truth as fallback with zero error
                est_x, est_y = true_x, true_y
                est_vx, est_vy = true_vx, true_vy
                est_psi = true_psi
                est_sog = true_sog
                pos_error = 0.0
                vel_error = 0.0
                heading_error = 0.0
                track_age = 0.0

            results.append({
                "ship_id": sid,
                "est_found": est_found,
                "est_x": est_x,
                "est_y": est_y,
                "est_vx": est_vx,
                "est_vy": est_vy,
                "est_psi": est_psi,
                "est_sog": math.hypot(est_vx, est_vy) if est_found else true_sog,
                "true_x": true_x,
                "true_y": true_y,
                "true_vx": true_vx,
                "true_vy": true_vy,
                "true_psi": true_psi,
                "true_sog": true_sog,
                "pos_error": pos_error,
                "vel_error": vel_error,
                "heading_error": heading_error,
                "track_age": track_age if track_age != float("inf") else -1.0,
                "num_particles": num_particles,
                "eff_particles": eff_particles,
            })

        return results

    # -------------------------------------------------------
    def ingest(
        self,
        t: Ts,
        ready: Dict[AgentId, List[RxMsg]],
        true_states: Optional[Dict[int, TrueState]] = None,
    ):
        """
        Per-agent ingest:
        - inbox dedup (per-agent)
        - convert AIS COG -> internal yaw (single point)
        - preproc.ingest(SimpleMeas)
        - PF update/create strictly @ t_env (single PF time axis)
        """
        self.last_t = float(t)
        t_env = float(self.last_t)

        # Helper constants
        warn_self_rx = (os.environ.get("PF_SELF_RX_WARN", "0") == "1")

        lifecycle_dbg = (os.environ.get("PF_LIFECYCLE_DBG", "0") == "1")

        # -------------------------------------------------
        # Normalize truth dict ONCE per ingest call (debug-only)
        # -------------------------------------------------
        truth_by_sid = self._normalize_truth_dict(true_states)
        self._truth_by_sid_global = truth_by_sid

        truth_probe = (os.environ.get("PF_TRUTH_PROBE", "0") == "1")
        if truth_probe and truth_by_sid is None:
            # 只要 upstream 没传 true_states，这里会明确告诉你
            print(f"[PF-TRUTH-PROBE-NONE] t={t_env:.2f} true_states is None/empty (upstream did not pass truth)")


        if os.environ.get("PF_RX_READY", "0") == "1":
            sizes = {str(a): len(ms) for a, ms in ready.items()}
            print(f"[PF-READY] t={t_env:.2f} per_agent_counts={sizes} total={sum(sizes.values())}")

        for rx_agent, msgs in ready.items():
            st = self._ensure_agent_state(rx_agent)
            st.debug_msg_count += len(msgs)


            # Resolve ego sid once (used for hard self-drop)
            ego_sid_rx = self._ego_sid_of_agent(rx_agent)

            # -------------------------------------------------
            # cache truth for debug printing ONLY (never for PF)
            # IMPORTANT: use normalized truth_by_sid
            # -------------------------------------------------
            st.true_by_sid = truth_by_sid

            # ego_yaw_sim_rad (debug-only):
            # IMPORTANT: reuse the SAME truth extraction logic as _truth_print (i.e., _truth_unpack),
            # so ego_yaw works even if TrueState provides yaw/psi/heading but not vx/vy.

            prev_yaw = float(getattr(st, "last_ego_yaw_sim_rad", float("nan")))
            ego_yaw_sim_rad = prev_yaw if math.isfinite(prev_yaw) else 0.0  # default fallback

            ts_ego = None
            if isinstance(truth_by_sid, dict) and ego_sid_rx is not None:
                ts_ego = truth_by_sid.get(int(ego_sid_rx), None)

            if ts_ego is not None:
                # _truth_unpack prefers yaw_sim_rad/yaw/psi/heading; else derives from vx/vy; else NaN.
                _x, _y, yaw, _sog, _vx, _vy = self._truth_unpack(ts_ego)
                if math.isfinite(yaw):
                    ego_yaw_sim_rad = float(yaw)
                # else keep fallback (prev/0)

            st.ego_yaw_sim_rad = float(ego_yaw_sim_rad)
            st.last_ego_yaw_sim_rad = float(ego_yaw_sim_rad)

            # optional single-line debug for ego yaw (OFF by default)
            if os.environ.get("PF_EGO_YAW_DBG", "0") == "1":
                src = "truth" if ts_ego is not None and math.isfinite(st.ego_yaw_sim_rad) else "fallback"
                print(f"[PF-EGO-YAW] t={t_env:.2f} agent={rx_agent} ego_sid={ego_sid_rx} ego_yaw={st.ego_yaw_sim_rad:+.3f} src={src}")

            # inbox dedup
            seen: set[Tuple[int, str]] = set()
            sid_raw_cnt: Dict[int, int] = {}
            sid_preproc_cnt: Dict[int, int] = {}   # preproc outputs (before cleaned dedup)
            sid_clean_cnt: Dict[int, int] = {}     # final cleaned2 count (after dedup)


            if lifecycle_dbg:
                print(f"[PF-LIFECYCLE-RX] t={t_env:.2f} rx_agent={rx_agent} raw_msgs={len(msgs)}")

            for r in msgs:
                # truth probe switch (default OFF)
                truth_probe = (os.environ.get("PF_TRUTH_PROBE", "0") == "1")
                if truth_probe and not hasattr(self, "_truth_probe_once"):
                    self._truth_probe_once = set()

                sid = self.mmsi_to_sid.get(int(r.mmsi), None)
                if sid is None:
                    continue
                sid = int(sid)  # normalize sid type to python int

                # ---------------------------------------------------------
                # HARD RULE: ego does NOT ingest/fuse its own AIS packets.
                # Ego state comes from truth trajectory only.
                # Drop BEFORE preproc to avoid any side effect/pollution.
                # ---------------------------------------------------------
                if ego_sid_rx is not None and sid == int(ego_sid_rx):
                    if warn_self_rx:
                        print(
                            f"[PF-SELF-RX][DROP] t={t_env:.2f} rx_agent={rx_agent} "
                            f"ego_sid={int(ego_sid_rx)} got sid={int(sid)} "
                            f"msg_id={getattr(r,'msg_id',None)} tx_id={getattr(r,'tx_id',None)} "
                            f"rep={float(getattr(r,'reported_ts',-1.0)):.2f} arr={float(getattr(r,'arrival_time',-1.0)):.2f}"
                        )
                    continue


                # ---- TRUTH PROBE: validate normalized truth_by_sid key match ----
                if truth_probe and isinstance(truth_by_sid, dict):
                    ts_true = truth_by_sid.get(sid, None)
                    kind = "MISS" if ts_true is None else "HIT"
                    probe_key = (str(rx_agent), int(sid), kind)
                    if probe_key not in self._truth_probe_once:
                        self._truth_probe_once.add(probe_key)
                        keys = list(truth_by_sid.keys())
                        keys_preview = keys[:12]

                        if ts_true is None:
                            print(
                                f"[PF-TRUTH-PROBE-{kind}][{rx_agent}] t={t_env:.2f} "
                                f"mmsi={int(r.mmsi)} sid={sid} "
                                f"true_keys(sample)={keys_preview} len={len(keys)}"
                            )
                        else:
                            x = getattr(ts_true, "x", None)
                            y = getattr(ts_true, "y", None)
                            yaw = getattr(ts_true, "yaw", getattr(ts_true, "psi", None))
                            sog = getattr(ts_true, "sog", getattr(ts_true, "v", None))
                            vx = getattr(ts_true, "vx", None)
                            vy = getattr(ts_true, "vy", None)
                            print(
                                f"[PF-TRUTH-PROBE-{kind}][{rx_agent}] t={t_env:.2f} "
                                f"mmsi={int(r.mmsi)} sid={sid} TRUE(xy)=({x},{y}) yaw={yaw} sog={sog} vx={vx} vy={vy} "
                                f"| true_keys(sample)={keys_preview} len={len(keys)}"
                            )

                sid_raw_cnt[sid] = sid_raw_cnt.get(sid, 0) + 1

                # =========================================================
                # 1) Build canonical msg_id (string) and inbox dedup on (sid, mid)
                # =========================================================
                # 在构造 RxMsg 前（就在 msg = RxMsg(...) 之前）
                mid = getattr(r, "msg_id", None)
                mid = str(mid).strip() if mid is not None else ""
                #if DBG_MID:
                    #print(f"[MID-CANON] t={t_env:.2f} sid={sid} raw={getattr(r,'msg_id',None)} canon={mid}")
                if (mid == "") or (mid.lower() in ("none", "null")):
                    # 关键：必须“确定性”，不能用 uuid
                    # 同一条 AIS 包重复到达/乱序重放，以下字段应保持一致 -> 去重才会生效
                    mid = (
                        f"m{int(r.mmsi)}"
                        f"_t{float(getattr(r,'reported_ts',-1.0)):.6f}"
                        f"_x{float(getattr(r,'reported_x',0.0)):.2f}"
                        f"_y{float(getattr(r,'reported_y',0.0)):.2f}"
                        f"_v{float(getattr(r,'reported_sog',0.0)):.3f}"
                        f"_c{float(getattr(r,'reported_cog',0.0)):.4f}"
                    )
                # 归一化为 str
                mid = str(mid)

                key = (int(sid), mid)
                if key in seen:
                    if os.environ.get("PF_RX_DETAIL", "0") == "1":
                        pf_log(f"[PF-RxDup][{rx_agent}] t={t_env:.2f} sid={sid} msg_id={mid}", sid=int(sid))
                    continue
                seen.add(key)

                # =========================================================
                # 2) yaw semantics (already fixed in your version)
                # =========================================================
                received_raw_cog = float(getattr(r, "reported_cog", 0.0))
                yaw_sim_rad = _wrap_pi(received_raw_cog)

                # Debug-only derived nautical COG (never used by PF)
                cog_north_cw_rad = _yaw_sim_rad_to_cog_north_cw_rad_debug(yaw_sim_rad)

                # =========================================================
                # 3) Create SimpleMeas and ATTACH the SAME canonical mid
                # =========================================================
                meas = SimpleMeas(
                    ship_id=int(sid),
                    x=float(r.reported_x),
                    y=float(r.reported_y),
                    sog_mps=float(r.reported_sog),
                    cog_rad=float(yaw_sim_rad),  # legacy name; semantically yaw_sim_rad
                    arrival_time=float(r.arrival_time),
                    reported_ts=float(r.reported_ts),
                    msg_id=mid,
                )

                #if DBG_MID:
                #    print(f"[MID-MEAS] t={t_env:.2f} sid={sid} meas.msg_id={getattr(meas,'msg_id',None)}")

                setattr(meas, "cog_north_cw_rad", float(cog_north_cw_rad))
                setattr(meas, "yaw_sim_rad", float(yaw_sim_rad))
                # Debug-only: ego yaw is derived from truth; keep it out of preproc/PF unless explicitly enabled.
                setattr(meas, "dbg_ego_yaw_sim_rad", float(st.ego_yaw_sim_rad))
                if PF_ATTACH_EGO_YAW:
                    setattr(meas, "ego_yaw_sim_rad", float(st.ego_yaw_sim_rad))

                setattr(meas, "mmsi", int(getattr(r, "mmsi", -1)))



                # Forward-compat alias (some PF code uses meas.yaw)
                try:
                    setattr(meas, "yaw", float(yaw_sim_rad))
                except Exception:
                    pass

                # Debug print + assertion at the MEAS INJECTION point
                if PF_DEBUG_YAW and ((PF_DEBUG_YAW_SID is None) or (int(sid) == int(PF_DEBUG_YAW_SID))):
                    global _pf_yaw_dbg_n
                    meas_yaw = float(getattr(meas, "yaw", getattr(meas, "cog_rad", 0.0)))
                    err = abs(_wrap_pi(meas_yaw - received_raw_cog))

                    if _pf_yaw_dbg_n < PF_DEBUG_YAW_MAX:
                        print(
                            f"[PF-YAW-DBG][{rx_agent}] t={t_env:.2f} sid={int(sid)} "
                            f"raw_cog={received_raw_cog:+.6f} meas_yaw={meas_yaw:+.6f} "
                            f"err_wrap={err:.6e} tol={PF_DEBUG_YAW_TOL:.2e}"
                        )
                        _pf_yaw_dbg_n += 1

                    if err > float(PF_DEBUG_YAW_TOL):
                        raise AssertionError(
                            f"[PF-YAW-CONTRACT-FAIL][{rx_agent}] t={t_env:.2f} sid={int(sid)} "
                            f"raw_cog={received_raw_cog:+.6f} meas_yaw={meas_yaw:+.6f} "
                            f"err_wrap={err:.6e} tol={PF_DEBUG_YAW_TOL:.2e}"
                        )

                # [NOTE] msg_id is now passed in ctor above; keep setattr for backward safety if desired
                # setattr(meas, "msg_id", mid)


                cleaned = st.preproc.ingest(meas)
                if not cleaned:
                    sid_preproc_cnt[sid] = sid_preproc_cnt.get(sid, 0)
                    sid_clean_cnt[sid] = sid_clean_cnt.get(sid, 0)
                    continue

                sid_preproc_cnt[sid] = sid_preproc_cnt.get(sid, 0) + len(cleaned)

                # order by arrival_time for determinism only (PF axis is still t_env)
                # order by arrival_time for determinism only (PF axis is still t_env)
                cleaned = sorted(
                    cleaned,
                    key=lambda c: float(getattr(c, "arrival_time", getattr(c, "reported_ts", 0.0)))
                )

                # =========================================================
                # 1) dedup CLEANED by (sid, msg_id) BEFORE PF update
                # =========================================================
                seen_cleaned: set[tuple[int, str]] = set()
                cleaned2: list[SimpleMeas] = []

                for cm in cleaned:
                    # 如果 preproc 丢了 msg_id，给一个保守 fallback（避免 None 造成无法去重）
                    cmid = getattr(cm, "msg_id", None)
                    cmid = str(cmid).strip() if cmid is not None else ""


                    if (cmid == "") or (cmid.lower() in ("none", "null")):
                        # 仅依赖 SimpleMeas 核心字段：ship_id/reported_ts/x/y/sog/cog
                        cmid = (
                            f"s{int(cm.ship_id)}"
                            f"_t{float(cm.reported_ts):.6f}"
                            f"_x{float(cm.x):.2f}"
                            f"_y{float(cm.y):.2f}"
                            f"_v{float(cm.sog_mps):.3f}"
                            f"_c{float(getattr(cm,'cog_rad',0.0)):.4f}"
                        )

                    setattr(cm, "msg_id", cmid)

                    k = (int(cm.ship_id), cmid)
                    if k in seen_cleaned:
                        continue
                    seen_cleaned.add(k)

                    # 确保 cm 上真的有 msg_id（供 update() 使用）
                    setattr(cm, "msg_id", cmid)

                    cleaned2.append(cm)

                # =========================================================
                # 2) PF updates ONLY on cleaned2
                # =========================================================
                sid_clean_cnt[sid] = sid_clean_cnt.get(sid, 0) + len(cleaned2)

                for cm in cleaned2:
                    st.debug_update_count += 1

                    ts_rep = float(cm.reported_ts)
                    ts_arr = float(getattr(cm, "arrival_time", t_env))
                    ts_arr = min(ts_arr, t_env)
                    ts_arr = max(ts_arr, self.t0)

                    age_meas = max(0.0, t_env - ts_rep)  # ONLY age fed into PF (per TIME CONTRACT)

                    # Debug-only: do not leak truth-derived ego yaw into the main path by default.
                    setattr(cm, "dbg_ego_yaw_sim_rad", float(st.ego_yaw_sim_rad))
                    if PF_ATTACH_EGO_YAW:
                        setattr(cm, "ego_yaw_sim_rad", float(st.ego_yaw_sim_rad))

                    self._update_with_meas_for_agent(
                        rx_agent=rx_agent,
                        st=st,
                        m=cm,
                        t_env=t_env,
                        ts_rep=ts_rep,
                        ts_arr=ts_arr,
                        age_meas=age_meas,
                    )


            if lifecycle_dbg:
                active_sids = list(st.tracks.keys())
                print(
                    f"[PF-LIFECYCLE-TRACKS] t={t_env:.2f} rx_agent={rx_agent} "
                    f"tracks={active_sids} raw_cnt={sid_raw_cnt} preproc_cnt={sid_preproc_cnt} cleaned_cnt={sid_clean_cnt}"

                )

        # Align all PFs to env-step time (RL time axis)
        self._sync_all_tracks_to_now(t_env)

    # -------------------------------------------------------
    # Optional helper: export per-agent meta (wrapper-friendly)
    # -------------------------------------------------------
    def export_track_meta(self, agent_id: AgentId, t: Ts) -> Dict[int, Dict[str, float]]:
        """
        Return per-track meta for this agent at env time t_env.
        This is read-only and does not mutate PF.
        Keys: sid -> {info_age, silence, valid, u_stale, u_silence}
        """
        st = self.agent_states.get(agent_id, None)
        if st is None:
            return {}
        t_env = float(t)
        try:
            age_stale = float(getattr(self.pf_noise.age, "age_stale", 8.0))
        except Exception:
            age_stale = 8.0
        max_age = float(self.max_age)
        out: Dict[int, Dict[str, float]] = {}
        for sid, tr in list(st.tracks.items()):
            last_rep = getattr(tr, "last_reported_ts", None)
            last_upd = getattr(tr, "last_update_ts", None)
            info_age = float("inf") if last_rep is None else max(0.0, t_env - float(last_rep))
            silence  = float("inf") if last_upd is None else max(0.0, t_env - float(last_upd))
            valid = 1.0 if (bool(getattr(tr, "valid", False)) and (silence <= max_age)) else 0.0
            u_stale = float(np.clip(info_age / max(age_stale, 1e-6), 0.0, 1.0)) if math.isfinite(info_age) else 1.0
            u_silence = float(np.clip(silence / max(max_age, 1e-6), 0.0, 1.0)) if math.isfinite(silence) else 1.0
            out[int(sid)] = {
                "info_age": float(info_age),
                "silence": float(silence),
                "valid": float(valid),
                "u_stale": float(u_stale),
                "u_silence": float(u_silence),
            }
        return out


    # -------------------------------------------------------

    def _update_with_meas_for_agent(
        self,
        rx_agent: AgentId,
        st: AgentPFState,
        m: SimpleMeas,
        t_env: float,
        ts_rep: float,
        ts_arr: float,
        age_meas: float,
    ):

        sid = int(m.ship_id)
        ego = self.agent_to_ego_sid.get(rx_agent, None)
        if ego is not None and int(sid) == int(ego):
            return

        ts_now = float(t_env)

        dt_comm = max(0.0, float(ts_arr) - float(ts_rep))
        dt_hold = max(0.0, float(ts_now) - float(ts_arr))
        relock = "none"

        # age_meas 已经是 dt_proj = t_env - ts_rep
        # 1. 提取基础观测值
        mx_rep = float(m.x)
        my_rep = float(m.y)
        sog_mps = float(m.sog_mps)

        tr = st.tracks.get(sid, None)
        msg_id_val = getattr(m, "msg_id", None)
        msg_id_val = str(msg_id_val).strip() if msg_id_val is not None else ""

        if (msg_id_val == "") or (msg_id_val.lower() in ("none", "null")):
            msg_id_val = (
                f"s{int(m.ship_id)}"
                f"_t{float(ts_rep):.6f}"
                f"_x{float(m.x):.2f}"
                f"_y{float(m.y):.2f}"
                f"_v{float(m.sog_mps):.3f}"
                f"_c{float(getattr(m,'cog_rad',0.0)):.4f}"
            )

        if tr is not None:
            # robust dedup: block both consecutive and non-consecutive repeats
            if getattr(tr, "last_fused_msg_id", None) == msg_id_val:
                return
            recent = getattr(tr, "recent_fused_msg_ids", None)
            if recent is not None and msg_id_val in recent:
                return

        # 2. 提取并确定 Yaw：强制 raw_cog 直通 = PF yaw (ENU, +X=0, CCW+)
        yaw_sim_val = getattr(m, "cog_rad", None)
        if yaw_sim_val is None:
            # 仅在极少数情况下（m.cog_rad 不存在）才退化使用 yaw_sim 字段
            yaw_sim_val = getattr(m, "yaw_sim_rad", None)

        # 最后的兜底
        if yaw_sim_val is None:
            vx = getattr(m, "vx_sim_mps", None)
            vy = getattr(m, "vy_sim_mps", None)
            if vx is not None and vy is not None and (abs(float(vx)) + abs(float(vy)) > 1e-6):
                yaw_sim_val = math.atan2(float(vy), float(vx))
            else:
                yaw_sim_val = 0.0  # fallback only

        yaw_sim_rad = _wrap_pi(float(yaw_sim_val))

        # 3. 投影到当前时间 (用于 PF 初始猜测)
        dt_proj = float(max(0.0, age_meas))
        mx_use = mx_rep + sog_mps * dt_proj * math.cos(yaw_sim_rad)
        my_use = my_rep + sog_mps * dt_proj * math.sin(yaw_sim_rad)

        # 4. PF measurement dict
        meas_pf = {
            "x": float(mx_use),
            "y": float(my_use),
            "sog": float(sog_mps),
            "yaw": float(yaw_sim_rad),
            "cog": float(yaw_sim_rad),  # legacy alias
            "msg_id": msg_id_val,
            "age": float(age_meas),
        }

        # Optional assert (debug-only): meas yaw must equal raw (wrapped)
        if os.environ.get("PF_YAW_ASSERT", "0") == "1":
            raw = getattr(m, "cog_rad", None)
            if raw is not None:
                if abs(_wrap_pi(float(raw)) - float(meas_pf["yaw"])) > 1e-6:
                    print(
                        f"[PF-YAW-ASSERT-FAIL] sid={sid} raw={float(raw):+.6f} yaw={float(meas_pf['yaw']):+.6f}"
                    )

        # 辅助 Debug 字典（当前保留，但不参与 PF 逻辑）
        meas_rep = {
            "x": mx_rep,
            "y": my_rep,
            "yaw": yaw_sim_rad,
            "ego_yaw": float(getattr(m, "dbg_ego_yaw_sim_rad", getattr(m, "ego_yaw_sim_rad", float("nan")))),

        }

        # 5. 获取或初始化 Track
        tr = st.tracks.get(sid, None)

        # Debug 开关
        FUSE_DBG = (os.environ.get("PF_RX_FUSE_DBG", "0") == "1")
        try:
            FUSE_SID = int(os.environ.get("PF_RX_FUSE_SID", "-1"))
        except Exception:
            FUSE_SID = -1
        fuse_on = FUSE_DBG and (FUSE_SID < 0 or sid == FUSE_SID)
        truth_on = fuse_on and (os.environ.get("PF_RX_TRUTH_DBG", "1") == "1")
        upd_on = fuse_on and (os.environ.get("PF_RX_UPD_DBG", "1") == "1")

        def _truth_print(tag, pf_obj):
            if not truth_on:
                return

            ts_true = self._truth_lookup(st, sid)
            tx, ty, tyaw, tsog, tvx, tvy = self._truth_unpack(ts_true)

            tx = float(tx); ty = float(ty); tyaw = float(tyaw)
            tsog = float(tsog); tvx = float(tvx); tvy = float(tvy)

            # --- PF state ---
            px   = float(pf_obj.x[0]) if hasattr(pf_obj, "x") else float("nan")
            py   = float(pf_obj.x[1]) if hasattr(pf_obj, "x") else float("nan")
            pyaw = float(pf_obj.x[3]) if hasattr(pf_obj, "x") else float("nan")

            meas_x = float(meas_pf.get("x", float("nan")))
            meas_y = float(meas_pf.get("y", float("nan")))
            meas_yaw = float(meas_pf.get("yaw", float("nan")))

            # --- true course from velocity (diagnostic) ---
            true_course_vel = float("nan")
            true_speed_v = float("nan")
            if math.isfinite(tvx) and math.isfinite(tvy):
                true_speed_v = math.hypot(tvx, tvy)
                if true_speed_v > 1e-6:
                    true_course_vel = math.atan2(tvy, tvx)

            # --- true course from position delta (diagnostic) ---
            if not hasattr(st, "_truth_hist"):
                st._truth_hist = {}  # sid -> (t_env, x, y)

            prev = st._truth_hist.get(sid, None)
            true_course_pos = float("nan")
            dt_pos = float("nan")
            if prev is not None:
                t_prev, x_prev, y_prev = prev
                dt_pos = float(ts_now - float(t_prev))
                dxp = tx - float(x_prev)
                dyp = ty - float(y_prev)
                if dt_pos > 1e-6 and (abs(dxp) + abs(dyp)) > 1e-6:
                    true_course_pos = math.atan2(dyp, dxp)

            # update history (use current env time as key)
            st._truth_hist[sid] = (float(ts_now), float(tx), float(ty))

            # --- distances (CRITICAL) ---
            d_true_pf   = math.hypot(tx - px,     ty - py)
            d_true_meas = math.hypot(tx - meas_x, ty - meas_y)
            d_pf_meas   = math.hypot(px - meas_x, py - meas_y)  # 你已有

            # --- yaw errors ---
            err_meas_trueyaw = _wrap_pi(meas_yaw - tyaw)
            err_pf_trueyaw   = _wrap_pi(pyaw - tyaw)
            err_pf_meas      = _wrap_pi(pyaw - meas_yaw)

            err_meas_course_vel = float("nan")
            if math.isfinite(true_course_vel):
                err_meas_course_vel = _wrap_pi(meas_yaw - true_course_vel)

            err_meas_course_pos = float("nan")
            if math.isfinite(true_course_pos):
                err_meas_course_pos = _wrap_pi(meas_yaw - true_course_pos)

            # --- pretty dt_pos ---
            dt_pos_str = "nan"
            if math.isfinite(dt_pos):
                dt_pos_str = f"{dt_pos:.3f}s"

            # =========================================================
            # NEW: EGO_TRUE(xy) + two residuals to diagnose frame mismatch
            #   H1: PF is (local), MEAS is (global):  (pf + ego_true) ~ meas
            #   H2: PF is (global), MEAS is (local):  pf ~ (meas + ego_true)
            # =========================================================
            ego_true_x = float("nan")
            ego_true_y = float("nan")

            # 1) derive ego_sid for this rx_agent
            ego_sid = self.agent_to_ego_sid.get(rx_agent, None)
            if ego_sid is None and isinstance(rx_agent, str) and rx_agent.startswith("ship_"):
                try:
                    ego_sid = int(rx_agent.split("_")[-1])
                except Exception:
                    ego_sid = None

            # 2) lookup ego truth in the same normalized truth dict
            if ego_sid is not None:
                ts_ego = self._truth_lookup(st, int(ego_sid))
                ex, ey, _eyaw, _esog, _evx, _evy = self._truth_unpack(ts_ego)
                try:
                    ego_true_x = float(ex)
                    ego_true_y = float(ey)
                except Exception:
                    ego_true_x = float("nan")
                    ego_true_y = float("nan")

            # global residual (same as d_pf_meas)
            r_g = float(d_pf_meas)

            # H1/H2 residuals (translation-only hypothesis)
            r_h1 = float("nan")
            r_h2 = float("nan")
            if math.isfinite(ego_true_x) and math.isfinite(ego_true_y):
                pf_plus_ego_x = px + ego_true_x
                pf_plus_ego_y = py + ego_true_y
                meas_minus_ego_x = meas_x - ego_true_x
                meas_minus_ego_y = meas_y - ego_true_y
                # H1: PF(local)->global then compare to meas(global)
                r_h1 = math.hypot(pf_plus_ego_x - meas_x, pf_plus_ego_y - meas_y)
                # H2: MEAS(local)->global then compare to pf(global)
                r_h2 = math.hypot(px - (meas_minus_ego_x), py - (meas_minus_ego_y))

            else:
                pf_plus_ego_x = pf_plus_ego_y = float("nan")
                meas_minus_ego_x = meas_minus_ego_y = float("nan")
                r_h1 = float("nan")
                r_h2 = float("nan")

            # --- print (2 lines) ---
            # 在 PRED-NOW 时把参考系诊断信息显式塞进第一行；其他 tag 也打印，但你主要看 PRED-NOW 即可
            if str(tag) == "PRED-NOW":
                print(
                    f"[PF-TRUTH-{tag}][{rx_agent}] sid={sid} "
                    f"TRUE(xy)=({tx:.1f},{ty:.1f}) yaw_true={tyaw:+.3f} | "
                    f"MEAS(xy)=({meas_x:.1f},{meas_y:.1f}) yaw_meas={meas_yaw:+.3f} | "
                    f"PF(xy)=({px:.1f},{py:.1f}) yaw_pf={pyaw:+.3f} | "
                    f"EGO_TRUE(xy)=({ego_true_x:.1f},{ego_true_y:.1f}) | "
                    f"r_g(d_pf_meas)={r_g:.1f}m r_h1(pf+ego~meas)={r_h1:.1f}m r_h2(pf~meas-ego)={r_h2:.1f}m | "

                    f"pf+ego=({pf_plus_ego_x:.1f},{pf_plus_ego_y:.1f}) meas-ego=({meas_minus_ego_x:.1f},{meas_minus_ego_y:.1f}) | "
                    f"d_true_pf={d_true_pf:.1f}m d_true_meas={d_true_meas:.1f}m | "
                    f"err_pf_meas={err_pf_meas:+.3f} err_meas_trueyaw={err_meas_trueyaw:+.3f} err_pf_trueyaw={err_pf_trueyaw:+.3f}"
                )
            else:
                print(
                    f"[PF-TRUTH-{tag}][{rx_agent}] sid={sid} "
                    f"TRUE(xy)=({tx:.1f},{ty:.1f}) yaw_true={tyaw:+.3f} | "
                    f"MEAS(xy)=({meas_x:.1f},{meas_y:.1f}) yaw_meas={meas_yaw:+.3f} | "
                    f"PF(xy)=({px:.1f},{py:.1f}) yaw_pf={pyaw:+.3f} | "
                    f"d_true_pf={d_true_pf:.1f}m d_true_meas={d_true_meas:.1f}m d_pf_meas={d_pf_meas:.1f}m | "
                    f"err_pf_meas={err_pf_meas:+.3f} err_meas_trueyaw={err_meas_trueyaw:+.3f} err_pf_trueyaw={err_pf_trueyaw:+.3f}"
                )

            print(
                f"    TRUE_v=({tvx:+.2f},{tvy:+.2f}) sog_true={tsog:.2f} sog_from_v={true_speed_v:.2f} | "
                f"course_vel={true_course_vel:+.3f} (err_meas_course_vel={err_meas_course_vel:+.3f}) | "
                f"course_pos={true_course_pos:+.3f} dt_pos={dt_pos_str} (err_meas_course_pos={err_meas_course_pos:+.3f})"
            )

        def _upd_print(tag, pf_obj, ret):
            if not upd_on:
                return
            # 现场计算 Neff，而不是读不存在的属性
            try:
                w = getattr(pf_obj, "w", None)
                if w is not None:
                    w_arr = np.array(w)
                    w_sum = np.sum(w_arr)
                    if w_sum > 0:
                        w_arr /= w_sum
                    neff = 1.0 / (np.sum(w_arr ** 2) + 1e-12)
                else:
                    neff = -2.0  # no weights
            except Exception:
                neff = -3.0  # calculation failed

            print(f"[PF-UPD-{tag}] neff={neff:.1f} ret={ret}")

        def _write_meta(track, x, y):
            track.last_meas_ts = float(ts_rep)
            track.last_reported_ts = float(ts_rep)
            track.last_arrival_ts = float(ts_arr)
            track.last_update_ts = float(ts_now)
            track.last_meas_x = float(x)
            track.last_meas_y = float(y)

            if msg_id_val is not None:
                track.last_fused_msg_id = msg_id_val
                try:
                    track.recent_fused_msg_ids.append(msg_id_val)
                except Exception:
                    pass

            track.valid = True

        # --- A. 新 Track 初始化 ---
        if tr is None:
            pf = self._new_pf_for_ship(sid, rx_agent)
            pf.init_from_meas(ts_now, meas_pf)
            _truth_print("NEW-INIT", pf)
            tr = PFTrack(pf=pf)
            _write_meta(tr, meas_pf["x"], meas_pf["y"])
            st.tracks[sid] = tr
            return

        pf = tr.pf
        if pf.last_ts is None:
            pf.init_from_meas(ts_now, meas_pf)

        # --- B. 预测 (Predict) ---
        eps = 1e-6
        if pf.last_ts is not None and ts_now > float(pf.last_ts) + eps:
            pf.predict(ts_now - float(pf.last_ts))

        _truth_print("PRED-NOW", pf)

        # --- C. 软/硬重锁检测 (Relock Logic) ---
        px_pre = float(pf.x[0])
        py_pre = float(pf.x[1])
        dist_pre = float(math.hypot(meas_pf["x"] - px_pre, meas_pf["y"] - py_pre))

        last_upd = getattr(tr, "last_update_ts", None)
        dt_gap = max(0.0, ts_now - float(last_upd)) if last_upd is not None else 0.0

        v_cap = 3.0
        soft_thr = self.soft_relock_dist + 1.5 * v_cap * dt_gap
        hard_thr = self.hard_relock_dist + 3.0 * v_cap * dt_gap

        # =========================================================
        # Stage-4: instrumentation ONLY (no PF logic change)
        #   - Record extra parameters/diagnostics into pf_stats_write
        #   - Computed from existing variables + pf.cfg.noise(.age)
        # =========================================================
        # NOTE: These are "best-effort"; any failure falls back to ""/nan
        pf_cfg = getattr(pf, "cfg", None)
        pf_noise = getattr(pf_cfg, "noise", None) if pf_cfg is not None else None
        pf_age_cfg = getattr(pf_noise, "age", None) if pf_noise is not None else None

        # pre-state yaw/yawd (for logging only)
        try:
            s4_pyaw_pre = float(pf.x[3])
        except Exception:
            s4_pyaw_pre = float("nan")
        try:
            s4_yawd_pre = float(pf.x[4])
        except Exception:
            s4_yawd_pre = float("nan")

        # measurement yaw (already wrapped) and yaw innovation abs
        try:
            s4_meas_yaw = float(meas_pf.get("yaw", float("nan")))
        except Exception:
            s4_meas_yaw = float("nan")
        try:
            s4_yaw_innov_abs = abs(_wrap_pi(float(s4_meas_yaw) - float(s4_pyaw_pre)))
        except Exception:
            s4_yaw_innov_abs = float("nan")

        # age band (0=fresh,1=stale,2=very_stale,3=older)
        s4_age_band = ""
        try:
            af = float(getattr(pf_age_cfg, "age_fresh", 1.0))
            as_ = float(getattr(pf_age_cfg, "age_stale", 3.0))
            av = float(getattr(pf_age_cfg, "age_very_stale", 5.0))
            if float(age_meas) <= af:
                s4_age_band = 0
            elif float(age_meas) <= as_:
                s4_age_band = 1
            elif float(age_meas) <= av:
                s4_age_band = 2
            else:
                s4_age_band = 3
        except Exception:
            s4_age_band = ""

        # Q-gap theoretical extra diffusion (pos/sog/yaw/yawd), as configured
        s4_use_q_gap = ""
        s4_use_r_cont = ""
        s4_q_sigma_pos = ""
        s4_q_sigma_sog = ""
        s4_q_sigma_yaw = ""
        s4_q_sigma_yawd = ""
        try:
            s4_use_q_gap = int(bool(getattr(pf_age_cfg, "use_q_gap_inflation", True)))
        except Exception:
            s4_use_q_gap = ""
        try:
            s4_use_r_cont = int(bool(getattr(pf_age_cfg, "use_r_continuous_inflation", False)))
        except Exception:
            s4_use_r_cont = ""

        try:
            if pf_age_cfg is not None:
                v_ref = max(0.0, float(sog_mps))
                gap = float(max(0.0, age_meas))
                # pos
                pos_gain = float(getattr(pf_age_cfg, "pos_age_v_gain", 0.6))
                q_pos_cap = float(getattr(pf_age_cfg, "q_pos_std_max", 60.0))
                qpos = pos_gain * v_ref * gap
                s4_q_sigma_pos = float(min(qpos, q_pos_cap))
                # sog
                sog_gain = float(getattr(pf_age_cfg, "sog_age_a_gain", 0.25))
                q_sog_cap = float(getattr(pf_age_cfg, "q_sog_std_max", 2.0))
                qsog = sog_gain * gap
                s4_q_sigma_sog = float(min(qsog, q_sog_cap))
                # yaw (optional, default 0)
                q_yaw_rate = float(getattr(pf_age_cfg, "q_yaw_age_rate_std", 0.0))
                q_yaw_cap = float(getattr(pf_age_cfg, "q_yaw_std_max", math.radians(45.0)))
                qyaw = q_yaw_rate * gap
                s4_q_sigma_yaw = float(min(qyaw, q_yaw_cap))
                # yawd (optional, default 0)
                q_yawd_rate = float(getattr(pf_age_cfg, "q_yawd_age_rate_std", 0.0))
                q_yawd_cap = float(getattr(pf_age_cfg, "q_yawd_std_max", math.radians(10.0)))
                qyawd = q_yawd_rate * gap
                s4_q_sigma_yawd = float(min(qyawd, q_yawd_cap))
        except Exception:
            pass

        # yaw gate hit predicate (logging only; PF implements its own gating)
        s4_yaw_gate_hit = ""
        try:
            if pf_age_cfg is not None:
                yaw_gate_enable = bool(getattr(pf_age_cfg, "yaw_gate_enable", True))
                yaw_gate_age = float(getattr(pf_age_cfg, "yaw_gate_age", 2.5))
                yaw_gate_innov = math.radians(float(getattr(pf_age_cfg, "yaw_gate_innov_deg", 35.0)))
                s4_yaw_gate_hit = int(bool(yaw_gate_enable) and (float(age_meas) >= yaw_gate_age) and (float(s4_yaw_innov_abs) >= yaw_gate_innov))
        except Exception:
            s4_yaw_gate_hit = ""

        # yaw_soft/yawd_soft gate predicates (approx using point-estimate yawd; logging only)
        s4_yaw_soft_pass = ""
        s4_yawd_soft_pass = ""
        try:
            if pf_noise is not None:
                yaw_soft_en = bool(getattr(pf_noise, "yaw_soft_enable", True))
                yaw_soft_max_age = float(getattr(pf_noise, "yaw_soft_max_age", 2.5))
                yaw_soft_min_speed = float(getattr(pf_noise, "yaw_soft_min_speed", 0.3))
                yaw_soft_max_abs_yawd = float(getattr(pf_noise, "yaw_soft_max_abs_yawd", 0.45))
                s4_yaw_soft_pass = int(bool(yaw_soft_en)
                                        and (float(age_meas) <= yaw_soft_max_age)
                                        and (float(sog_mps) >= yaw_soft_min_speed)
                                        and (math.isfinite(s4_yawd_pre) and abs(float(s4_yawd_pre)) <= yaw_soft_max_abs_yawd))
                yawd_soft_en = bool(getattr(pf_noise, "yawd_soft_enable", True))
                yawd_soft_max_innov = math.radians(float(getattr(pf_noise, "yawd_soft_max_innov_deg", 25.0)))
                s4_yawd_soft_pass = int(bool(yawd_soft_en) and (math.isfinite(s4_yaw_innov_abs) and float(s4_yaw_innov_abs) <= yawd_soft_max_innov))
        except Exception:
            s4_yaw_soft_pass = ""
            s4_yawd_soft_pass = ""

        # C1. Hard Relock (偏差极大，直接重置)
        if dist_pre >= float(hard_thr):
            relock = "hard"
            pf.init_from_meas(ts_now, meas_pf)
            _truth_print("HARD-INIT", pf)
            _write_meta(tr, meas_pf["x"], meas_pf["y"])
            st.tracks[sid] = tr

            # stats (hard init has no update stats)

            # stage-4 post-state (logging only)
            try:
                s4_px_post = float(pf.x[0]); s4_py_post = float(pf.x[1])
                s4_pyaw_post = float(pf.x[3]); s4_yawd_post = float(pf.x[4])
            except Exception:
                s4_px_post = s4_py_post = s4_pyaw_post = s4_yawd_post = ""

            _row = {
                "t_env": ts_now, "agent_id": str(rx_agent), "sid": sid, "msg_id": msg_id_val,
                "ts_rep": ts_rep, "ts_arr": ts_arr, "dt_comm": dt_comm, "dt_hold": dt_hold, "age_meas": age_meas,
                "mx_rep": mx_rep, "my_rep": my_rep, "mx_use": meas_pf["x"], "my_use": meas_pf["y"],
                "px_pre": px_pre, "py_pre": py_pre, "dist_pre": dist_pre, "relock": relock,
                "sigma_pos": "", "sigma_sog": "", "sigma_yaw": "",
                "neff": "", "resampled": "", "collapsed": "",
                "ret": "hard_init",
                # ---- Stage-4 fields (pure logging) ----
                "s4_dt_gap": float(dt_gap), "s4_soft_thr": float(soft_thr), "s4_hard_thr": float(hard_thr),
                "s4_meas_yaw": float(s4_meas_yaw) if (s4_meas_yaw == s4_meas_yaw) else "",
                "s4_pyaw_pre": float(s4_pyaw_pre) if (s4_pyaw_pre == s4_pyaw_pre) else "",
                "s4_yawd_pre": float(s4_yawd_pre) if (s4_yawd_pre == s4_yawd_pre) else "",
                "s4_yaw_innov_abs": float(s4_yaw_innov_abs) if (s4_yaw_innov_abs == s4_yaw_innov_abs) else "",
                "s4_age_band": s4_age_band,
                "s4_use_q_gap": s4_use_q_gap, "s4_use_r_cont": s4_use_r_cont,
                "s4_q_sigma_pos": s4_q_sigma_pos, "s4_q_sigma_sog": s4_q_sigma_sog,
                "s4_q_sigma_yaw": s4_q_sigma_yaw, "s4_q_sigma_yawd": s4_q_sigma_yawd,
                "s4_yaw_gate_hit": s4_yaw_gate_hit,
                "s4_yaw_soft_pass": s4_yaw_soft_pass, "s4_yawd_soft_pass": s4_yawd_soft_pass,
                "s4_px_post": s4_px_post, "s4_py_post": s4_py_post,
                "s4_pyaw_post": s4_pyaw_post, "s4_yawd_post": s4_yawd_post,

            }
            _row = self._stage34_autofill_and_inject_join_keys(
                _row,
                agent_id=rx_agent,
                sid=sid,
                msg_id=msg_id_val,
                t_env=ts_now,
                ts_rep=ts_rep,
                ts_arr=ts_arr,
            )

            pf_stats_write(_row)      # Stage-3 (CSV)
            self._stage4_write_jsonl(_row, path=self._stage4_log_path())     # Stage-4 (JSONL, run-scoped)
            pf_stats_stage4_write(_row, path=self._stage4_stats_csv_path()) # Stage-4 (CSV, run-scoped)

            return

        # C2. Soft Relock (偏差较大，拉回粒子但保留部分分布)
        if dist_pre >= float(soft_thr):

            did = pf.soft_relock(
                ts_now, meas_pf, age_meas,
                pos_thr_m=float(soft_thr),
                beta_pos=float(self.soft_relock_beta),
                beta_vel=float(self.soft_relock_beta) * 0.8,
                beta_yaw=float(self.soft_relock_beta) * 0.8,
                reset_weights=True, # 重置权重为均匀
            )
            if did:
                relock = "soft"
                _truth_print("SOFT-AFTER", pf)
                # [CRITICAL FIX] 不要 Return! 
                # 让代码继续往下走，用当前的 measurement 更新刚才重置后的均匀权重。
                # 这样 PF 才能立刻收敛，而不是等到下一帧。
                pass 

        # --- D. 观测更新 (Update) ---
        # 使用鲁棒更新 (带 Age Gating)
        x_pre = np.array(pf.x, dtype=float).copy()   # <<< moved BEFORE update
        ret = pf.step_delay_robust(ts_now, meas_pf, age_meas)

        _upd_print("NORM", pf, ret)
        _truth_print("NORM-AFTER", pf)

        _write_meta(tr, meas_pf["x"], meas_pf["y"])
        st.tracks[sid] = tr

        s = getattr(pf, "last_update_stats", None) or {}

        # stage-4 post-state (logging only)
        try:
            s4_px_post = float(pf.x[0]); s4_py_post = float(pf.x[1])
            s4_pyaw_post = float(pf.x[3]); s4_yawd_post = float(pf.x[4])
        except Exception:
            s4_px_post = s4_py_post = s4_pyaw_post = s4_yawd_post = ""

        _row = {
            "t_env": ts_now, "agent_id": str(rx_agent), "sid": sid, "msg_id": msg_id_val,
            "ts_rep": ts_rep, "ts_arr": ts_arr, "dt_comm": dt_comm, "dt_hold": dt_hold, "age_meas": age_meas,
            "mx_rep": mx_rep, "my_rep": my_rep, "mx_use": meas_pf["x"], "my_use": meas_pf["y"],
            "px_pre": float(x_pre[0]), "py_pre": float(x_pre[1]),  # now truly pre-update
            "dist_pre": dist_pre, "relock": relock,
            "sigma_pos": s.get("sigma_pos",""), "sigma_sog": s.get("sigma_sog",""), "sigma_yaw": s.get("sigma_yaw",""),
            "neff": s.get("neff",""),
            "resampled": int(bool(s.get("resampled", False))) if s else "",
            "collapsed": int(bool(s.get("collapsed", False))) if s else "",
            "ret": ret,

            # ---- Stage-4 fields (pure logging) ----
            "s4_dt_gap": float(dt_gap), "s4_soft_thr": float(soft_thr), "s4_hard_thr": float(hard_thr),
            "s4_meas_yaw": float(s4_meas_yaw) if (s4_meas_yaw == s4_meas_yaw) else "",
            "s4_pyaw_pre": float(s4_pyaw_pre) if (s4_pyaw_pre == s4_pyaw_pre) else "",
            "s4_yawd_pre": float(s4_yawd_pre) if (s4_yawd_pre == s4_yawd_pre) else "",
            "s4_yaw_innov_abs": float(s4_yaw_innov_abs) if (s4_yaw_innov_abs == s4_yaw_innov_abs) else "",
            "s4_age_band": s4_age_band,
            "s4_use_q_gap": s4_use_q_gap, "s4_use_r_cont": s4_use_r_cont,
            "s4_q_sigma_pos": s4_q_sigma_pos, "s4_q_sigma_sog": s4_q_sigma_sog,
            "s4_q_sigma_yaw": s4_q_sigma_yaw, "s4_q_sigma_yawd": s4_q_sigma_yawd,
            "s4_yaw_gate_hit": s4_yaw_gate_hit,
            "s4_yaw_soft_pass": s4_yaw_soft_pass, "s4_yawd_soft_pass": s4_yawd_soft_pass,
            "s4_like_min": s.get("like_min",""), "s4_like_max": s.get("like_max",""),
            "s4_px_post": s4_px_post, "s4_py_post": s4_py_post,
            "s4_pyaw_post": s4_pyaw_post, "s4_yawd_post": s4_yawd_post,

        }
        # include full PF last_update_stats as a nested object (stage-4 file only)
        _row_stage4 = dict(_row)
        _row_stage4["pf_last_update_stats"] = s

        _row = self._stage34_autofill_and_inject_join_keys(
            _row,
            agent_id=rx_agent,
            sid=sid,
            msg_id=msg_id_val,
            t_env=ts_now,
            ts_rep=ts_rep,
            ts_arr=ts_arr,
        )
        _row_stage4 = self._stage34_autofill_and_inject_join_keys(
            _row_stage4,
            agent_id=rx_agent,
            sid=sid,
            msg_id=msg_id_val,
            t_env=ts_now,
            ts_rep=ts_rep,
            ts_arr=ts_arr,
        )

        # --- Stage-4 paths: best-effort ensure parent dir exists ---
        _s4_jsonl_path = self._stage4_log_path()
        _s4_csv_path = self._stage4_stats_csv_path()
        try:
            if _s4_jsonl_path:
                os.makedirs(os.path.dirname(_s4_jsonl_path), exist_ok=True)
            if _s4_csv_path:
                os.makedirs(os.path.dirname(_s4_csv_path), exist_ok=True)
        except Exception:
            pass

        pf_stats_write(_row)  # Stage-3 (CSV)
        # Stage-4 JSONL can include nested dict
        self._stage4_write_jsonl(_row_stage4, path=_s4_jsonl_path)
        # Stage-4 CSV MUST be flat (avoid dict-valued columns)
        pf_stats_stage4_write(_row, path=_s4_csv_path)

        return
