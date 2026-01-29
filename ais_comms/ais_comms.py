# ais_comms/ais_comms.py
from __future__ import annotations
import uuid
import os, math, json
import numpy as np
from uuid import uuid4
from typing import Dict, List, Optional, Tuple
import time
from .datatypes import TrueState, RawTxMsg, RxMsg, ShipId, AgentId, Ts
from .scheduler import TxScheduler
from .ge_channel import GEChannel
from .event_queue import ArrivalQueue
from .config import load_yaml, sample_episode_params, export_params_json
from types import SimpleNamespace

# Phase 1：欺诈层
from .fraud_wrapper import FraudAgentWrapper, FraudWrapperConfig

# === 全局调试总开关：训练时设 False 可关闭一步一条输出 ===
DEBUG_AIS = False
AIS_TXDBG   = os.environ.get("AIS_TXDBG", "0") == "1"
AIS_RXDBG   = os.environ.get("AIS_RXDBG", "0") == "1"
AIS_READYDBG = os.environ.get("AIS_READYDBG", "0") == "1"
AIS_DBG_TX = (os.environ.get("AIS_DBG_TX", "0") == "1")
AIS_DBG_LINK = (os.environ.get("AIS_DBG_LINK", "0") == "1")
AIS_DBG_RX = (os.environ.get("AIS_DBG_RX", "0") == "1")
AIS_ASSERT_SELFLOOP = (os.environ.get("AIS_ASSERT_SELFLOOP", "1") == "1")  # 默认开
AIS_ASSERT_SELFLOOP_MSG = (os.environ.get("AIS_ASSERT_SELFLOOP_MSG", "0") == "1")
AIS_DBG_FOCUS_TX = os.environ.get("AIS_DBG_FOCUS_TX", "")  # e.g., "1"
AIS_DBG_FOCUS_RX = os.environ.get("AIS_DBG_FOCUS_RX", "")  # e.g., "ship_1"
AISDBG = bool(int(os.environ.get("SHIPRL_AISDBG", "0")))
AISDBG_RX = os.environ.get("SHIPRL_AISDBG_RX", "")  # e.g. "ship_1" or empty for all

class AISCommsSim:
    """
    AIS 通信仿真主模块：
      Tx 调度 → GE 信道(Good/Bad+突发段) → Lognormal 网络时延 → 到达队列

    关键修复（本版本）：
      1) GE 按“时间步”推进：每个 env step 统一 tick(dt)，发报时仅 pass_now() 判定；
      2) 每条链路独立 GE 状态：按 (tx_ship, rx_agent) 维护 GEChannel；
      3) 与 ArrivalQueue 的 per-link 统计完全对齐（AIS_TRACE_PER_LINK）。

    同时记录：
      - 链路级计数：attempts(尝试)、passed(GE通过)、dropped(GE丢弃)、delivered(pop_ready成功)
      - PPR = passed/attempts（链路通过率），PDR = delivered/attempts（已达可用率）
      - delay/age 样本与时间序列（支持 CSV 导出）
    """

    def __init__(
        self,
        rng: Optional[np.random.Generator] = None,
        cfg_path: Optional[str] = None,
        base_seed: int = 0,
        fraud_enable: bool = False,
        fraud_hist_len: int = 8,
    ):
        self.base_seed = int(base_seed)
        self.rng = rng if rng is not None else np.random.default_rng(self.base_seed)

        # IMPORTANT: do not hardcode developer absolute paths; keep package-relative default.
        self.cfg_path = cfg_path or os.path.join(os.path.dirname(__file__), "ais_config_pf_debug.yaml")

        if DEBUG_AIS:
            print("[AIS] using cfg_path =", self.cfg_path)
        self.cfg = load_yaml(self.cfg_path)

        # --- stable metadata for stage3 (avoid empty fields) ---
        self.cfg_name = str(self.cfg.get("name", "")) if isinstance(self.cfg, dict) else ""
        self.N = 0
        self.dt = 0.0
        self.ep_seed = 0
        self._ships: List[int] = []
        self._agent_ids: List[str] = []

        # Delay/reorder RNG streams (episode-scoped seeds set in reset()).
        # rng_delay: lognormal delay sampling
        # rng_reorder: reorder Bernoulli (so reorder on/off does not perturb delay stream)
        self.rng_delay: Optional[np.random.Generator] = None
        self.rng_reorder: np.random.Generator = np.random.default_rng(0)


        # 运行期本集参数缓存
        self._ep_params = None

        # ship_id -> mmsi（内部测试段：999000000 + ship_id）
        self.mmsi_of_ship: Dict[ShipId, int] = {}

        # 组件
        self.scheduler = TxScheduler(rng=self.rng)
        self.arrivals = ArrivalQueue()

        # ship_id -> agent_id
        self.agent_of_ship: Dict[ShipId, AgentId] = {}
        # agent_id -> ship_id (reverse mapping for distance-based loss)
        self._ship_of_agent: Dict[AgentId, ShipId] = {}

        # —— Delay 模型（lognormal + clip）默认值，reset 时会被 _apply_episode_params 覆盖
        self.mu_ln = float(np.log(1.2))  # 对数均值（≈1.2s）
        self.sigma_ln = 0.35
        self.max_delay = 6.0

        # —— Clock 模型（默认关闭）
        self.clock_enable = False
        self.clock_offset_s = 0.0
        self.clock_drift_ppm = 0.0
        self.clock_of_ship: Dict[ShipId, Tuple[float, float]] = {}  # sid -> (offset_s, drift_ppm)

        # —— Obs config 缓存
        self.obs_slot_K = 16
        self.obs_slot_ttl = 45.0
        self.obs_age_cap = 4.0
        self.obs_miss_mask = True

        # —— 字段噪声 & 区域偏置
        self.noise_pos_m = 15.0
        self.noise_sog_mps = 1.0
        self.noise_cog_rad = math.radians(1.0)
        self.noise_rot_rads = 0.01  # rot noise std (rad/s), typical AIS rot resolution ~0.01 rad/s
        self.reg_bias_enable = True
        self.reg_bias_rects = []

        # —— Previous yaw tracking for rot computation (per ship)
        self._prev_yaw: Dict[ShipId, float] = {}
        self._prev_t: Dict[ShipId, float] = {}

        # -------- AR(1) 噪声参数（时间相关 SOG/COG） --------
        self.sog_ar1_rho = float(self.cfg.get("noise_sog_ar1_rho", 0.95))
        self.cog_ar1_rho = float(self.cfg.get("noise_cog_ar1_rho", 0.95))
        self.sog_ar1_rho = max(0.0, min(self.sog_ar1_rho, 0.999))
        self.cog_ar1_rho = max(0.0, min(self.cog_ar1_rho, 0.999))
        self._sog_ar1_state: Dict[ShipId, float] = {}
        self._cog_ar1_state: Dict[ShipId, float] = {}

        # —— 概率性字段错误
        self.fe_mmsi_conflict_prob = 0.0
        self.fe_type_missing_prob = 0.0
        self.fe_draft_missing_prob = 0.0
        self.fe_sog_zero_stick_prob = 0.0

        # 指标与时序
        self.reset_metrics()

        # 可选：链路级导出缓冲
        self._link_samples: List[dict] = []
        self._ar1_samples: List[dict] = []
        self.age_minus_delay_samples: List[float] = []

        # GPS bias + drift
        gps_cfg = self.cfg.get("gps_bias", {}) or {}
        self.gps_bias_enable = bool(gps_cfg.get("enable", False))
        self.gps_pos_bias_max = float(gps_cfg.get("pos_bias_max_m", 0.0))
        self.gps_drift_max_mps = float(gps_cfg.get("drift_max_mps", 0.0))
        self._gps_bias_xy: Dict[ShipId, tuple[float, float]] = {}
        self._gps_drift_xy: Dict[ShipId, tuple[float, float]] = {}
        self._gps_bias_t0: float = 0.0
        self._gps_dbg_rows: List[dict] = []

        # ========= 报文乱序仿真参数 =========
        # ========= Reorder (defaults, overwritten by episode params) =========
        self.reorder_enable = False
        self.reorder_prob = 0.0
        self.reorder_extra_delay = 0.0

        # 乱序统计：按 (tx_ship, rx_agent) 维护“上一次计划到达时间”
        self._last_arrival_by_link: Dict[Tuple[int, str], float] = {}
        self._reorder_total: int = 0
        self._reorder_hit: int = 0
        self._reorder_samples: List[dict] = []

        # ========= Phase 1 Patch: 欺诈层 =========
        fw_cfg = FraudWrapperConfig(enable=bool(fraud_enable), hist_len=int(fraud_hist_len))
        self.fraud_wrapper = FraudAgentWrapper(fw_cfg)

        # attacker 集合：None=未指定（旧逻辑），set()=本局无 attacker
        self.attacker_ships: Optional[set[ShipId]] = None
        fraud_cfg = self.cfg.get("fraud", {}) or {}
        self.fraud_debug: bool = bool(fraud_cfg.get("debug", False))

        # ========= 关键修复：每链路独立 GEChannel =========
        # (tx_ship, rx_agent) -> GEChannel
        self.channel_by_link: Dict[Tuple[int, str], GEChannel] = {}
        # 缓存 episode GE 参数，便于新链路初始化时复用
        self._ge_params = dict(
            p_g2b=0.01, p_b2g=0.2, drop_bad=True,
            burst_enable=False, burst_prob=0.0, burst_dur_s=0.0, burst_extra_drop=0.0,
            dist_enable=False, dist_ref_m=18520.0, dist_max_m=55560.0, dist_loss_exp=2.0,
        )

        # Distance-based loss configuration (cached for easy access)
        self.dist_enable = False
        self.dist_ref_m = 18520.0   # 10 nm in meters
        self.dist_max_m = 55560.0   # 30 nm in meters
        self.dist_loss_exp = 2.0

        # “时间推进”参考点
        self._last_tick_t: float = 0.0

        # ---- Stage3 contexts ----
        self._stage3_run_uuid = ""
        self._stage3_worker_index = -1
        self._stage3_vector_index = -1
        self._stage3_episode_uid = ""
        self._stage3_episode_idx = -1
        self._stage3_schema_version = "stage3_v1"
        self._stage3_comm_stats_path = None
        self._stage3_comm_stats_comm_path = None
        self._stage3_episodes_path = None
        # local fallback: ensure episode_idx monotonically increases per-process if env forgets to set it
        self._stage3_local_episode_counter: int = 0
        # effective params snapshot for stage3 (filled in reset)
        self._effective: dict | None = None

     # ---------------- Stage3 helpers ----------------
    def _stage3_autofill_context_if_missing(self) -> None:
        """
        Ensure Stage3 join keys exist even if the env forgets to call set_stage3_*_context().
        This avoids placeholder outputs (worker=-1/episode_idx=-1) and avoids hard failures.
        """
        # run_uuid
        if not str(getattr(self, "_stage3_run_uuid", "")).strip():
            ru = str(os.environ.get("SHIPRL_RUN_UUID", "")).strip()
            self._stage3_run_uuid = ru if ru else str(uuid4())

        # worker/env indices
        if int(getattr(self, "_stage3_worker_index", -1)) < 0:
            wi = None
            for k in ("RAY_WORKER_INDEX", "RLLIB_WORKER_INDEX", "WORKER_INDEX"):
                v = os.environ.get(k, "").strip()
                if v:
                    wi = v
                    break
            try:
                self._stage3_worker_index = int(wi) if wi is not None else 0
            except Exception:
                self._stage3_worker_index = 0

        if int(getattr(self, "_stage3_vector_index", -1)) < 0:
            vi = None
            for k in ("RAY_ENV_INDEX", "RLLIB_ENV_INDEX", "ENV_INDEX", "ENV_ID"):
                v = os.environ.get(k, "").strip()
                if v:
                    vi = v
                    break
            try:
                self._stage3_vector_index = int(vi) if vi is not None else 0
            except Exception:
                self._stage3_vector_index = 0
        # episode join keys
        if (not str(getattr(self, "_stage3_episode_uid", "")).strip()) or int(getattr(self, "_stage3_episode_idx", -1)) < 0:
            self._stage3_episode_uid = str(uuid4()).replace("-", "")
            self._stage3_episode_idx = int(self._stage3_local_episode_counter)
            self._stage3_local_episode_counter += 1

    def _stage3_build_effective_snapshot(self) -> dict:
        """
        Minimal effective snapshot used by stage3_flush_episode().
        Keep JSON-safe primitives only.
        """
        try:
            ep_seed_i = int(getattr(self, "_ep_seed", getattr(self, "ep_seed", 0)))
        except Exception:
            ep_seed_i = 0

        # clock per ship (effective)
        try:
            clock_of_ship = {
                str(int(sid)): {"offset_s": float(off), "drift_ppm": float(dppm)}
                for sid, (off, dppm) in getattr(self, "clock_of_ship", {}).items()
            }
        except Exception:
            clock_of_ship = {}

        # gps bias/drift per ship (effective)
        try:
            gps_bias_xy = {
                str(int(sid)): [float(xy[0]), float(xy[1])]
                for sid, xy in getattr(self, "_gps_bias_xy", {}).items()
            }
        except Exception:
            gps_bias_xy = {}
        try:
            gps_drift_xy = {
                str(int(sid)): [float(xy[0]), float(xy[1])]
                for sid, xy in getattr(self, "_gps_drift_xy", {}).items()
            }
        except Exception:
            gps_drift_xy = {}

        return dict(
            derived_seeds=dict(
                ep_seed=int(ep_seed_i),
                scheduler=(int(ep_seed_i + 2025) if ep_seed_i else 0),
                rng_delay=(int(ep_seed_i + 424242) if ep_seed_i else 0),
                rng_reorder=(int(ep_seed_i + 1337) if ep_seed_i else 0),
            ),
            delay=dict(
                mu_ln=float(getattr(self, "mu_ln", 0.0)),
                sigma_ln=float(getattr(self, "sigma_ln", 0.0)),
                clip_s=float(getattr(self, "max_delay", 0.0)),
            ),
            reorder=dict(
                enable=bool(getattr(self, "reorder_enable", False)),
                prob=float(getattr(self, "reorder_prob", 0.0)),
                extra_delay_s=float(getattr(self, "reorder_extra_delay", 0.0)),
            ),
            obs_cache=dict(
                slot_K=int(getattr(self, "obs_slot_K", 0)),
                slot_ttl_s=float(getattr(self, "obs_slot_ttl", 0.0)),
                age_cap_s=float(getattr(self, "obs_age_cap", 0.0)),
                miss_mask=bool(getattr(self, "obs_miss_mask", False)),
            ),
            noise=dict(
                pos_m=float(getattr(self, "noise_pos_m", 0.0)),
                sog_mps=float(getattr(self, "noise_sog_mps", 0.0)),
                cog_rad=float(getattr(self, "noise_cog_rad", 0.0)),
                rot_rads=float(getattr(self, "noise_rot_rads", 0.01)),
                reg_bias_enable=bool(getattr(self, "reg_bias_enable", False)),
            ),
            clock=dict(
                enable=bool(getattr(self, "clock_enable", False)),
                per_ship=clock_of_ship,
            ),
            gps=dict(
                enable=bool(getattr(self, "gps_bias_enable", False)),
                t0=float(getattr(self, "_gps_bias_t0", 0.0)),
                bias_xy=gps_bias_xy,
                drift_xy=gps_drift_xy,
            ),
            distance=dict(
                enable=bool(getattr(self, "dist_enable", False)),
                ref_m=float(getattr(self, "dist_ref_m", 18520.0)),
                max_m=float(getattr(self, "dist_max_m", 55560.0)),
                loss_exp=float(getattr(self, "dist_loss_exp", 2.0)),
            ),
        )

    def _stage3_paths_or_raise(self) -> tuple[str, str, str]:
        if not (self._stage3_comm_stats_path and self._stage3_comm_stats_comm_path and self._stage3_episodes_path):
            raise RuntimeError("Stage3 paths not set. Call set_stage3_run_context(...) first.")
        return (self._stage3_comm_stats_path, self._stage3_comm_stats_comm_path, self._stage3_episodes_path)

    @staticmethod
    def _stage3_file_lock(f):
        # advisory lock, works on Linux/WSL/Runpod; ensures single-writer semantics at write time
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

    @staticmethod
    def _stage3_file_unlock(f):
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _stage3_ensure_csv_header(self, path: str, schema_name: str, columns: list[str]) -> None:
        import os, time, hashlib, csv
        # NOTE:
        #   - Do NOT bake run_uuid into the header guard; re-running with the same output dir
        #     will otherwise ALWAYS trip "schema mismatch".
        #   - Guard schema by (schema_name, schema_version, cols_hash, header columns line).
        cols_line = ",".join(columns)
        cols_hash = hashlib.md5(cols_line.encode("utf-8")).hexdigest()[:12]
        header0 = f"# stage3_schema={schema_name} version={self._stage3_schema_version} cols_hash={cols_hash}\n"
        header1 = cols_line + "\n"

        # Default: do NOT crash training on mismatch; rotate the old file instead.
        # Set STAGE3_STRICT_SCHEMA=1 to restore hard-fail behavior.
        strict = (os.environ.get("STAGE3_STRICT_SCHEMA", "0") == "1")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a+", encoding="utf-8") as f:
            self._stage3_file_lock(f)
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size == 0:
                    f.write(header0)
                    f.write(header1)
                    f.flush()
                    os.fsync(f.fileno())
                    return
                f.seek(0)
                line0 = f.readline()
                line1 = f.readline()
                ok0 = (
                    line0.startswith(f"# stage3_schema={schema_name} ")
                    and (f"version={self._stage3_schema_version}" in line0)
                    and (f"cols_hash={cols_hash}" in line0)
                )
                ok1 = (line1 == header1)

                if ok0 and ok1:
                    # Extra guard: detect legacy-broken CSV (JSON blobs not quoted => column count mismatch).
                    # If the first data row does not match the expected column count, rotate the file.
                    probe = f.readline()
                    if probe:
                        try:
                            row = next(csv.reader([probe]))
                            if len(row) == len(columns):
                                return
                        except Exception:
                            pass
                    else:
                        # header-only file
                        return

                if strict:
                    raise RuntimeError(
                        f"Stage3 schema/header mismatch for {path}. "
                        "Refuse to append to prevent schema pollution. "
                        f"(set STAGE3_STRICT_SCHEMA=0 to auto-rotate)"
                    )
                # Auto-rotate old file to avoid killing training
                bak = f"{path}.bak_{int(time.time())}_pid{os.getpid()}"
                try:
                    os.replace(path, bak)
                except Exception:
                    # best-effort: if another worker already rotated, continue
                    pass
            finally:
                self._stage3_file_unlock(f)

        # Create fresh file with the expected header (best-effort single-writer)
        with open(path, "w", encoding="utf-8") as nf:
            self._stage3_file_lock(nf)
            try:
                nf.write(header0)
                nf.write(header1)
                nf.flush()
                os.fsync(nf.fileno())
            finally:
                self._stage3_file_unlock(nf)

    def _stage3_append_csv_row(self, path: str, columns: list[str], row: dict) -> None:
        # IMPORTANT:
        #   Must use csv.writer to correctly quote/escape fields that contain commas
        #   (e.g., JSON blobs in ships_json/episode_params_json/effective_json).
        import os, csv, json as _json
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", newline="", encoding="utf-8") as f:
            self._stage3_file_lock(f)
            try:
                # assume header already ensured
                w = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                values: list[str] = []
                for k in columns:
                    v = row.get(k, "")
                    if v is None:
                        v = ""
                    if isinstance(v, (dict, list)):
                        v = _json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                    values.append(str(v))
                w.writerow(values)
                f.flush()
                os.fsync(f.fileno())
            finally:
                self._stage3_file_unlock(f)

    def _stage3_append_jsonl(self, path: str, rec: dict) -> None:
        import os, json
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a+", encoding="utf-8") as f:
            self._stage3_file_lock(f)
            try:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                self._stage3_file_unlock(f)

    def stage3_flush_episode(self, term_reason: str = "", timeout: bool = False) -> None:
        """
        Stage3 closed-loop:
          - episodes_stage3.jsonl: episode meta + params (join keys included)
          - comm_stats_comm_stage3.csv: comm-only metrics (join keys included)
          - comm_stats_stage3.csv: per-agent metrics (join keys included; minimal baseline)
        """
        import time
        comm_stats_path, comm_stats_comm_path, episodes_path = self._stage3_paths_or_raise()

        # --- schema / columns (fixed) ---
        COMM_STATS_COMM_COLS = [
            "wall_time_unix","cfg_path","cfg_name","base_seed","ep_seed",
            "run_uuid","pid","worker_index","vector_index","episode_uid","episode_idx",
            "N","dt","ships_json","episode_params_json","effective_json",
            "tx_total","rx_delivered_total","rx_late_total","rx_ooo_total","rx_dropped_total",
            "pass_rate","drop_rate","deliver_rate","pdr",
            "delay_mean_s","delay_p95_s","delay_max_s",
            "age_mean_s","age_p95_s","age_max_s",
            "reorder_count","reorder_rate",
        ]
        COMM_STATS_COLS = COMM_STATS_COMM_COLS[:]  # baseline: same + agent_id; obs stats can be extended later
        COMM_STATS_COLS.insert(COMM_STATS_COLS.index("tx_total"), "agent_id")

        # Ensure headers (strict schema guard)
        self._stage3_ensure_csv_header(comm_stats_comm_path, "comm_stats_comm_stage3", COMM_STATS_COMM_COLS)
        self._stage3_ensure_csv_header(comm_stats_path, "comm_stats_stage3", COMM_STATS_COLS)

        # --- ensure Stage3 join keys exist (env may forget to set them) ---
        self._stage3_autofill_context_if_missing()

        # hard clamp (never emit -1 placeholders into files)
        try:
            self._stage3_worker_index = max(0, int(self._stage3_worker_index))
        except Exception:
            self._stage3_worker_index = 0
        try:
            self._stage3_vector_index = max(0, int(self._stage3_vector_index))
        except Exception:
            self._stage3_vector_index = 0

        if not str(self._stage3_run_uuid).strip():
            raise RuntimeError("stage3_run_uuid still empty after autofill.")
        if (not str(self._stage3_episode_uid).strip()) or int(self._stage3_episode_idx) < 0:
            raise RuntimeError("episode_uid/episode_idx still invalid after autofill.")

        # episode params snapshots (best-effort; keep your existing export formats)
        try:
            from dataclasses import asdict, is_dataclass
            ep = getattr(self, "_ep_params", None)
            if ep is not None and is_dataclass(ep):
                ep_params = asdict(ep)
            elif hasattr(ep, "__dict__"):
                ep_params = dict(ep.__dict__)
            else:
                ep_params = ep if isinstance(ep, dict) else {}
        except Exception:
            ep_params = {}

        # effective params are built at reset(); if missing, build best-effort now
        eff_params = getattr(self, "_effective", None)
        if not isinstance(eff_params, dict):
            eff_params = self._stage3_build_effective_snapshot()

        # Prefer authoritative source: agent_of_ship keys (always reflects current env wiring)
        try:
            ships_json = [int(sid) for sid in sorted(list(getattr(self, "agent_of_ship", {}).keys()))]
        except Exception:
            ships_json = []
        if not ships_json:
            # fallback to cached _ships if present
            try:
                ships_src = getattr(self, "_ships", None) or []
                ships_json = [int(s) for s in ships_src]
            except Exception:
                ships_json = []

        # N/dt should not be placeholders:
        #   - N is known at reset()
        #   - dt may not be explicitly set on AISCommsSim; estimate from recorded timeseries if needed.
        N_val = int(getattr(self, "N", 0) or (len(ships_json) if ships_json else 0))
        dt_val = float(getattr(self, "dt", 0.0) or 0.0)
        if dt_val <= 0.0:
            try:
                if len(getattr(self, "ts", [])) >= 2:
                    diffs = np.diff(np.asarray(self.ts, dtype=np.float64))
                    diffs = diffs[diffs > 1e-9]
                    dt_val = float(np.median(diffs)) if diffs.size > 0 else 0.0
            except Exception:
                dt_val = 0.0

        # episode length (best-effort, avoids None placeholders)
        ep_len_steps = None
        ep_len_s = None
        try:
            if len(getattr(self, "ts", [])) >= 2:
                ep_len_steps = int(len(self.ts) - 1)
                ep_len_s = float(self.ts[-1] - self.ts[0])
        except Exception:
            ep_len_steps, ep_len_s = None, None

        # comm metrics snapshot (aligned keys)
        snap = self.metrics_snapshot()

        # totals snapshot (if you already track them, align here; else default 0)
        tx_total = int(getattr(self._metrics, "tx_total", 0))
        rx_delivered_total = int(getattr(self._metrics, "rx_delivered_total", 0))
        rx_late_total = int(getattr(self._metrics, "rx_late_total", 0))
        rx_ooo_total = int(getattr(self._metrics, "rx_ooo_total", 0))
        rx_dropped_total = int(getattr(self._metrics, "rx_dropped_total", 0))

        # --- episodes_stage3.jsonl (one line per episode) ---
        ep_rec = {
            "wall_time_unix": time.time(),
            "run_uuid": self._stage3_run_uuid,
            "pid": __import__("os").getpid(),
            "worker_index": self._stage3_worker_index,
            "vector_index": self._stage3_vector_index,
            "episode_uid": self._stage3_episode_uid,
            "episode_idx": self._stage3_episode_idx,
            "cfg_path": getattr(self, "cfg_path", ""),
            "cfg_name": getattr(self, "cfg_name", self.cfg_name),
            "base_seed": int(getattr(self, "base_seed", 0)),
            "ep_seed": int(getattr(self, "_ep_seed", getattr(self, "ep_seed", 0))),
            "N": int(N_val),
            "dt": float(dt_val),
            "ships": ships_json,  # list[int]
            "episode_params": ep_params,
            "effective": eff_params,
            "term_reason": term_reason,
            "timeout": bool(timeout),
            "episode_len_steps": ep_len_steps,
            "episode_len_s": ep_len_s,
        }
        self._stage3_append_jsonl(episodes_path, ep_rec)

        # --- comm_stats_comm_stage3.csv (one row per episode) ---
        row_comm = {
            "wall_time_unix": time.time(),
            "cfg_path": getattr(self, "cfg_path", ""),
            "cfg_name": getattr(self, "cfg_name", self.cfg_name),
            "base_seed": int(getattr(self, "base_seed", 0)),
            "ep_seed": int(getattr(self, "_ep_seed", getattr(self, "ep_seed", 0))),
            "run_uuid": self._stage3_run_uuid,
            "pid": __import__("os").getpid(),
            "worker_index": self._stage3_worker_index,
            "vector_index": self._stage3_vector_index,
            "episode_uid": self._stage3_episode_uid,
            "episode_idx": self._stage3_episode_idx,

            "N": int(N_val),
            "dt": float(dt_val),
            "ships_json": ships_json,  # list[int] -> JSON string via csv writer

            "episode_params_json": ep_params,
            "effective_json": eff_params,
            "tx_total": tx_total,
            "rx_delivered_total": rx_delivered_total,
            "rx_late_total": rx_late_total,
            "rx_ooo_total": rx_ooo_total,
            "rx_dropped_total": rx_dropped_total,
            "pass_rate": snap.get("pass_rate", ""),
            "drop_rate": snap.get("drop_rate", ""),
            "deliver_rate": snap.get("deliver_rate", ""),
            "pdr": snap.get("pdr", ""),
            "delay_mean_s": snap.get("delay_mean_s", ""),
            "delay_p95_s": snap.get("delay_p95_s", ""),
            "delay_max_s": snap.get("delay_max_s", ""),
            "age_mean_s": snap.get("age_mean_s", ""),
            "age_p95_s": snap.get("age_p95_s", ""),
            "age_max_s": snap.get("age_max_s", ""),
            "reorder_count": snap.get("reorder_count", ""),
            "reorder_rate": snap.get("reorder_rate", ""),
        }
        self._stage3_append_csv_row(comm_stats_comm_path, COMM_STATS_COMM_COLS, row_comm)

        # --- comm_stats_stage3.csv (baseline per-agent rows) ---
        # If you already have per-agent obs stats elsewhere, extend here; for now just emit rows per agent_id.
        # Prefer authoritative mapping to avoid empty per-agent file when _agent_ids cache is missing/stale.
        agent_ids = []
        try:
            aof = getattr(self, "agent_of_ship", {}) or {}
            agent_ids = [str(aof[sid]) for sid in sorted(aof.keys())]
        except Exception:
            agent_ids = []
        if not agent_ids:
            agent_ids = list(getattr(self, "_agent_ids", [])) or []

        for aid in agent_ids:
            row = dict(row_comm)
            row["agent_id"] = str(aid)
            self._stage3_append_csv_row(comm_stats_path, COMM_STATS_COLS, row)

        # IMPORTANT: invalidate per-episode join keys after flush.
        # This prevents the next episode from silently reusing stale keys if the env forgets to set context.
        self._stage3_episode_uid = ""
        self._stage3_episode_idx = -1

    def set_stage3_run_context(
        self,
        run_uuid: str,
        worker_index: int,
        vector_index: int,
        stage3_schema_version: str = "stage3_v1",
        comm_stats_path: str | None = None,
        comm_stats_comm_path: str | None = None,
        episodes_path: str | None = None,
    ) -> None:
        ru = str(run_uuid).strip()
        self._stage3_run_uuid = ru if ru else str(uuid4())
        # Best-effort infer worker/env indices if caller passes -1
        wi = int(worker_index)
        vi = int(vector_index)
        if wi < 0:
            for k in ("RAY_WORKER_INDEX", "RLLIB_WORKER_INDEX", "WORKER_INDEX"):
                if os.environ.get(k, ""):
                    try:
                        wi = int(os.environ[k])
                        break
                    except Exception:
                        pass
        if vi < 0:
            for k in ("RAY_ENV_INDEX", "RLLIB_ENV_INDEX", "ENV_INDEX", "ENV_ID"):
                if os.environ.get(k, ""):
                    try:
                        vi = int(os.environ[k])
                        break
                    except Exception:
                        pass
        # enforce non-negative defaults (avoid -1 placeholders)
        self._stage3_worker_index = wi if wi >= 0 else 0
        self._stage3_vector_index = vi if vi >= 0 else 0

        self._stage3_schema_version = str(stage3_schema_version)
        self._stage3_comm_stats_path = comm_stats_path
        self._stage3_comm_stats_comm_path = comm_stats_comm_path
        self._stage3_episodes_path = episodes_path

    def set_stage3_episode_context(self, episode_uid: str, episode_idx: int) -> None:
        self._stage3_episode_uid = str(episode_uid)
        self._stage3_episode_idx = int(episode_idx)

    # ---------------- lifecycle ----------------
    def reset(self, ships: List[ShipId], t0: Ts, agent_map: Dict[ShipId, AgentId]):
        """每个 episode 开始调用。"""
        self.agent_of_ship = dict(agent_map)
        # Reverse mapping: agent_id -> ship_id (for distance-based loss)
        self._ship_of_agent: Dict[AgentId, ShipId] = {
            str(aid): int(sid) for sid, aid in agent_map.items()
        }
        # Stage3 required caches (avoid placeholder outputs)
        # Use stable ship_id order for deterministic stage3 outputs.
        ship_ids_sorted = [int(s) for s in sorted(list(ships))]
        self._ships = ship_ids_sorted
        self._agent_ids = [str(self.agent_of_ship[int(sid)]) for sid in ship_ids_sorted if int(sid) in self.agent_of_ship]
        self.N = int(len(ship_ids_sorted))

        assert len(set(self.agent_of_ship.values())) == len(self.agent_of_ship), f"agent_id not unique! agent_of_ship={self.agent_of_ship}"
        # 默认分配内部测试 MMSI（全局唯一）
        self.mmsi_of_ship = {sid: 999000000 + int(sid) for sid in ships}
        assert len(set(self.mmsi_of_ship.values())) == len(self.mmsi_of_ship), "MMSI not unique!"
        assert set(self.agent_of_ship.keys()) == set(self.mmsi_of_ship.keys()), "ship_id map mismatch!"

        # 队列与调度器复位
        self.arrivals = ArrivalQueue()
        for sid in ships:
            self.scheduler.reset_ship(sid, t0)

        # 指标复位
        self.reset_metrics()

        # AR(1) 噪声状态复位
        self._sog_ar1_state.clear()
        self._cog_ar1_state.clear()

        # Previous yaw tracking for rot computation
        self._prev_yaw.clear()
        self._prev_t.clear()

        # 采样 episode 参数并应用
        ep_seed = self.base_seed + int(self.rng.integers(1, 10_000_000))
        self._ep_seed = int(ep_seed)  # 可选：存下来便于 debug/复现
        # legacy alias (some writers read ep_seed instead of _ep_seed)
        self.ep_seed = int(self._ep_seed)

        self._ep_params = sample_episode_params(self.cfg, seed=self._ep_seed)
        self._apply_episode_params(self._ep_params)

        # ✅ 放在这里：reset() 作用域里有 ep_seed
        if hasattr(self, "scheduler") and self.scheduler is not None:
            self.scheduler.reseed(self._ep_seed + 2025)
            


        # === PATCH: delay/reorder 独立 RNG（保证 reorder 开关不扰动主 RNG stream） ===
        # 用 ep_seed 派生，确保每个 episode 内可复现 & 对照实验严格可比
        self.rng_delay = np.random.default_rng(int(ep_seed) + 424242)
        self.rng_reorder = np.random.default_rng(int(ep_seed) + 1337)

        # per-ship clock params
        self.clock_of_ship.clear()
        if self.clock_enable:
            ck = self.cfg.get("clock", {}) or {}
            off_cfg = ck.get("offset_s", self.clock_offset_s)
            drift_cfg = ck.get("drift_ppm", self.clock_drift_ppm)

            def _pick_ship(val, fallback):
                try:
                    if isinstance(val, (list, tuple)) and len(val) == 2:
                        lo, hi = float(val[0]), float(val[1])
                        return float(self.rng.uniform(lo, hi))
                    return float(val)
                except Exception:
                    return float(fallback)

            for sid in ships:
                off = _pick_ship(off_cfg, self.clock_offset_s)
                drf = _pick_ship(drift_cfg, self.clock_drift_ppm)
                self.clock_of_ship[sid] = (float(off), float(drf))
        else:
            for sid in ships:
                self.clock_of_ship[sid] = (0.0, 0.0)

        # GPS bias + drift 初始化
        self._gps_bias_xy.clear()
        self._gps_drift_xy.clear()
        self._gps_bias_t0 = float(t0)
        self._gps_dbg_rows = []

        if self.gps_bias_enable:
            for sid in ships:
                bx = float(self.rng.uniform(-self.gps_pos_bias_max, self.gps_pos_bias_max))
                by = float(self.rng.uniform(-self.gps_pos_bias_max, self.gps_pos_bias_max))
                vx = float(self.rng.uniform(-self.gps_drift_max_mps, self.gps_drift_max_mps))
                vy = float(self.rng.uniform(-self.gps_drift_max_mps, self.gps_drift_max_mps))
                self._gps_bias_xy[sid] = (bx, by)
                self._gps_drift_xy[sid] = (vx, vy)
        else:
            for sid in ships:
                self._gps_bias_xy[sid] = (0.0, 0.0)
                self._gps_drift_xy[sid] = (0.0, 0.0)

        # 欺诈历史复位
        if self.fraud_wrapper is not None:
            self.fraud_wrapper.reset_episode()

        # 关键：时间推进参考点 & 清空链路 channel（每局重新开始）
        self._last_tick_t = float(t0)
        self.channel_by_link.clear()
        # ---- Stage3 context & effective snapshot (must be ready before episode ends) ----
        self._stage3_autofill_context_if_missing()
        self._effective = self._stage3_build_effective_snapshot()
        log_cfg = self.cfg.get("logging", {}) or {}
        if DEBUG_AIS and log_cfg.get("print_params", False):
            print("[AIS] episode params:", self._ep_params)

        if log_cfg.get("export_episode_params", True):
            outp = os.path.abspath(f"ais_episode_params_{int(t0)}.json")
            export_params_json(outp, self._ep_params)

        # =========================
        # Phase-2: episodes.jsonl
        #   - only append ONE line per episode (JSONL)
        #   - record sampled EpisodeParams for traceability / paper statistics
        # =========================
        if log_cfg.get("export_episodes_jsonl", True):
            try:
                import time
                from dataclasses import asdict, is_dataclass

                # ---------- identifiers (pid / run_uuid / worker / env) ----------
                pid = int(os.getpid())

                # run_uuid priority:
                #   1) env var SHIPRL_RUN_UUID (recommended: set once per training run)
                #   2) cached on self._run_uuid (auto-generated per-process if absent)
                run_uuid = str(os.environ.get("SHIPRL_RUN_UUID", "")).strip()
                if not run_uuid:
                    run_uuid = str(getattr(self, "_run_uuid", "")).strip()
                if not run_uuid:
                    run_uuid = str(uuid4())
                    setattr(self, "_run_uuid", run_uuid)

                # best-effort RLlib/Ray worker/env identifiers (may be empty if not set)
                worker_id = (
                    os.environ.get("RAY_WORKER_INDEX", None)
                    or os.environ.get("RLLIB_WORKER_INDEX", None)
                    or os.environ.get("WORKER_INDEX", None)
                    or ""
                )
                env_id = (
                    os.environ.get("RAY_ENV_INDEX", None)
                    or os.environ.get("RLLIB_ENV_INDEX", None)
                    or os.environ.get("ENV_INDEX", None)
                    or os.environ.get("ENV_ID", None)
                    or ""
                )

                # output path priority:
                #   1) env var AIS_EPISODES_JSONL
                #   2) YAML: logging.episodes_jsonl_path
                #   3) default: ./episodes.jsonl
                jsonl_path = (
                    os.environ.get("AIS_EPISODES_JSONL", None)
                    or log_cfg.get("episodes_jsonl_path", None)
                    or "episodes.jsonl"
                )
                jsonl_path = os.path.abspath(str(jsonl_path))
                os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)

                # Guard: never pollute stage3 episodes jsonl with legacy schema
                st3p = getattr(self, "_stage3_episodes_path", None)
                if st3p:
                    try:
                        if os.path.abspath(str(st3p)) == jsonl_path:
                            # Skip legacy episodes logger to prevent schema mixing
                            raise RuntimeError("Skip legacy episodes.jsonl: path equals stage3 episodes_path.")
                    except Exception:
                        pass

                ep = self._ep_params
                if ep is not None and is_dataclass(ep):
                    ep_d = asdict(ep)
                elif hasattr(ep, "__dict__"):
                    ep_d = dict(ep.__dict__)
                else:
                    ep_d = ep

                ep_seed_i = int(getattr(self, "_ep_seed", -1))

                # ---------- effective params snapshot (post-apply, post-clamp, post-per-ship sampling) ----------
                # Note:
                #   - delay uses internal lognormal params (mu_ln/sigma_ln/max_delay) after clamp
                #   - noise_cog stored as radians internally
                #   - clock_of_ship is per-ship effective offset/drift used for reported_ts
                #   - gps bias/drift are per-ship effective values if enabled
                try:
                    clock_of_ship = {
                        str(int(sid)): {"offset_s": float(off), "drift_ppm": float(dppm)}
                        for sid, (off, dppm) in getattr(self, "clock_of_ship", {}).items()
                    }
                except Exception:
                    clock_of_ship = {}
                try:
                    gps_bias_xy = {
                        str(int(sid)): [float(xy[0]), float(xy[1])]
                        for sid, xy in getattr(self, "_gps_bias_xy", {}).items()
                    }
                except Exception:
                    gps_bias_xy = {}

                try:
                    gps_drift_xy = {
                        str(int(sid)): [float(xy[0]), float(xy[1])]
                        for sid, xy in getattr(self, "_gps_drift_xy", {}).items()
                    }
                except Exception:
                    gps_drift_xy = {}

                effective = dict(
                    # derived seeds (deterministic given ep_seed_i by design)
                    derived_seeds=dict(
                        ep_seed=int(ep_seed_i),
                        scheduler=(int(ep_seed_i + 2025) if ep_seed_i >= 0 else -1),
                        rng_delay=(int(ep_seed_i + 424242) if ep_seed_i >= 0 else -1),
                        rng_reorder=(int(ep_seed_i + 1337) if ep_seed_i >= 0 else -1),
                    ),
                    # delay (internal)
                    delay=dict(
                        mu_ln=float(getattr(self, "mu_ln", 0.0)),
                        sigma_ln=float(getattr(self, "sigma_ln", 0.0)),
                        clip_s=float(getattr(self, "max_delay", 0.0)),
                    ),
                    # obs cache (internal)
                    obs_cache=dict(
                        slot_K=int(getattr(self, "obs_slot_K", 0)),
                        slot_ttl_s=float(getattr(self, "obs_slot_ttl", 0.0)),
                        age_cap_s=float(getattr(self, "obs_age_cap", 0.0)),
                        miss_mask=bool(getattr(self, "obs_miss_mask", False)),
                    ),
                    # noise/bias (internal)
                    noise=dict(
                        pos_m=float(getattr(self, "noise_pos_m", 0.0)),
                        sog_mps=float(getattr(self, "noise_sog_mps", 0.0)),
                        cog_rad=float(getattr(self, "noise_cog_rad", 0.0)),
                        rot_rads=float(getattr(self, "noise_rot_rads", 0.01)),
                        reg_bias_enable=bool(getattr(self, "reg_bias_enable", False)),
                    ),
                    # reorder (internal)
                    reorder=dict(
                        enable=bool(getattr(self, "reorder_enable", False)),
                        prob=float(getattr(self, "reorder_prob", 0.0)),
                        extra_delay_s=float(getattr(self, "reorder_extra_delay", 0.0)),
                    ),
                    # clock (effective per ship)
                    clock=dict(
                        enable=bool(getattr(self, "clock_enable", False)),
                        per_ship=clock_of_ship,
                    ),
                    # gps bias/drift (effective per ship)
                    gps=dict(
                        enable=bool(getattr(self, "gps_bias_enable", False)),
                        t0=float(getattr(self, "_gps_bias_t0", 0.0)),
                        bias_xy=gps_bias_xy,
                        drift_xy=gps_drift_xy,
                    ),
                    # distance-based loss
                    distance=dict(
                        enable=bool(getattr(self, "dist_enable", False)),
                        ref_m=float(getattr(self, "dist_ref_m", 18520.0)),
                        max_m=float(getattr(self, "dist_max_m", 55560.0)),
                        loss_exp=float(getattr(self, "dist_loss_exp", 2.0)),
                    ),
                )


                rec = dict(
                    t0=float(t0),
                    cfg_path=str(self.cfg_path),
                    cfg_name=str(self.cfg.get("name", "")) if isinstance(self.cfg, dict) else "",

                    pid=pid,
                    run_uuid=str(run_uuid),
                    worker=str(worker_id),
                    env=str(env_id),

                    base_seed=int(self.base_seed),
                    ep_seed=int(ep_seed_i),
                    ships=[int(s) for s in ships],
                    episode_params=ep_d,
                    effective=effective,
                    wall_time_unix=float(time.time()),
                )
 
                line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))

                # best-effort inter-process safe append (Linux flock)
                try:
                    import fcntl
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                        f.write(line + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
            except Exception:
                # never break env reset because of logging
                pass
 

    # ---------------- metrics ----------------
    def reset_metrics(self):
        self.ts, self.ppr_s, self.pdr_s = [], [], []
        self.dly_avg_s, self.age_avg_s = [], []

        self._link_attempts = 0
        self._link_passed = 0
        self._link_dropped = 0
        self._link_delivered = 0

        self.delay_samples: List[float] = []
        self.age_samples: List[float] = []
        self.age_minus_delay_samples = []

        # --- Per-ship metrics (for diagnostic analysis) ---
        # tx_ship -> list of delay/age samples
        self._per_ship_delay: Dict[int, List[float]] = {}
        self._per_ship_age: Dict[int, List[float]] = {}
        self._per_ship_rx_count: Dict[int, int] = {}
        # Last RX info per ship (for PF error attribution)
        self._last_rx_by_ship: Dict[int, Dict[str, float]] = {}

        self._link_samples = []
        self._ar1_samples = []

        self.bad_occ_s, self.delay_p95_s, self.age_p95_s = [], [], []
        self.age_delay_gap_avg_s, self.age_delay_gap_p95_s = [], []

        self._last_arrival_by_link = {}
        self._reorder_total = 0
        self._reorder_hit = 0
        self._reorder_samples = []

        # --- COMPAT: legacy metrics container expected by metrics_snapshot/stage3 writers ---
        self._metrics = SimpleNamespace(
            # counters
            tx_attempts=0, passed=0, dropped=0, delivered=0,
            # ratios
            ppr=0.0, pdr=0.0,
            # delay/age stats
            delay_avg=0.0, delay_p95=0.0, delay_max=0.0,
            age_avg=0.0, age_p95=0.0, age_max=0.0,
            bad_occupancy=0.0,
            age_delay_gap_avg=0.0, age_delay_gap_p95=0.0,
            # reorder
            reorder_count=0, reorder_rate=0.0,

            # stage3_flush_episode currently expects these (keep them even if you don’t yet use them)
            tx_total=0,
            rx_delivered_total=0,
            rx_late_total=0,
            rx_ooo_total=0,
            rx_dropped_total=0,
        )

    def _bad_occupancy_avg(self) -> float:
        """per-link GE 的坏态占用率平均（仅用于时序指标展示）。"""
        if not self.channel_by_link:
            return 0.0
        vals = [ch.bad_occupancy() for ch in self.channel_by_link.values()]
        return float(np.mean(vals)) if vals else 0.0

    # =============================================================================
    # Stage-3 Metrics Schema (DO NOT RENAME without updating stage3 writers)
    #
    # This schema is consumed by:
    #   - MiniShipAISCommsEnv Stage-3 comm_stats writer (per-step, per-agent)
    #   - AISCommsSim.export_csv() timeseries dump
    #
    # Canonical (Stage-3) keys returned by metrics_snapshot():
    #   Counters:
    #     tx_attempts, passed, dropped, delivered
    #   Ratios:
    #     ppr (passed/attempts), pdr (delivered/attempts)
    #   Delay/Age (seconds):
    #     delay_avg, delay_p95, age_avg, age_p95
    #   Channel state:
    #     bad_occupancy (avg GE bad-state occupancy across links)
    #   Age-Delay gap (seconds):
    #     age_delay_gap_avg, age_delay_gap_p95
    #   Reordering:
    #     reorder_rate
    #
    # The Stage-3 comm_stats file only requires the subset below:
    #   ppr, pdr, delay_avg, age_avg, bad_occupancy, delay_p95, age_p95,
    #   age_delay_gap_avg, age_delay_gap_p95
    # =============================================================================
    STAGE3_COMM_METRICS_KEYS = [
        "ppr", "pdr",
        "delay_avg", "age_avg",
        "bad_occupancy",
        "delay_p95", "age_p95",
        "age_delay_gap_avg", "age_delay_gap_p95",
    ]

    def metrics_snapshot(self) -> Dict[str, float]:
        """
        Stage-3 canonical snapshot for logging/export.

        All values are computed from internal counters/samples and then
        mirrored into self._metrics for backward compatibility.
        """
        import numpy as _np
        from types import SimpleNamespace

        # Ensure legacy container exists (robust against future refactors)
        if not hasattr(self, "_metrics") or self._metrics is None:
            self._metrics = SimpleNamespace()

        # -------- counters --------
        tx = int(getattr(self, "_link_attempts", 0))
        pa = int(getattr(self, "_link_passed", 0))
        dr = int(getattr(self, "_link_dropped", 0))
        dv = int(getattr(self, "_link_delivered", 0))

        # -------- reducers --------
        def _p95(arr) -> float:
            return float(_np.percentile(arr, 95)) if arr else 0.0

        delay_avg = float(_np.mean(self.delay_samples)) if self.delay_samples else 0.0
        delay_p95 = _p95(self.delay_samples)
        delay_max = float(max(self.delay_samples)) if self.delay_samples else 0.0

        age_avg = float(_np.mean(self.age_samples)) if self.age_samples else 0.0
        age_p95 = _p95(self.age_samples)
        age_max = float(max(self.age_samples)) if self.age_samples else 0.0

        # Age-Delay gap (clock offset effect); keep signed statistic as in your existing pipeline
        age_delay_gap_avg = float(_np.mean(self.age_minus_delay_samples)) if self.age_minus_delay_samples else 0.0
        age_delay_gap_p95 = _p95(self.age_minus_delay_samples)

        # ratios
        ppr = float(pa / tx) if tx > 0 else 0.0
        pdr = float(dv / tx) if tx > 0 else 0.0

        # channel occupancy (avg over links)
        bad_occupancy = float(self._bad_occupancy_avg())

        # reorder
        reorder_count = int(getattr(self, "_reorder_hit", 0))
        reorder_rate = float(reorder_count / self._reorder_total) if getattr(self, "_reorder_total", 0) > 0 else 0.0

        # -------- mirror into legacy container (for stage3 writers / backward compat) --------
        self._metrics.tx_attempts = tx
        self._metrics.passed = pa
        self._metrics.dropped = dr
        self._metrics.delivered = dv

        self._metrics.ppr = ppr
        self._metrics.pdr = pdr

        self._metrics.delay_avg = delay_avg
        self._metrics.delay_p95 = delay_p95
        self._metrics.delay_max = delay_max

        self._metrics.age_avg = age_avg
        self._metrics.age_p95 = age_p95
        self._metrics.age_max = age_max

        self._metrics.bad_occupancy = bad_occupancy
        self._metrics.age_delay_gap_avg = age_delay_gap_avg
        self._metrics.age_delay_gap_p95 = age_delay_gap_p95

        self._metrics.reorder_count = reorder_count
        self._metrics.reorder_rate = reorder_rate

        # Optional: keep the “total” aliases used by stage3_flush_episode (same meaning in your current code)
        self._metrics.tx_total = tx
        self._metrics.rx_delivered_total = dv
        self._metrics.rx_late_total = int(getattr(self._metrics, "rx_late_total", 0))
        self._metrics.rx_ooo_total = reorder_count
        self._metrics.rx_dropped_total = dr

        # -------- return snapshot dict (keep your existing keys + stage3 CSV keys) --------
        snap = {
            # canonical keys
            "ppr": ppr,
            "pdr": pdr,
            "delay_avg": delay_avg,
            "delay_p95": delay_p95,
            "delay_max": delay_max,
            "age_avg": age_avg,
            "age_p95": age_p95,
            "age_max": age_max,
            "reorder_count": reorder_count,
            "reorder_rate": reorder_rate,
            "bad_occupancy": bad_occupancy,
            "age_delay_gap_avg": age_delay_gap_avg,
            "age_delay_gap_p95": age_delay_gap_p95,

            # stage3 / CSV fields (explicit mapping)
            "pass_rate": ppr,
            "drop_rate": float(dr / tx) if tx > 0 else 0.0,
            "deliver_rate": pdr,
            "delay_mean_s": delay_avg,
            "delay_p95_s": delay_p95,
            "delay_max_s": delay_max,
            "age_mean_s": age_avg,
            "age_p95_s": age_p95,
            "age_max_s": age_max,
        }
        return snap

    def get_per_ship_metrics(self) -> Dict[int, Dict[str, Any]]:
        """
        Per-ship communication metrics for diagnostic analysis.

        Enables attributing PF errors to specific ship's communication quality:
        - Which ship's messages are delayed/dropped
        - Last RX timing for each ship (for PF error causality)

        Returns:
            Dict[ship_id, {
                "rx_count": int,          # Total messages received from this ship
                "delay_mean": float,      # Mean delay (seconds)
                "delay_p95": float,       # 95th percentile delay
                "delay_max": float,
                "age_mean": float,        # Mean message age
                "age_p95": float,
                "age_max": float,
                "last_rx_delay_s": float, # Most recent delay (for attribution)
                "last_rx_age_s": float,   # Most recent age
                "last_rx_report_ts": float,  # Timestamp in last message
                "last_rx_arrival_ts": float, # When last message arrived
            }]
        """
        import numpy as np

        def _p95(arr) -> float:
            return float(np.percentile(arr, 95)) if len(arr) >= 2 else (float(max(arr)) if arr else 0.0)

        result = {}
        for sid in self._ships:
            delays = self._per_ship_delay.get(sid, [])
            ages = self._per_ship_age.get(sid, [])
            last_rx = self._last_rx_by_ship.get(sid, {})

            result[sid] = {
                "rx_count": self._per_ship_rx_count.get(sid, 0),
                "delay_mean": float(np.mean(delays)) if delays else 0.0,
                "delay_p95": _p95(delays),
                "delay_max": float(max(delays)) if delays else 0.0,
                "age_mean": float(np.mean(ages)) if ages else 0.0,
                "age_p95": _p95(ages),
                "age_max": float(max(ages)) if ages else 0.0,
                # Last RX info (most recent message from this ship)
                "last_rx_delay_s": last_rx.get("last_rx_delay_s", 0.0),
                "last_rx_age_s": last_rx.get("last_rx_age_s", 0.0),
                "last_rx_report_ts": last_rx.get("last_rx_report_ts", 0.0),
                "last_rx_arrival_ts": last_rx.get("last_rx_arrival_ts", 0.0),
            }

        return result

    def get_episode_params(self) -> Dict[str, Any]:
        """
        Get complete episode parameters for staging recording.

        Returns a dict with:
          - episode_params: raw sampled parameters for this episode
          - effective: derived/computed parameters actually in use
          - config_path: path to the config file used
          - ep_seed: episode seed

        This method is the single source of truth for episode-level AIS params.
        """
        from dataclasses import is_dataclass, asdict

        result: Dict[str, Any] = {
            "config_path": str(getattr(self, "cfg_path", "")),
            "ep_seed": int(getattr(self, "_ep_seed", getattr(self, "ep_seed", 0))),
        }

        # Raw episode params
        try:
            ep = getattr(self, "_ep_params", None)
            if ep is not None and is_dataclass(ep):
                result["episode_params"] = asdict(ep)
            elif hasattr(ep, "__dict__"):
                result["episode_params"] = dict(ep.__dict__)
            elif isinstance(ep, dict):
                result["episode_params"] = ep
            else:
                result["episode_params"] = {}
        except Exception:
            result["episode_params"] = {}

        # Effective params (computed at reset)
        try:
            result["effective"] = self._stage3_build_effective_snapshot()
        except Exception:
            result["effective"] = {}

        return result

    def record_timeseries(self, t: Ts):
        m = self.metrics_snapshot()
        self.ts.append(float(t))
        self.ppr_s.append(float(m["ppr"]))
        self.pdr_s.append(float(m["pdr"]))
        self.dly_avg_s.append(float(m["delay_avg"]))
        self.age_avg_s.append(float(m["age_avg"]))

        self.bad_occ_s.append(float(m["bad_occupancy"]))
        self.delay_p95_s.append(float(m["delay_p95"]))
        self.age_p95_s.append(float(m["age_p95"]))
        self.age_delay_gap_avg_s.append(float(m["age_delay_gap_avg"]))
        self.age_delay_gap_p95_s.append(float(m["age_delay_gap_p95"]))

    # ---------------- export ----------------
    def export_csv(self, path: str):
        import csv, os

        snap = self.metrics_snapshot()
        bad_occ_last = float(snap.get("bad_occupancy", 0.0))
        dly_p95_last = float(snap.get("delay_p95", 0.0))
        age_p95_last = float(snap.get("age_p95", 0.0))
        amd_avg_last = float(snap.get("age_delay_gap_avg", 0.0))
        amd_p95_last = float(snap.get("age_delay_gap_p95", 0.0))

        log_cfg = self.cfg.get("logging", {}) or {}

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            # Stage-3 timeseries header (schema-locked)
            w.writerow(["t"] + self.STAGE3_COMM_METRICS_KEYS)

            T = len(self.ts)
            for i in range(T):
                t_i = float(self.ts[i]) if i < len(self.ts) else 0.0
                ppr_i = float(self.ppr_s[i]) if i < len(self.ppr_s) else float(snap.get("ppr", 0.0))
                pdr_i = float(self.pdr_s[i]) if i < len(self.pdr_s) else float(snap.get("pdr", 0.0))
                dly_i = float(self.dly_avg_s[i]) if i < len(self.dly_avg_s) else float(snap.get("delay_avg", 0.0))
                age_i = float(self.age_avg_s[i]) if i < len(self.age_avg_s) else float(snap.get("age_avg", 0.0))

                bad_occ_i = float(self.bad_occ_s[i]) if i < len(self.bad_occ_s) else bad_occ_last
                dly_p95_i = float(self.delay_p95_s[i]) if i < len(self.delay_p95_s) else dly_p95_last
                age_p95_i = float(self.age_p95_s[i]) if i < len(self.age_p95_s) else age_p95_last
                amd_avg_i = float(self.age_delay_gap_avg_s[i]) if i < len(self.age_delay_gap_avg_s) else amd_avg_last
                amd_p95_i = float(self.age_delay_gap_p95_s[i]) if i < len(self.age_delay_gap_p95_s) else amd_p95_last

                w.writerow([
                    t_i, ppr_i, pdr_i,
                    dly_i, age_i,
                    bad_occ_i, dly_p95_i, age_p95_i,
                    amd_avg_i, amd_p95_i
                ])

        # link samples
        if log_cfg.get("export_link_samples", False) and self._link_samples:
            link_path = log_cfg.get("link_samples_path", None)
            if not link_path:
                base, _ = os.path.splitext(path)
                link_path = base + "_link_samples.csv"

            with open(link_path, "w", newline="") as lf:
                dw = csv.DictWriter(lf, fieldnames=[
                    "t_true", "tx_ship", "rx_agent", "delay",
                    "reported_ts", "arrival_time", "age", "age_minus_delay",
                    "offset_s", "drift_ppm", "expected_offset_at_t",
                    "is_reordered",
                ])
                dw.writeheader()
                for row in self._link_samples:
                    dw.writerow(row)

            if DEBUG_AIS:
                print(f"[Saved] link samples CSV -> {os.path.abspath(link_path)}")

        # ar1 samples
        if log_cfg.get("export_ar1_samples", True) and self._ar1_samples:
            ar1_path = log_cfg.get("ar1_samples_path", None)
            if not ar1_path:
                base, _ = os.path.splitext(path)
                ar1_path = base + "_ar1_noise.csv"

            with open(ar1_path, "w", newline="") as af:
                dw = csv.DictWriter(af, fieldnames=[
                    "t_true",
                    "tx_ship",
                    "sog_true", "cog_true",
                    "sog_noise", "cog_noise",
                    "sog_noisy", "cog_noisy",
                ])
                dw.writeheader()
                for row in self._ar1_samples:
                    dw.writerow(row)

            if DEBUG_AIS:
                print(f"[Saved] AR(1) noise CSV -> {os.path.abspath(ar1_path)}")

        # gps debug
        if self._gps_dbg_rows:
            gps_path = os.path.abspath("ais_metrics_gps_bias.csv")
            with open(gps_path, "w", newline="") as gf:
                dw = csv.DictWriter(gf, fieldnames=[
                    "t_true", "tx_ship",
                    "x_true", "y_true",
                    "bx", "by", "vx", "vy",
                    "x_noisy", "y_noisy",
                    "dx", "dy",
                ])
                dw.writeheader()
                for row in self._gps_dbg_rows:
                    dw.writerow(row)
            if DEBUG_AIS:
                print(f"[Saved] GPS bias+drift CSV -> {gps_path}")

        # reorder samples
        if log_cfg.get("export_reorder_samples", False) and self._reorder_samples:
            reorder_path = log_cfg.get("reorder_samples_path", None)
            if not reorder_path:
                base, _ = os.path.splitext(path)
                reorder_path = base + "_reorder_samples.csv"

            with open(reorder_path, "w", newline="") as rf:
                dw = csv.DictWriter(rf, fieldnames=[
                    "t_true", "tx_ship", "rx_agent",
                    "delay", "arrival_time",
                    "last_arrival_prev", "is_reordered",
                ])
                dw.writeheader()
                for row in self._reorder_samples:
                    dw.writerow(row)

            if DEBUG_AIS:
                print(f"[Saved] reorder samples CSV -> {os.path.abspath(reorder_path)}")

            if self._reorder_total > 0:
                rate = self._reorder_hit / float(self._reorder_total)
                print(f"[AIS] reorder stats: {self._reorder_hit}/{self._reorder_total} = {rate:.3f}")

    # ---------------- episode params ----------------
    def _apply_episode_params(self, ep):
        # 保存 GE 参数（新链路建 channel 时复用）
        # Distance-based loss parameters (with defaults for backward compatibility)
        self.dist_enable = bool(getattr(ep, "dist_enable", False))
        self.dist_ref_m = float(getattr(ep, "dist_ref_m", 18520.0))
        self.dist_max_m = float(getattr(ep, "dist_max_m", 55560.0))
        self.dist_loss_exp = float(getattr(ep, "dist_loss_exp", 2.0))

        self._ge_params = dict(
            p_g2b=float(ep.ge_p_g2b),
            p_b2g=float(ep.ge_p_b2g),
            drop_bad=bool(ep.ge_drop_bad),
            burst_enable=bool(ep.burst_enable),
            burst_prob=float(ep.burst_prob),
            burst_dur_s=float(ep.burst_dur_s),
            burst_extra_drop=float(ep.burst_extra_drop),
            dist_enable=self.dist_enable,
            dist_ref_m=self.dist_ref_m,
            dist_max_m=self.dist_max_m,
            dist_loss_exp=self.dist_loss_exp,
        )

        # delay
        self.mu_ln = float(np.log(max(1e-3, ep.delay_mu)))
        self.sigma_ln = float(ep.delay_sigma)
        self.max_delay = float(ep.delay_clip)

        # clock
        self.clock_enable = bool(ep.clock_enable)
        self.clock_offset_s = float(ep.clock_offset_s)
        self.clock_drift_ppm = float(ep.clock_drift_ppm)

        # scheduler
        self.scheduler.set_params(
            base_period=ep.sch_base_period_s,
            jitter_frac=ep.sch_jitter_frac,
            sog_breaks=ep.sch_breaks,
            sog_periods=ep.sch_periods,
            drop_on_idle=False,
            drop_prob_on_idle=ep.sch_drop_prob_on_idle,
            sog_idle_thr_mps=ep.sch_idle_thr_mps,
        )

        # queue
        self.arrivals.set_limits(ttl_s=ep.q_ttl_s, max_inflight=ep.q_max_inflight)

        # obs cache
        self.obs_slot_K = int(ep.obs_slot_K)
        self.obs_slot_ttl = float(ep.obs_slot_ttl_s)
        self.obs_age_cap = float(ep.obs_age_cap_s)
        self.obs_miss_mask = bool(ep.obs_miss_mask)

        # noise / bias
        self.noise_pos_m = float(abs(ep.noise_pos_m))
        self.noise_sog_mps = float(abs(ep.noise_sog_mps))
        self.noise_cog_rad = float(np.deg2rad(abs(ep.noise_cog_deg)))

        self.reg_bias_enable = bool(ep.reg_bias_enable)
        self.reg_bias_rects = list(ep.reg_bias_rects)

        # field errors
        self.fe_mmsi_conflict_prob = float(ep.fe_mmsi_conflict_prob)
        self.fe_type_missing_prob = float(ep.fe_type_missing_prob)
        self.fe_draft_missing_prob = float(ep.fe_draft_missing_prob)
        self.fe_sog_zero_stick_prob = float(ep.fe_sog_zero_stick_prob)

        # reorder (NOW: episode-driven)
        self.reorder_enable = bool(getattr(ep, "reorder_enable", False))
        self.reorder_prob   = float(getattr(ep, "reorder_prob", 0.0))
        self.reorder_extra_delay = float(getattr(ep, "reorder_extra_delay_s", 0.0))

        #print("[AIS][EP] reorder_enable=", self.reorder_enable,"reorder_prob=", self.reorder_prob,"extra_delay_s=", self.reorder_extra_delay)

    # ---------------- helpers ----------------
    def _get_link_channel(self, link_key: Tuple[int, str]) -> GEChannel:
        """按 (tx_ship, rx_agent) 获取独立 GEChannel；不存在则创建并注入 episode 参数。"""
        ch = self.channel_by_link.get(link_key, None)
        if ch is not None:
            return ch

        p = self._ge_params
        ch = GEChannel(
            rng=self.rng,
            p_g2b=p["p_g2b"],
            p_b2g=p["p_b2g"],
            drop_bad=p["drop_bad"],
        )
        # 若 GEChannel 支持 set_params，则把 burst 也注入
        if hasattr(ch, "set_params"):
            try:
                ch.set_params(
                    p_g2b=p["p_g2b"],
                    p_b2g=p["p_b2g"],
                    drop_bad=p["drop_bad"],
                    burst_enable=p["burst_enable"],
                    burst_prob=p["burst_prob"],
                    burst_dur_s=p["burst_dur_s"],
                    burst_extra_drop=p["burst_extra_drop"],
                    step_dt=1.0,
                )
            except Exception:
                pass

        self.channel_by_link[link_key] = ch
        return ch

    def _apply_regional_bias(self, x: float, y: float, sog: float):
        if not self.reg_bias_enable:
            return x, y, sog
        for rect in self.reg_bias_rects:
            try:
                (xr, yr, pos_rng, sog_rng) = rect
                if xr[0] <= x <= xr[1] and yr[0] <= y <= yr[1]:
                    pos_bias = float(self.rng.uniform(*pos_rng))
                    sog_bias = float(self.rng.uniform(*sog_rng))
                    theta = float(self.rng.uniform(0.0, 2.0 * math.pi))
                    x += pos_bias * math.cos(theta)
                    y += pos_bias * math.sin(theta)
                    sog = max(0.0, sog + sog_bias)
                    break
            except Exception:
                continue
        return x, y, sog

    def _apply_gps_bias_drift(
        self,
        sid: ShipId,
        t_true: float,
        x: float,
        y: float,
        x_true0: float = None,
        y_true0: float = None,
    ):
        if not self.gps_bias_enable:
            return x, y

        bx, by = self._gps_bias_xy.get(sid, (0.0, 0.0))
        vx, vy = self._gps_drift_xy.get(sid, (0.0, 0.0))
        dt = float(t_true - self._gps_bias_t0)

        x_noisy = x + bx + vx * dt
        y_noisy = y + by + vy * dt

        if x_true0 is not None and y_true0 is not None:
            self._gps_dbg_rows.append(dict(
                t_true=float(t_true),
                tx_ship=int(sid),
                x_true=float(x_true0),
                y_true=float(y_true0),
                bx=float(bx), by=float(by),
                vx=float(vx), vy=float(vy),
                x_noisy=float(x_noisy),
                y_noisy=float(y_noisy),
                dx=float(x_noisy - x_true0),
                dy=float(y_noisy - y_true0),
            ))

        return x_noisy, y_noisy

    def _apply_field_noise(self, sid: ShipId, x: float, y: float, sog: float, cog: float):
        # 位置白噪声
        if self.noise_pos_m > 0.0:
            x += float(self.rng.normal(0.0, self.noise_pos_m))
            y += float(self.rng.normal(0.0, self.noise_pos_m))

        # SOG AR(1)
        v_prev = float(self._sog_ar1_state.get(sid, 0.0))
        rho_v = self.sog_ar1_rho
        sigma_v = self.noise_sog_mps
        if sigma_v > 0.0 and rho_v > 0.0:
            eps_v = float(self.rng.normal(0.0, sigma_v))
            v_new = rho_v * v_prev + math.sqrt(max(0.0, 1.0 - rho_v ** 2)) * eps_v
        else:
            v_new = 0.0
        self._sog_ar1_state[sid] = v_new
        sog = max(0.0, sog + v_new)

        # COG AR(1)
        h_prev = float(self._cog_ar1_state.get(sid, 0.0))
        rho_h = self.cog_ar1_rho
        sigma_h = self.noise_cog_rad
        if sigma_h > 0.0 and rho_h > 0.0:
            eps_h = float(self.rng.normal(0.0, sigma_h))
            h_new = rho_h * h_prev + math.sqrt(max(0.0, 1.0 - rho_h ** 2)) * eps_h
        else:
            h_new = 0.0
        self._cog_ar1_state[sid] = h_new
        cog = float(cog + h_new)

        # wrap
        try:
            from ..utils.math import wrap_to_pi
            cog = float(wrap_to_pi(cog))
        except Exception:
            cog = (cog + math.pi) % (2.0 * math.pi) - math.pi

        return x, y, sog, cog

    def _apply_field_errors(self, raw_dict: dict):
        if self.rng.random() < self.fe_sog_zero_stick_prob:
            raw_dict["sog"] = 0.0
        if "ship_type" in raw_dict and (self.rng.random() < self.fe_type_missing_prob):
            raw_dict["ship_type"] = None
        if "draft_m" in raw_dict and (self.rng.random() < self.fe_draft_missing_prob):
            raw_dict["draft_m"] = None
        return raw_dict

    def _maybe_conflict_mmsi(self, true_mmsi: int, tx_sid: ShipId) -> int:
        if self.fe_mmsi_conflict_prob <= 0.0:
            return true_mmsi
        if self.rng.random() >= self.fe_mmsi_conflict_prob:
            return true_mmsi
        cand = [m for sid, m in self.mmsi_of_ship.items() if sid != tx_sid]
        if not cand:
            return true_mmsi
        return int(self.rng.choice(cand))

    def _apply_clock(self, tx_ship: ShipId, tx_ts_true: float) -> float:
        if not self.clock_enable:
            return float(tx_ts_true)
        off, dppm = self.clock_of_ship.get(tx_ship, (0.0, 0.0))
        return float(tx_ts_true - (float(off) + float(dppm) * 1e-6 * float(tx_ts_true)))

    def sample_delay(self) -> float:

        # Use episode-scoped rng_delay if present (so reorder toggles do not perturb delay stream).
        rng = self.rng_delay if self.rng_delay is not None else self.rng
        d = float(rng.lognormal(mean=self.mu_ln, sigma=self.sigma_ln))

        d = max(0.0, min(d, self.max_delay))

        if self.reorder_enable and self.reorder_prob > 0.0:
            # IMPORTANT: use a separate RNG so reorder on/off doesn't shift other randomness
            if self.rng_reorder.random() < self.reorder_prob:
                d = max(0.0, d + max(0.0, float(self.reorder_extra_delay)))

        return d



    def _rx_list(self, tx_ship: ShipId) -> List[AgentId]:
        if tx_ship not in self.agent_of_ship:
            raise KeyError(f"tx_ship={tx_ship} not in agent_of_ship={list(self.agent_of_ship.keys())}")

        tx_agent = self.agent_of_ship[tx_ship]

        # 稳定顺序：按 ship_id 排序，避免多 worker 下日志顺序飘
        return [
            self.agent_of_ship[sid]
            for sid in sorted(self.agent_of_ship.keys())
            if self.agent_of_ship[sid] != tx_agent
        ]


    # ---------------- main step ----------------
    def step(self, t: Ts, true_states: Dict[ShipId, TrueState]) -> Dict[AgentId, List[RxMsg]]:
        """
        输入：当前仿真时刻 t、各 ship 的 TrueState（世界坐标系下的真值）
        输出：ready 字典（agent_id -> List[RxMsg]），为在本时刻到达并可被 observation 读取的报文

        坐标系约定：
          - TrueState.x/y: 世界坐标
          - RawTxMsg.x/y : 世界坐标 + 偏置/噪声
          - RxMsg.reported_x/y: 世界坐标 + 偏置/噪声（与 rx_agent 无关）
        """
        t = float(t)

        # ======= 关键修复：按时间推进所有链路的 GE 状态机一次 =======
        dt = t - float(self._last_tick_t)
        if dt < 0.0:
            dt = 0.0

        # Capture env step dt for stage3 (first positive dt wins)
        try:
            if float(getattr(self, "dt", 0.0)) <= 0.0 and dt > 1e-9:
                self.dt = float(dt)
        except Exception:
            pass

        self._last_tick_t = t

        for ch in self.channel_by_link.values():
            # 新版 GEChannel：tick(dt)
            if hasattr(ch, "tick"):
                try:
                    ch.tick(dt)
                except Exception:
                    pass
        # ===========================================================

        # 逐船判断是否需要发报
        for sid, st in true_states.items():
            if self.scheduler.should_tx(sid, t, st.sog):

                # ---- always compute rx_agents (do NOT hide behind debug flags) ----
                tx_agent = self.agent_of_ship[sid]
                rx_agents = self._rx_list(sid)

                # self-loop：分发层面 100% 不允许
                if tx_agent in rx_agents:
                    raise RuntimeError(f"[AIS][SELF-LOOP] tx_agent={tx_agent} in rx_agents={rx_agents} at t={t:.2f}")

                if os.environ.get("AIS_DBG_TX", "0") == "1":
                    print(f"[AIS-DBG-TX] t={t:.2f} tx_ship={int(sid)} tx_agent={tx_agent} -> rx_agents={rx_agents}", flush=True)

                # ---- DEBUG: AIS 发送决策 ----
                if AIS_DBG_TX and (AIS_DBG_FOCUS_TX in ("", str(int(sid)))):
                    print(
                        f"[AIS-DBG-TX] t={t:.2f} tx_sid={int(sid)} sog={float(st.sog):.3f} "
                        f"-> rx_agents={rx_agents}",
                        flush=True,
                    )

                # ---- HARD ASSERT: 分发列表不应包含 self ----
                if AIS_ASSERT_SELFLOOP:
                    tx_agent_id = str(self.agent_of_ship[sid])
                    if tx_agent_id in rx_agents:
                        raise RuntimeError(
                            f"[AISDBG][SELF-LOOP-RXLIST] {tx_agent_id} appears in its own rx list at t={t:.2f}! "
                            f"rx_agents={rx_agents}"
                        )

                x_true_w, y_true_w = float(st.x), float(st.y)
                
                # [FIX START] 强制基于 vx, vy 计算 SOG/COG，防止 st.cog 属性读取失败或为 0
                vx_val = float(getattr(st, "vx", 0.0))
                vy_val = float(getattr(st, "vy", 0.0))
                
                sog_true = (vx_val**2 + vy_val**2)**0.5
                if sog_true > 1e-3:
                    cog_true = math.atan2(vy_val, vx_val) # 返回 (-pi, pi]
                else:
                    # 静止时尝试读取 heading/psi，否则保持 0
                    cog_true = float(getattr(st, "heading", getattr(st, "psi", getattr(st, "yaw", 0.0))))
                
                # >>> DEBUG PROBE 1: 发送端 <<<
                #if sid == 1: # 或者是你关注的任何 ship_id
                    #print(f"[PROBE-TX] sid={sid} vx={vx_val:.2f} vy={vy_val:.2f} "
                          #f"atan2(rad)={math.atan2(vy_val, vx_val):.4f} "
                          #f"cog_true(to_msg)={cog_true:.4f}")

                x_w, y_w = x_true_w, y_true_w
                sog_w, cog_w = sog_true, cog_true

                # 1) 区域 bias
                x_w, y_w, sog_w = self._apply_regional_bias(x_w, y_w, sog_w)

                # 2) GPS bias+drift
                x_w, y_w = self._apply_gps_bias_drift(
                    sid, t, x_w, y_w,
                    x_true0=x_true_w, y_true0=y_true_w,
                )

                # 3) 白噪声 + AR(1)
                x_w, y_w, sog_w, cog_w = self._apply_field_noise(sid, x_w, y_w, sog_w, cog_w)

                # AR(1) 样本记录
                try:
                    v_new = float(self._sog_ar1_state.get(sid, 0.0))
                    h_new = float(self._cog_ar1_state.get(sid, 0.0))
                    self._ar1_samples.append(dict(
                        t_true=float(t),
                        tx_ship=int(sid),
                        sog_true=float(sog_true),
                        cog_true=float(cog_true),
                        sog_noise=float(v_new),
                        cog_noise=float(h_new),
                        sog_noisy=float(sog_w),
                        cog_noisy=float(cog_w),
                    ))
                except Exception:
                    pass

                # 4) Compute ROT (rate of turn) and nav_status
                # Try to get rot from TrueState first, otherwise compute from yaw difference
                rot_true = float(getattr(st, "rot", 0.0))
                if rot_true == 0.0 and sid in self._prev_yaw and sid in self._prev_t:
                    # Compute rot from yaw difference: omega = d_yaw / dt
                    prev_yaw = self._prev_yaw[sid]
                    prev_t = self._prev_t[sid]
                    dt_yaw = t - prev_t
                    if dt_yaw > 1e-6:
                        # Handle angle wrap-around (-pi, pi]
                        d_yaw = cog_true - prev_yaw
                        if d_yaw > math.pi:
                            d_yaw -= 2 * math.pi
                        elif d_yaw < -math.pi:
                            d_yaw += 2 * math.pi
                        rot_true = d_yaw / dt_yaw

                # Update previous yaw tracking
                self._prev_yaw[sid] = cog_true
                self._prev_t[sid] = t

                # Add noise to rot
                rot_w = rot_true
                if self.noise_rot_rads > 0.0:
                    rot_w += float(self.rng.normal(0.0, self.noise_rot_rads))

                # Get nav_status from TrueState (default 0 = underway using engine)
                nav_status = int(getattr(st, "nav_status", 0))

                # 5) MMSI 与可选冲突
                true_mmsi = self.mmsi_of_ship.get(sid, 999000000 + int(sid))
                report_mmsi = self._maybe_conflict_mmsi(true_mmsi, sid)
                # 1. 生成唯一 ID
                unique_id = str(uuid.uuid4())[:8]  # 取前8位通常足够且易读
                # 6) RawTx
                raw = RawTxMsg(
                    msg_id=unique_id,
                    tx_ship=sid,
                    mmsi=report_mmsi,
                    tx_ts_true=float(t),
                    x=x_w, y=y_w,
                    sog=sog_w, cog=cog_w,
                    rot=rot_w,
                    nav_status=nav_status,
                )

                if os.environ.get("AIS_TRACE_TX_BIAS", "0") == "1":
                    dx0 = float(raw.x) - float(x_true_w)
                    dy0 = float(raw.y) - float(y_true_w)
                    print(
                        f"[TxBIAS] t={t:.2f} sid={sid} mmsi={report_mmsi} "
                        f"true=({x_true_w:.2f},{y_true_w:.2f}) raw=({raw.x:.2f},{raw.y:.2f}) "
                        f"d=({dx0:.2f},{dy0:.2f})"
                    )

                # 6) 字段错误
                raw_d = raw.__dict__.copy()
                raw_d = self._apply_field_errors(raw_d)
                raw = RawTxMsg(**raw_d)

                # 7) 欺诈层（仅 attacker 生效）
                apply_fraud = (
                    self.fraud_wrapper is not None
                    and self.fraud_wrapper.cfg.enable
                    and (self.attacker_ships is None or sid in self.attacker_ships)
                )
                if self.fraud_debug:
                    print(f"[FraudGate] t={t:.1f} sid={int(sid)} apply={apply_fraud} attackers={self.attacker_ships}")

                if apply_fraud:
                    forged = self.fraud_wrapper.apply(sid, raw)
                    if forged is None:
                        if self.fraud_debug:
                            print(f"[FraudDrop] t={t:.1f} sid={int(sid)} (SILENT / dropped before GE)")
                        continue
                    raw = forged

                # 8) 逐接收端链路：GE + delay + queue
                for rx in self._rx_list(sid):
                    self._link_attempts += 1

                    link_key = (int(sid), str(rx))
                    ch = self._get_link_channel(link_key)

                    # ======= 计算发送端与接收端的距离（用于距离衰减） =======
                    distance_m = None
                    if self.dist_enable:
                        # 找到接收端 agent 对应的 ship_id
                        rx_sid = self._ship_of_agent.get(rx, None)
                        if rx_sid is not None and rx_sid in true_states:
                            rx_st = true_states[rx_sid]
                            # 计算距离 (tx 是 sid, rx 是 rx_sid)
                            dx = x_true_w - float(rx_st.x)
                            dy = y_true_w - float(rx_st.y)
                            distance_m = math.sqrt(dx * dx + dy * dy)
                    # ==================================================

                    # ======= 关键修复：本步只"判定"，不推进状态 =======
                    passed = None
                    if hasattr(ch, "pass_now"):
                        try:
                            passed = bool(ch.pass_now(distance_m=distance_m))
                        except Exception:
                            passed = None
                    if passed is None:
                        # 兼容兜底：旧版 GEChannel 只有 step_pass()
                        try:
                            passed = bool(ch.step_pass())
                        except Exception:
                            passed = True
                    # ==================================================

                    if not passed:
                        self._link_dropped += 1
                        continue
                    self._link_passed += 1

                    net_delay = self.sample_delay()
                    at = float(t + net_delay)

                    # 乱序检测
                    last_at = self._last_arrival_by_link.get(link_key, None)
                    is_reordered = bool(last_at is not None and at < last_at - 1e-6)
                    self._last_arrival_by_link[link_key] = at if last_at is None else max(last_at, at)

                    self._reorder_total += 1
                    if is_reordered:
                        self._reorder_hit += 1

                    if self.cfg.get("logging", {}).get("export_reorder_samples", False):
                        self._reorder_samples.append(dict(
                            t_true=float(raw.tx_ts_true),
                            tx_ship=int(sid),
                            rx_agent=str(rx),
                            delay=float(net_delay),
                            arrival_time=float(at),
                            last_arrival_prev=(float(last_at) if last_at is not None else None),
                            is_reordered=bool(is_reordered),
                        ))

                    reported_ts = self._apply_clock(sid, raw.tx_ts_true)
                    age = max(0.0, at - reported_ts)

                    msg = RxMsg(
                        msg_id=raw.msg_id,
                        rx_agent=rx,
                        mmsi=raw.mmsi,
                        reported_x=raw.x,
                        reported_y=raw.y,
                        reported_sog=raw.sog,
                        reported_cog=raw.cog,
                        reported_rot=raw.rot,
                        reported_nav_status=raw.nav_status,
                        reported_ts=reported_ts,
                        arrival_time=at,
                        age=age,
                    )

                    # stats
                    self.delay_samples.append(net_delay)
                    self.age_samples.append(age)
                    amd = float(age - net_delay)
                    self.age_minus_delay_samples.append(amd)

                    # Per-ship tracking (for diagnostic analysis)
                    tx_sid = int(sid)
                    if tx_sid not in self._per_ship_delay:
                        self._per_ship_delay[tx_sid] = []
                        self._per_ship_age[tx_sid] = []
                        self._per_ship_rx_count[tx_sid] = 0
                    self._per_ship_delay[tx_sid].append(net_delay)
                    self._per_ship_age[tx_sid].append(age)
                    self._per_ship_rx_count[tx_sid] += 1
                    # Last RX info for this ship (useful for PF error attribution)
                    self._last_rx_by_ship[tx_sid] = {
                        "last_rx_delay_s": float(net_delay),
                        "last_rx_age_s": float(age),
                        "last_rx_report_ts": float(reported_ts),
                        "last_rx_arrival_ts": float(at),
                    }

                    if self.cfg.get("logging", {}).get("export_link_samples", False):
                        off, dppm = self.clock_of_ship.get(sid, (0.0, 0.0))
                        self._link_samples.append(dict(
                            t_true=float(raw.tx_ts_true),
                            tx_ship=int(sid),
                            rx_agent=str(rx),
                            delay=float(net_delay),
                            reported_ts=float(reported_ts),
                            arrival_time=float(at),
                            age=float(age),
                            age_minus_delay=float(amd),
                            offset_s=float(off),
                            drift_ppm=float(dppm),
                            expected_offset_at_t=float(off + dppm * 1e-6 * float(raw.tx_ts_true)),
                            is_reordered=bool(is_reordered),
                        ))


                    if AIS_DBG_LINK and (AIS_DBG_FOCUS_TX in ("", str(int(sid)))) and (AIS_DBG_FOCUS_RX in ("", str(rx))):
                        print(
                            f"[AIS-DBG-LINK] t={t:.2f} tx={int(sid)} -> rx={rx} PASS "
                            f"delay={net_delay:.3f} at={at:.2f} reported_ts={reported_ts:.2f} age={age:.2f} "
                            f"mmsi={int(raw.mmsi)} reorder={is_reordered}",
                            flush=True,
                        )


                    # push
                    self.arrivals.push(at, str(rx), msg, link_key=link_key)

        # pop
        ready = self.arrivals.pop_ready(t)

        # ✅ 关键：ready 必须覆盖所有 agent，哪怕为空，否则 track lifecycle 会断
        # Stable order for determinism
        all_agents = [self.agent_of_ship[sid] for sid in sorted(self.agent_of_ship.keys())]

        ready_full = {a: ready.get(a, []) for a in all_agents}

        if os.environ.get("AIS_DBG_RX", "0") == "1":
            n_total = sum(len(v) for v in ready_full.values())
            parts = " | ".join([f"{a}:{len(ready_full[a])}" for a in all_agents])
            print(f"[AIS-DBG-RX] t={t:.2f} total={n_total} {parts}", flush=True)


        self._link_delivered += sum(len(v) for v in ready.values())

        if os.environ.get("AIS_TRACE_PER_LINK", "0") == "1":
            d = getattr(self.arrivals, "delivered_by_link", {})
            p = getattr(self.arrivals, "popped_by_link", {})
            q = getattr(self.arrivals, "dropped_ttl_by_link", {})
            keys = set(d.keys()) | set(p.keys()) | set(q.keys())
            if keys:
                parts = []
                for k in sorted(keys):
                    parts.append(f"{k}:delv={d.get(k,0)},pop={p.get(k,0)},ttl={q.get(k,0)}")
                print(f"[LinkDBG] t={t:.2f} " + " | ".join(parts))


        self.record_timeseries(t)

        # ---- DEBUG: AIS 接收汇总（每个 rx_agent）----
        if AIS_DBG_RX:
            for rx_agent, msgs in ready.items():
                if AIS_DBG_FOCUS_RX not in ("", rx_agent):
                    continue
                summary = [
                    (
                        int(getattr(m, "mmsi", -1)),
                        round(float(getattr(m, "reported_ts", 0.0)), 2),
                        round(float(getattr(m, "arrival_time", 0.0)), 2),
                        round(float(getattr(m, "age", 0.0)), 2),
                    )
                    for m in msgs
                ]
                print(f"[AIS-DBG-RX] t={t:.2f} rx_agent={rx_agent} n={len(msgs)} msgs={summary}", flush=True)

                # 第二道“自收自发”证据：仅在未启用 mmsi_conflict 时建议打开
                if AIS_ASSERT_SELFLOOP_MSG and rx_agent.startswith("ship_"):
                    rx_sid = int(rx_agent.split("_")[1])
                    self_mmsi = self.mmsi_of_ship.get(rx_sid, None)
                    if self_mmsi is not None:
                        for m in msgs:
                            if int(getattr(m, "mmsi", -999)) == int(self_mmsi):
                                raise RuntimeError(
                                    f"[AISDBG][SELF-LOOP-MSG] rx_agent={rx_agent} received msg with its own mmsi={int(self_mmsi)} "
                                    f"at t={t:.2f}. (Disable AIS_ASSERT_SELFLOOP_MSG if mmsi_conflict enabled.)"
                                )


        return ready_full

    # -------- fraud control --------
    def set_fraud_enable(self, enable: bool) -> None:
        if self.fraud_wrapper is not None:
            self.fraud_wrapper.cfg.enable = bool(enable)

    def set_attackers(self, attackers: Optional[list[ShipId] | set[ShipId]]) -> None:
        if attackers is None:
            self.attacker_ships = None
        else:
            self.attacker_ships = set(int(s) for s in attackers)

        if self.fraud_debug:
            print(f"[FraudSetAttackers] attackers={self.attacker_ships}")
