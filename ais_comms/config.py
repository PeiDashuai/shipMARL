# ais_comms/config.py
from __future__ import annotations
import pprint

import os
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple, List
import numpy as np
import dataclasses as dc


try:
    import yaml
except Exception:
    yaml = None



@dataclass
class GEParams:
    alpha: float; beta: float
    p_good_pass: float; p_bad_pass: float

@dataclass
class TxRateProfile:
    # 基线上报周期随航态变化（示意）
    at_anchor_s: Tuple[float,float]   # (min,max)
    low_speed_s: Tuple[float,float]
    cruise_s:    Tuple[float,float]
    high_turn_s: Tuple[float,float]
    jitter_eps:  float                # U(1-ε, 1+ε)

@dataclass
class ErrorParams:
    pos_sigma_hf: float
    sog_sigma_hf: float; cog_sigma_hf_deg: float
    bias_rho: float; bias_sigma_step: float
    spike_prob: float; spike_range_m: Tuple[float,float]

@dataclass
class ClockParams:
    sigma_delta0: float
    sigma_drift: float

@dataclass
class RSSIQualParams:
    s0: float; fading_sigma: float
    age_tau: float
    w_rssi: float; w_age: float; w_field: float

@dataclass
class FieldErrorParams:
    draft_miss_p: float; type_err_p: float
    length_miss_p: float; beam_miss_p: float
    mmsi_charerr_p: float; id_swap_p: float

@dataclass
class LimitParams:
    max_delay: float
    max_step_offset_m: float

@dataclass
class AISConfig:
    ge: GEParams
    tx_rate: TxRateProfile
    err: ErrorParams
    clk: ClockParams
    rssi: RSSIQualParams
    field: FieldErrorParams
    limits: LimitParams
    # 其他：region_map_path, seeds, profiles 等


@dc.dataclass
class EpisodeParams:
    # —— Channel / GE ——
    ge_p_g2b: float
    ge_p_b2g: float
    ge_drop_bad: float
    burst_enable: bool
    burst_prob: float
    burst_dur_s: float
    burst_extra_drop: float
    # —— Delay ——
    delay_mu: float
    delay_sigma: float
    delay_clip: float
    # —— Clock ——
    clock_enable: bool
    clock_offset_s: float
    clock_drift_ppm: float
    # —— Scheduler ——
    # —— Scheduler —— 说明：sch_breaks 的单位统一为 m/s；sch_periods 为秒
    sch_base_period_s: float
    sch_jitter_frac: float
    sch_breaks: list          # m/s
    sch_periods: list         # seconds
    sch_drop_prob_on_idle: float
    sch_idle_thr_mps: float

    # —— Queue ——
    q_ttl_s: float
    q_max_inflight: int
    # —— Obs ——
    obs_slot_K: int
    obs_slot_ttl_s: float
    obs_age_cap_s: float
    obs_miss_mask: bool
    # —— Noise ——
    noise_pos_m: float
    noise_sog_mps: float
    noise_cog_deg: float
    reg_bias_enable: bool
    reg_bias_rects: list
    # —— Field errors ——
    fe_mmsi_conflict_prob: float
    fe_type_missing_prob: float
    fe_draft_missing_prob: float
    fe_sog_zero_stick_prob: float

    # —— Reorder —— (packet reordering via extra delay)
    reorder_enable: bool
    reorder_prob: float
    reorder_extra_delay_s: float   # 建议采样成单值（每局固定），而不是 lo/hi

def _pick(val, rng: np.random.Generator):
    """标量直接返回；[a,b] 采 U[a,b]；bool保持；list保持。"""
    if isinstance(val, (int, float, bool, list)):
        return val
    if isinstance(val, tuple):
        a, b = val; return float(rng.uniform(a, b))
    if isinstance(val, (set, dict)):
        return val
    return val

def _range_to_tuple(x):
    # yaml 里可能是 list，变 tuple 便于判别
    if isinstance(x, list) and len(x) == 2 and all(isinstance(v, (int,float)) for v in x):
        return (float(x[0]), float(x[1]))
    return x

def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml not available. pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_episode_params(cfg: Dict[str, Any], seed: int) -> EpisodeParams:
    rng = np.random.default_rng(seed)

    # ------------------ 工具函数 ------------------
    def _to_float_pair(v) -> Tuple[float, float]:
        """把标量或 [lo, hi] 统一成 (lo, hi) 的 float 区间。"""
        if isinstance(v, (list, tuple)) and len(v) == 2:
            lo, hi = v
        else:
            lo = hi = v
        return float(lo), float(hi)

    def _to_int_pair(v) -> Tuple[int, int]:
        """把标量或 [lo, hi] 统一成 (lo, hi) 的 int 区间。"""
        if isinstance(v, (list, tuple)) and len(v) == 2:
            lo, hi = v
        else:
            lo = hi = v
        return int(lo), int(hi)

    def _sample_float(v, default_val: float) -> float:
        """支持标量或区间；缺省用 default_val。"""
        if v is None:
            return float(default_val)
        lo, hi = _to_float_pair(v)
        return float(rng.uniform(lo, hi)) if lo != hi else float(lo)

    def _sample_int(v, default_val: int) -> int:
        """支持标量或区间；缺省用 default_val（闭区间采样）。"""
        if v is None:
            return int(default_val)
        lo, hi = _to_int_pair(v)
        if lo == hi:
            return int(lo)
        # numpy 的 integers 上界是开区间，所以 +1
        return int(rng.integers(lo, hi + 1))

    def _sample_prob_bool(v, default_prob: float = 1.0) -> bool:
        """
        drop_bad 等可以写成布尔或 [lo, hi] 概率：
          - bool 直接返回；
          - [lo, hi] 表示从区间中采一个 p，然后做一次伯努利试验；
          - 标量（0~1）也按概率处理。
        """
        if isinstance(v, bool):
            return v
        if isinstance(v, (list, tuple)):
            lo, hi = _to_float_pair(v)
            p = float(rng.uniform(lo, hi)) if lo != hi else float(lo)
            return bool(rng.random() < p)
        try:
            p = float(v)
            return bool(rng.random() < p)
        except Exception:
            return bool(rng.random() < default_prob)

    # ------------------ Reorder ------------------
    ro = cfg.get("reorder", {}) or {}
    reorder_enable = bool(ro.get("enable", False))
    reorder_prob   = _sample_float(ro.get("prob", 0.0), 0.0)

    # extra_delay_s: [lo,hi] -> sample one value for this episode
    ex = ro.get("extra_delay_s", [0.0, 0.0])
    reorder_extra_delay_s = _sample_float(ex, 0.0)


    # ------------------ 读取各分组配置 ------------------
    ch      = cfg.get("channel", {}) or {}
    ge_cfg  = ch.get("ge", {}) or {}
    bs_cfg  = ch.get("burst", {}) or {}

    dl      = cfg.get("delay", {}) or {}
    ck      = cfg.get("clock", {}) or {}

    sch     = cfg.get("scheduler", {}) or {}
    q       = cfg.get("queue", {}) or {}
    ob      = cfg.get("obs", {}) or {}

    nz      = cfg.get("noise", {}) or {}
    rb      = (nz.get("regional_bias", {}) or {})

    fe      = cfg.get("field_errors", {}) or {}

    # ------------------ Channel / GE ------------------
    ge_p_g2b        = _sample_float(ge_cfg.get("p_g2b", 0.01), 0.01)
    ge_p_b2g        = _sample_float(ge_cfg.get("p_b2g", 0.20), 0.20)
    ge_drop_bad     = _sample_prob_bool(ge_cfg.get("drop_bad", True), default_prob=1.0)

    burst_enable    = bool(bs_cfg.get("enable", False))
    burst_prob      = _sample_float(bs_cfg.get("prob", 0.0), 0.0)
    burst_dur_s     = _sample_float(bs_cfg.get("dur_s", 0.0), 0.0)
    burst_extra_drop= _sample_float(bs_cfg.get("extra_drop", 0.0), 0.0)

    # ------------------ Delay ------------------
    delay_mu        = _sample_float(dl.get("lognormal_mu", 1.2), 1.2)
    delay_sigma     = _sample_float(dl.get("lognormal_sigma", 0.35), 0.35)
    delay_clip      = _sample_float(dl.get("clip_max", 5.0), 5.0)

    # ------------------ Clock ------------------
    clock_enable    = bool(ck.get("enable", False))
    clock_offset_s  = _sample_float(ck.get("offset_s", 0.0), 0.0)
    clock_drift_ppm = _sample_float(ck.get("drift_ppm", 0.0), 0.0)

    # ------------------ Scheduler ------------------
    sch_base_period_s = _sample_float(sch.get("base_period_s", 2.5), 2.5)
    sch_jitter_frac   = _sample_float(sch.get("jitter_frac", 0.1), 0.1)

    # 全面统一为 m/s：只读 sog_mps_breaks（不再支持/兜底 knots）
    sch_breaks  = list(sch.get("sog_mps_breaks", []))                     # m/s
    sch_periods = list(sch.get("sog_period_s",    [10.0, 6.0, 3.0, 2.0])) # s
    if not sch_breaks:
        sch_breaks = [0.00, 1.03, 2.57, 7.20, 11.83]

    sch_drop_prob_on_idle = _sample_float(sch.get("drop_prob_on_idle", 0.0), 0.0)
    sch_idle_thr_mps      = _sample_float(sch.get("idle_thr_mps", 0.0), 0.0)



    # ------------------ Queue ------------------
    q_ttl_s        = _sample_float(q.get("ttl_s", [20.0, 40.0]), 30.0)
    q_max_inflight = _sample_int(q.get("max_inflight", [512, 2048]), 1024)

    # ------------------ Obs ------------------
    obs_slot_K       = int(ob.get("slot_K", 16))
    obs_slot_ttl_s   = _sample_float(ob.get("slot_ttl_s", [30.0, 60.0]), 45.0)
    obs_age_cap_s    = _sample_float(ob.get("age_cap_s",  [3.0, 6.0]), 4.0)
    obs_miss_mask    = bool(ob.get("miss_mask", True))

    # ------------------ Noise / Regional bias ------------------
    # NOTE: 这些默认值是根据 calib_fvessel_ais.py 的真实统计反推的“目标 std”，
    # 若 yaml 里没有专门覆盖，就使用这里的范围。

    # 位置噪声：略小于之前的 15–25 m，避免 e_pos 均值偏大
    noise_pos_m      = _sample_float(nz.get("position_m", [8.0, 18.0]), 12.0)

    # SOG 噪声（AR(1) 稳态 std ≈ noise_sog_mps）：
    # 目标是让 e_sog 的 std 接近真实的 1.56 m/s
    noise_sog_mps    = _sample_float(nz.get("sog_mps", [1.0, 2.0]), 1.5)

    # COG 噪声（度）：真实数据 std ≈ 0.66 rad ≈ 38°
    # 给一个 [20, 40]° 的范围，缺省 30°，经 AR(1) 平滑后能接近真实的 COG 抖动
    noise_cog_deg = _sample_float(nz.get("cog_deg", [25.0, 45.0]), 35.0)

    reg_bias_enable  = bool(rb.get("enable", False))
    reg_bias_rects   = list(rb.get("rects", []))

    # ------------------ Field errors ------------------
    fe_mmsi_conflict_prob  = _sample_float(fe.get("mmsi_conflict_prob", 0.0), 0.0)
    fe_type_missing_prob   = _sample_float(fe.get("type_missing_prob",  0.0), 0.0)
    fe_draft_missing_prob  = _sample_float(fe.get("draft_missing_prob", 0.0), 0.0)
    fe_sog_zero_stick_prob = _sample_float(fe.get("sog_zero_stick_prob",0.0), 0.0)

    # ------------------ 打包输出 ------------------
    # ------------------ Debug: print sampled episode params ------------------
    log_cfg = cfg.get("logging", {}) or {}
    print_params = bool(log_cfg.get("print_params", False)) or (os.environ.get("AIS_PRINT_EP_PARAMS", "0") == "1")
    export_ep = bool(log_cfg.get("export_episode_params", False))
    export_path = str(log_cfg.get("episode_params_path", "./ais_episode_params.json"))

    if print_params:
        # 1) 先把最关键、最容易出错的项单独强调打印
#        print("\n[AIS][EP] ===== sampled episode params =====")
#        print(f"[AIS][EP] seed={seed}")
#        print(f"[AIS][EP] ge_p_g2b={ge_p_g2b:.6f} ge_p_b2g={ge_p_b2g:.6f} ge_drop_bad={ge_drop_bad} (type={type(ge_drop_bad).__name__})")
#        print(f"[AIS][EP] burst_enable={burst_enable} prob={burst_prob:.6f} dur_s={burst_dur_s:.3f} extra_drop={burst_extra_drop:.6f}")
#        print(f"[AIS][EP] delay_mu={delay_mu:.6f} delay_sigma={delay_sigma:.6f} delay_clip={delay_clip:.3f}")
#        print(f"[AIS][EP] clock_enable={clock_enable} offset_s={clock_offset_s:.3f} drift_ppm={clock_drift_ppm:.3f}")
#        print(f"[AIS][EP] sch_base_period_s={sch_base_period_s:.3f} sch_jitter_frac={sch_jitter_frac:.3f}")
#        print(f"[AIS][EP] sch_breaks(m/s)={sch_breaks}")
#        print(f"[AIS][EP] sch_periods(s)={sch_periods}")
#        print(f"[AIS][EP] q_ttl_s={q_ttl_s:.3f} q_max_inflight={q_max_inflight}")
#        print(f"[AIS][EP] obs_slot_K={obs_slot_K} obs_slot_ttl_s={obs_slot_ttl_s:.3f} obs_age_cap_s={obs_age_cap_s:.3f} obs_miss_mask={obs_miss_mask}")
#        print(f"[AIS][EP] noise_pos_m={noise_pos_m:.3f} noise_sog_mps={noise_sog_mps:.3f} noise_cog_deg={noise_cog_deg:.3f}")
#        print(f"[AIS][EP] reg_bias_enable={reg_bias_enable} reg_bias_rects={reg_bias_rects}")
#        print(f"[AIS][EP] field_err: mmsi_conflict={fe_mmsi_conflict_prob:.6f} type_missing={fe_type_missing_prob:.6f} draft_missing={fe_draft_missing_prob:.6f} sog_zero_stick={fe_sog_zero_stick_prob:.6f}")

        # 2) 再打印一个“可复制粘贴到论文/日志”的扁平 dict（排序输出）
        ep_dict = {
            "seed": seed,
            "ge_p_g2b": float(ge_p_g2b),
            "ge_p_b2g": float(ge_p_b2g),
            "ge_drop_bad": bool(ge_drop_bad),

            "burst_enable": bool(burst_enable),
            "burst_prob": float(burst_prob),
            "burst_dur_s": float(burst_dur_s),
            "burst_extra_drop": float(burst_extra_drop),

            "delay_mu": float(delay_mu),
            "delay_sigma": float(delay_sigma),
            "delay_clip": float(delay_clip),

            "clock_enable": bool(clock_enable),
            "clock_offset_s": float(clock_offset_s),
            "clock_drift_ppm": float(clock_drift_ppm),

            "sch_base_period_s": float(sch_base_period_s),
            "sch_jitter_frac": float(sch_jitter_frac),
            "sch_breaks": list(sch_breaks),
            "sch_periods": list(sch_periods),
            "sch_drop_prob_on_idle": float(sch_drop_prob_on_idle),
            "sch_idle_thr_mps": float(sch_idle_thr_mps),

            "q_ttl_s": float(q_ttl_s),
            "q_max_inflight": int(q_max_inflight),

            "obs_slot_K": int(obs_slot_K),
            "obs_slot_ttl_s": float(obs_slot_ttl_s),
            "obs_age_cap_s": float(obs_age_cap_s),
            "obs_miss_mask": bool(obs_miss_mask),

            "noise_pos_m": float(noise_pos_m),
            "noise_sog_mps": float(noise_sog_mps),
            "noise_cog_deg": float(noise_cog_deg),
            "reg_bias_enable": bool(reg_bias_enable),
            "reg_bias_rects": list(reg_bias_rects),

            "fe_mmsi_conflict_prob": float(fe_mmsi_conflict_prob),
            "fe_type_missing_prob": float(fe_type_missing_prob),
            "fe_draft_missing_prob": float(fe_draft_missing_prob),
            "fe_sog_zero_stick_prob": float(fe_sog_zero_stick_prob),
        }

#        print("[AIS][EP] --- ep_dict (sorted) ---")
#        for k in sorted(ep_dict.keys()):
#            print(f"[AIS][EP] {k} = {ep_dict[k]}")
#        print("[AIS][EP] ================================\n")

    # 可选：导出 json（写到 logging.episode_params_path 或默认 ./ais_episode_params.json）
#    if export_ep:
#        try:
#            with open(export_path, "w", encoding="utf-8") as f:
#                json.dump(ep_dict if 'ep_dict' in locals() else dump_episode_params(
#                    EpisodeParams(
#                        ge_p_g2b=ge_p_g2b, ge_p_b2g=ge_p_b2g, ge_drop_bad=ge_drop_bad,
#                        burst_enable=burst_enable, burst_prob=burst_prob, burst_dur_s=burst_dur_s, burst_extra_drop=burst_extra_drop,
#                        delay_mu=delay_mu, delay_sigma=delay_sigma, delay_clip=delay_clip,
#                        clock_enable=clock_enable, clock_offset_s=clock_offset_s, clock_drift_ppm=clock_drift_ppm,
#                        sch_base_period_s=sch_base_period_s, sch_jitter_frac=sch_jitter_frac,
#                        sch_breaks=sch_breaks, sch_periods=sch_periods,
#                        sch_drop_prob_on_idle=sch_drop_prob_on_idle, sch_idle_thr_mps=sch_idle_thr_mps,
#                        q_ttl_s=q_ttl_s, q_max_inflight=q_max_inflight,
#                        obs_slot_K=obs_slot_K, obs_slot_ttl_s=obs_slot_ttl_s, obs_age_cap_s=obs_age_cap_s, obs_miss_mask=obs_miss_mask,
#                        noise_pos_m=noise_pos_m, noise_sog_mps=noise_sog_mps, noise_cog_deg=noise_cog_deg,
#                        reg_bias_enable=reg_bias_enable, reg_bias_rects=reg_bias_rects,
#                        fe_mmsi_conflict_prob=fe_mmsi_conflict_prob, fe_type_missing_prob=fe_type_missing_prob,
#                        fe_draft_missing_prob=fe_draft_missing_prob, fe_sog_zero_stick_prob=fe_sog_zero_stick_prob,
#                    )
#                ), f, indent=2, ensure_ascii=False)
#            if print_params:
#               print(f"[AIS][EP] exported episode params -> {export_path}")
#        except Exception as e:
#            print(f"[AIS][EP][WARN] export_episode_params failed: {e}")

    # ------------------ 打包输出 ------------------
    return EpisodeParams(
        # GE / burst
        ge_p_g2b=ge_p_g2b,
        ge_p_b2g=ge_p_b2g,
        ge_drop_bad=ge_drop_bad,
        burst_enable=burst_enable,
        burst_prob=burst_prob,
        burst_dur_s=burst_dur_s,
        burst_extra_drop=burst_extra_drop,

        # delay
        delay_mu=delay_mu,
        delay_sigma=delay_sigma,
        delay_clip=delay_clip,

        # clock
        clock_enable=clock_enable,
        clock_offset_s=clock_offset_s,
        clock_drift_ppm=clock_drift_ppm,

        # scheduler
        sch_base_period_s=sch_base_period_s,
        sch_jitter_frac=sch_jitter_frac,
        sch_breaks=sch_breaks,
        sch_periods=sch_periods,
        sch_drop_prob_on_idle=sch_drop_prob_on_idle,
        sch_idle_thr_mps=sch_idle_thr_mps,

        # queue
        q_ttl_s=q_ttl_s,
        q_max_inflight=q_max_inflight,

        # obs
        obs_slot_K=obs_slot_K,
        obs_slot_ttl_s=obs_slot_ttl_s,
        obs_age_cap_s=obs_age_cap_s,
        obs_miss_mask=obs_miss_mask,

        # noise & regional bias
        noise_pos_m=noise_pos_m,
        noise_sog_mps=noise_sog_mps,
        noise_cog_deg=noise_cog_deg,
        reg_bias_enable=reg_bias_enable,
        reg_bias_rects=reg_bias_rects,

        # field errors
        fe_mmsi_conflict_prob=fe_mmsi_conflict_prob,
        fe_type_missing_prob=fe_type_missing_prob,
        fe_draft_missing_prob=fe_draft_missing_prob,
        fe_sog_zero_stick_prob=fe_sog_zero_stick_prob,

        # reorder
        reorder_enable=reorder_enable,
        reorder_prob=reorder_prob,
        reorder_extra_delay_s=reorder_extra_delay_s,

    )



def dump_episode_params(ep: EpisodeParams) -> Dict[str, Any]:
    d = dc.asdict(ep)
    return d

def export_params_json(path: str, ep: EpisodeParams):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dump_episode_params(ep), f, indent=2, ensure_ascii=False)

