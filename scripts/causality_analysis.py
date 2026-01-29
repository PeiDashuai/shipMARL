#!/usr/bin/env python3
"""
causality_analysis.py — Analyze AIS → PF → RL causality chain

This script analyzes the relationship between:
  1. AIS communication quality → PF tracking accuracy
  2. PF tracking accuracy → RL collision avoidance performance

Metrics analyzed:
  - Communication: PPR (packet pass rate), PDR (packet delivery rate),
                   delay_mean, delay_p95, age_mean, age_p95
  - PF Tracking: pos_error (m), vel_error (m/s), heading_error (rad),
                 track_age (s), eff_particles
  - RL Performance: r_total, c_near, c_coll, collision_count, arrival_rate

Usage:
    python scripts/causality_analysis.py --input TEST_STAGING --mode train

Output:
    - Correlation matrices (comm→PF, PF→RL)
    - Time-series plots of key metrics
    - Statistical summary tables
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_stage3_data(stage3_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load all stage3 JSONL data, grouped by episode_uid."""
    episodes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for fp in sorted(stage3_dir.glob("episodes.*.jsonl")):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    rec_type = rec.get("type")
                    if rec_type == "shard_header":
                        continue
                    uid = rec.get("episode_uid")
                    if uid:
                        episodes[uid].append(rec)
                except Exception:
                    continue

    return dict(episodes)


def extract_episode_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract metrics from a single episode's records."""
    comm_metrics = []
    pf_metrics = []
    rl_metrics = []

    episode_info = {}
    episode_end = {}

    for rec in records:
        rec_type = rec.get("type")

        if rec_type in ("episode_init", "episode"):
            episode_info = rec

        elif rec_type == "episode_end":
            episode_end = rec

        elif rec_type == "step":
            step_idx = rec.get("step_idx", 0)
            t_sim = rec.get("t_sim", 0.0)

            # Extract comm_stats
            cs = rec.get("comm_stats")
            if cs:
                comm_metrics.append({
                    "step": step_idx,
                    "t_sim": t_sim,
                    "ppr": cs.get("ppr", 0.0),
                    "pdr": cs.get("pdr", 0.0),
                    "delay_mean": cs.get("delay_mean", 0.0),
                    "delay_p95": cs.get("delay_p95", 0.0),
                    "age_mean": cs.get("age_mean", 0.0),
                    "age_p95": cs.get("age_p95", 0.0),
                    "dropped": cs.get("dropped", 0),
                    "passed": cs.get("passed", 0),
                    "delivered": cs.get("delivered", 0),
                })

            # Extract PF estimates
            pf_list = rec.get("pf_estimates", [])
            for pf in pf_list:
                pf_metrics.append({
                    "step": step_idx,
                    "t_sim": t_sim,
                    "ship_id": pf.get("ship_id", -1),
                    "pos_error": pf.get("pos_error", 0.0),
                    "vel_error": pf.get("vel_error", 0.0),
                    "heading_error": abs(pf.get("heading_error", 0.0)),
                    "track_age": pf.get("track_age", 0.0),
                    "eff_particles": pf.get("eff_particles", 0.0),
                })

            # Extract RL data
            rl_list = rec.get("rl_data", [])
            for rl in rl_list:
                rl_metrics.append({
                    "step": step_idx,
                    "t_sim": t_sim,
                    "agent_id": rl.get("agent_id", ""),
                    "r_total": rl.get("r_total", 0.0),
                    "r_task": rl.get("r_task", 0.0),
                    "r_shaped": rl.get("r_shaped", 0.0),
                    "c_near": rl.get("c_near", 0.0),
                    "c_rule": rl.get("c_rule", 0.0),
                    "c_coll": rl.get("c_coll", 0.0),
                    "c_time": rl.get("c_time", 0.0),
                    "risk": rl.get("risk", 0.0),
                    "guard_triggered": rl.get("guard_triggered", False),
                })

    return {
        "episode_info": episode_info,
        "episode_end": episode_end,
        "comm_metrics": comm_metrics,
        "pf_metrics": pf_metrics,
        "rl_metrics": rl_metrics,
    }


def compute_episode_summary(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Compute summary statistics for an episode."""
    summary: Dict[str, float] = {}

    # Comm stats summary
    comm = metrics["comm_metrics"]
    if comm:
        summary["comm_ppr_mean"] = np.mean([c["ppr"] for c in comm])
        summary["comm_pdr_mean"] = np.mean([c["pdr"] for c in comm])
        summary["comm_delay_mean"] = np.mean([c["delay_mean"] for c in comm])
        summary["comm_delay_p95_mean"] = np.mean([c["delay_p95"] for c in comm])
        summary["comm_age_mean"] = np.mean([c["age_mean"] for c in comm])
        summary["comm_age_p95_mean"] = np.mean([c["age_p95"] for c in comm])
        summary["comm_dropped_total"] = sum(c["dropped"] for c in comm)

    # PF metrics summary
    pf = metrics["pf_metrics"]
    if pf:
        summary["pf_pos_error_mean"] = np.mean([p["pos_error"] for p in pf])
        summary["pf_pos_error_max"] = np.max([p["pos_error"] for p in pf])
        summary["pf_vel_error_mean"] = np.mean([p["vel_error"] for p in pf])
        summary["pf_heading_error_mean"] = np.mean([p["heading_error"] for p in pf])
        summary["pf_track_age_mean"] = np.mean([p["track_age"] for p in pf])
        summary["pf_eff_particles_mean"] = np.mean([p["eff_particles"] for p in pf])

    # RL metrics summary
    rl = metrics["rl_metrics"]
    if rl:
        summary["rl_reward_total"] = sum(r["r_total"] for r in rl)
        summary["rl_reward_mean"] = np.mean([r["r_total"] for r in rl])
        summary["rl_c_near_sum"] = sum(r["c_near"] for r in rl)
        summary["rl_c_coll_sum"] = sum(r["c_coll"] for r in rl)
        summary["rl_risk_mean"] = np.mean([r["risk"] for r in rl])
        summary["rl_guard_triggers"] = sum(1 for r in rl if r["guard_triggered"])

    # Episode outcome
    ep_end = metrics["episode_end"]
    if ep_end:
        term_reason = ep_end.get("term_reason", "unknown")
        summary["is_collision"] = 1.0 if "coll" in term_reason.lower() else 0.0
        summary["is_arrival"] = 1.0 if "arriv" in term_reason.lower() else 0.0
        summary["is_timeout"] = 1.0 if "timeout" in term_reason.lower() or "truncat" in term_reason.lower() else 0.0
        summary["total_steps"] = float(ep_end.get("total_steps", 0))

    return summary


def compute_correlations(
    summaries: List[Dict[str, float]],
    x_keys: List[str],
    y_keys: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute correlation matrix between x and y variables."""
    correlations: Dict[str, Dict[str, float]] = {}

    for x_key in x_keys:
        correlations[x_key] = {}
        x_vals = [s.get(x_key, np.nan) for s in summaries]
        x_arr = np.array(x_vals)

        for y_key in y_keys:
            y_vals = [s.get(y_key, np.nan) for s in summaries]
            y_arr = np.array(y_vals)

            # Filter out NaN values
            mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
            if mask.sum() < 3:
                correlations[x_key][y_key] = np.nan
                continue

            x_clean = x_arr[mask]
            y_clean = y_arr[mask]

            if np.std(x_clean) < 1e-9 or np.std(y_clean) < 1e-9:
                correlations[x_key][y_key] = np.nan
                continue

            corr = np.corrcoef(x_clean, y_clean)[0, 1]
            correlations[x_key][y_key] = float(corr)

    return correlations


def print_correlation_matrix(
    correlations: Dict[str, Dict[str, float]],
    title: str,
    x_label: str,
    y_label: str,
):
    """Print correlation matrix in a readable format."""
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print(f" {x_label} (rows) vs {y_label} (cols)")
    print(f"{'=' * 70}")

    x_keys = list(correlations.keys())
    if not x_keys:
        print("  (no data)")
        return

    y_keys = list(correlations[x_keys[0]].keys())

    # Header
    max_x_len = max(len(k) for k in x_keys)
    header = " " * (max_x_len + 2)
    for y in y_keys:
        header += f"{y[:12]:>12}  "
    print(header)
    print("-" * len(header))

    # Rows
    for x in x_keys:
        row = f"{x:<{max_x_len}}  "
        for y in y_keys:
            val = correlations[x][y]
            if np.isnan(val):
                row += f"{'N/A':>12}  "
            else:
                row += f"{val:>12.3f}  "
        print(row)


def analyze_causality_chain(
    summaries: List[Dict[str, float]],
) -> Tuple[Dict, Dict, Dict]:
    """
    Analyze the full causality chain: Comm → PF → RL

    Returns:
        comm_to_pf: Correlations from communication metrics to PF accuracy
        pf_to_rl: Correlations from PF accuracy to RL performance
        comm_to_rl: Direct correlations from communication to RL (for comparison)
    """
    # Communication metrics
    comm_keys = [
        "comm_ppr_mean",
        "comm_pdr_mean",
        "comm_delay_mean",
        "comm_age_mean",
        "comm_dropped_total",
    ]

    # PF tracking metrics
    pf_keys = [
        "pf_pos_error_mean",
        "pf_pos_error_max",
        "pf_vel_error_mean",
        "pf_heading_error_mean",
        "pf_track_age_mean",
    ]

    # RL performance metrics
    rl_keys = [
        "rl_reward_mean",
        "rl_c_coll_sum",
        "rl_risk_mean",
        "is_collision",
        "is_arrival",
    ]

    # Compute correlations
    comm_to_pf = compute_correlations(summaries, comm_keys, pf_keys)
    pf_to_rl = compute_correlations(summaries, pf_keys, rl_keys)
    comm_to_rl = compute_correlations(summaries, comm_keys, rl_keys)

    return comm_to_pf, pf_to_rl, comm_to_rl


def print_summary_statistics(summaries: List[Dict[str, float]]):
    """Print summary statistics across all episodes."""
    print(f"\n{'=' * 70}")
    print(" Episode Summary Statistics")
    print(f"{'=' * 70}")
    print(f"  Total episodes analyzed: {len(summaries)}")

    if not summaries:
        return

    # Key metrics
    metrics = [
        ("comm_ppr_mean", "Comm PPR (mean)"),
        ("comm_pdr_mean", "Comm PDR (mean)"),
        ("comm_delay_mean", "Comm Delay (mean, s)"),
        ("comm_age_mean", "Comm Age (mean, s)"),
        ("pf_pos_error_mean", "PF Pos Error (mean, m)"),
        ("pf_vel_error_mean", "PF Vel Error (mean, m/s)"),
        ("rl_reward_mean", "RL Reward (mean)"),
        ("rl_c_coll_sum", "RL Collision Cost (sum)"),
        ("is_collision", "Collision Rate"),
        ("is_arrival", "Arrival Rate"),
    ]

    print("\n  Metric                      Mean       Std       Min       Max")
    print("  " + "-" * 60)

    for key, label in metrics:
        vals = [s.get(key, np.nan) for s in summaries]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            print(f"  {label:<25} {np.mean(vals):>8.3f}  {np.std(vals):>8.3f}  {np.min(vals):>8.3f}  {np.max(vals):>8.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze AIS → PF → RL causality chain"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input staging directory (containing stage3/)"
    )
    parser.add_argument(
        "--mode", "-m",
        default="train",
        help="Mode subdirectory (train/eval)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file for results (optional)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    # Locate stage3 directory
    input_dir = Path(args.input)
    stage3_dir = input_dir / "stage3" / args.mode

    if not stage3_dir.is_dir():
        print(f"[error] Stage3 directory not found: {stage3_dir}")
        return 1

    print(f"[causality] Loading data from: {stage3_dir}")

    # Load data
    episodes_data = load_stage3_data(stage3_dir)
    print(f"[causality] Loaded {len(episodes_data)} episodes")

    if not episodes_data:
        print("[error] No episode data found")
        return 1

    # Extract metrics for each episode
    print("[causality] Extracting metrics...")
    episode_summaries = []
    for uid, records in episodes_data.items():
        metrics = extract_episode_metrics(records)
        summary = compute_episode_summary(metrics)
        summary["episode_uid"] = uid
        episode_summaries.append(summary)

    # Print summary statistics
    print_summary_statistics(episode_summaries)

    # Analyze causality chain
    print("\n[causality] Computing correlation matrices...")
    comm_to_pf, pf_to_rl, comm_to_rl = analyze_causality_chain(episode_summaries)

    # Print correlation matrices
    print_correlation_matrix(
        comm_to_pf,
        "Communication → PF Tracking Accuracy",
        "Communication Quality",
        "PF Tracking Error"
    )

    print_correlation_matrix(
        pf_to_rl,
        "PF Tracking Accuracy → RL Performance",
        "PF Tracking Error",
        "RL Performance"
    )

    print_correlation_matrix(
        comm_to_rl,
        "Communication → RL Performance (Direct)",
        "Communication Quality",
        "RL Performance"
    )

    # Key findings
    print(f"\n{'=' * 70}")
    print(" Key Causality Findings")
    print(f"{'=' * 70}")

    # Find strongest correlations
    findings = []

    # Comm → PF
    for x, y_dict in comm_to_pf.items():
        for y, corr in y_dict.items():
            if not np.isnan(corr) and abs(corr) > 0.3:
                findings.append((abs(corr), "Comm→PF", x, y, corr))

    # PF → RL
    for x, y_dict in pf_to_rl.items():
        for y, corr in y_dict.items():
            if not np.isnan(corr) and abs(corr) > 0.3:
                findings.append((abs(corr), "PF→RL", x, y, corr))

    # Comm → RL (direct)
    for x, y_dict in comm_to_rl.items():
        for y, corr in y_dict.items():
            if not np.isnan(corr) and abs(corr) > 0.3:
                findings.append((abs(corr), "Comm→RL", x, y, corr))

    findings.sort(reverse=True)

    if findings:
        print("\n  Significant correlations (|r| > 0.3):")
        print("  " + "-" * 60)
        for abs_corr, chain, x, y, corr in findings[:15]:
            direction = "+" if corr > 0 else "-"
            print(f"  [{chain:8}] {x} → {y}: r={corr:+.3f} ({direction})")
    else:
        print("\n  No significant correlations found (|r| > 0.3)")
        print("  This may indicate:")
        print("    - Insufficient data (need more episodes)")
        print("    - Low variance in metrics (perfect/near-perfect conditions)")
        print("    - Complex non-linear relationships")

    # Save results if output specified
    if args.output:
        results = {
            "summary_statistics": {
                k: float(np.mean([s.get(k, np.nan) for s in episode_summaries]))
                for k in episode_summaries[0].keys() if k != "episode_uid"
            },
            "comm_to_pf": comm_to_pf,
            "pf_to_rl": pf_to_rl,
            "comm_to_rl": comm_to_rl,
            "significant_findings": [
                {"chain": chain, "x": x, "y": y, "correlation": corr}
                for _, chain, x, y, corr in findings
            ],
        }

        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[causality] Results saved to: {output_path}")

    print(f"\n{'=' * 70}")
    print(" Analysis complete!")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
