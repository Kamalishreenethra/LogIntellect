"""
predict.py — SOC Inference Pipeline
=====================================
Loads a trained model and runs hybrid detection on log files.

Usage:
    python predict.py                              # demo on built-in sample logs
    python predict.py --logs HDFS_2k.log
    python predict.py --logs HDFS_2k.log --threshold 0.40
    python predict.py --logs HDFS_2k.log --output results.json
    cat my.log | python predict.py                 # pipe from stdin

Output (stdout):
    Formatted SOC report with incident summaries

Output (--output):
    JSON file — one record per log line + incident list
"""

import argparse
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd

from soc_engine import (
    FEATURE_NAMES,
    extract_features,
    extract_features_df,
    rule_score,
    hybrid_score,
    map_risk_level,
    detect_attack_type,
    generate_summary,
    extract_ip,
    extract_timestamp,
    temporal_analysis,
    correlate_incidents,
)

DEFAULT_MODEL = "model.pkl"
DEFAULT_SHOW  = 25


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str) -> tuple:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: '{model_path}'\n"
            f"Run:  python train.py\nto create it first."
        )
    pkg = joblib.load(model_path)
    # Validate feature list matches current engine
    saved = pkg.get("feature_names", [])
    if saved and list(saved) != list(FEATURE_NAMES):
        print("  ⚠️  Feature mismatch — model was trained with different features.")
        print("      Re-run: python train.py")
    print(f"  ✅ Model loaded  |  threshold={pkg['threshold']}"
          f"  |  accuracy={pkg['accuracy']}%  recall={pkg['recall']}%")
    return pkg["model"], pkg["threshold"]


# ─────────────────────────────────────────────────────────────────────────────
# LOG LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_logs(log_path: str | None) -> list[str]:
    if log_path:
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Log file not found: '{log_path}'")
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f if l.strip()]
        print(f"  ✅ Loaded {len(lines):,} log lines from {log_path}")
    else:
        lines = [l.strip() for l in sys.stdin if l.strip()]
        print(f"  ✅ Read {len(lines):,} log lines from stdin")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# CORE DETECTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run(
    log_lines: list[str],
    model,
    threshold: float,
) -> tuple[list[dict], list[dict], dict]:
    """
    Full detection pipeline: features → ML prob → rule score → hybrid score → risk.

    Returns
    -------
    (results, incidents, temporal)
    """
    # ── Feature extraction — single unified call ───────────────────────────────
    feat_df = extract_features_df(log_lines)

    # ── ML probabilities ───────────────────────────────────────────────────────
    ml_proba = model.predict_proba(feat_df)[:, 1]  # P(anomaly)

    # ── Assemble per-log results ──────────────────────────────────────────────
    results = []
    for i, line in enumerate(log_lines):
        features = feat_df.iloc[i].to_dict()
        ml_prob  = float(ml_proba[i])

        r_score, fired = rule_score(features)
        h_score        = hybrid_score(ml_prob, r_score)

        if r_score >= 0.6:
            is_attack = True
        elif ml_prob >= threshold:
            is_attack = True
        else:
            is_attack = False

        risk_level  = map_risk_level(h_score)
        attack_type = detect_attack_type(features, is_attack)
        summary     = generate_summary(features, fired, is_attack)
        ip          = extract_ip(line)
        timestamp   = extract_timestamp(line)

        results.append({
            "index":       i,
            "log":         line,
            "is_attack":   is_attack,
            "ml_prob":     round(ml_prob, 4),
            "rule_score":  round(r_score, 4),
            "final_score": round(h_score, 4),
            "risk_level":  risk_level,
            "attack_type": attack_type,
            "summary":     summary,
            "fired_rules": fired,
            "ip":          ip,
            "timestamp":   timestamp.isoformat() if timestamp else None,
        })

    # ── Temporal analysis ─────────────────────────────────────────────────────
    temporal = temporal_analysis(results)

    # ── Incident correlation ──────────────────────────────────────────────────
    incidents = correlate_incidents(results)

    return results, incidents, temporal


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE REPORT
# ─────────────────────────────────────────────────────────────────────────────

RISK_ICON = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "✅"}


def print_report(
    results:  list[dict],
    incidents: list[dict],
    temporal:  dict,
    show: int = DEFAULT_SHOW,
) -> None:
    total    = len(results)
    n_attack = sum(1 for r in results if r["is_attack"])
    n_normal = total - n_attack
    by_risk  = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in results:
        by_risk[r["risk_level"]] += 1

    # ── Summary header ─────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  SOC LOG ANALYSIS REPORT")
    print("═" * 70)
    print(f"  Total logs    : {total:,}")
    print(f"  Anomalies     : {n_attack:,}  ({n_attack/total*100:.1f}%)")
    print(f"  Normal        : {n_normal:,}")
    print()
    print(f"  🔴 CRITICAL   : {by_risk['CRITICAL']:,}")
    print(f"  🟠 HIGH       : {by_risk['HIGH']:,}")
    print(f"  🟡 MEDIUM     : {by_risk['MEDIUM']:,}")
    print(f"  ✅ LOW        : {by_risk['LOW']:,}")

    # ── Per-log results ────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  {'#':<5} {'Risk':<12} {'ML':>6} {'Rule':>6} {'Score':>6}  Summary")
    print(f"{'─'*70}")

    for r in results[:show]:
        icon = RISK_ICON.get(r["risk_level"], "  ")
        flag = "⚠ " if r["is_attack"] else "  "
        print(
            f"  {r['index']+1:<5}"
            f" {icon} {r['risk_level']:<9}"
            f" {r['ml_prob']*100:>5.1f}%"
            f" {r['rule_score']*100:>5.1f}%"
            f" {r['final_score']*100:>5.1f}%"
            f"  {flag}{r['summary'][:55]}"
        )

    if total > show:
        print(f"\n  ... and {total-show:,} more. Use --show N to see more.")

    # ── Incidents ─────────────────────────────────────────────────────────────
    if incidents:
        print(f"\n{'═'*70}")
        print(f"  INCIDENT CORRELATION  ({len(incidents)} incident(s) detected)")
        print(f"{'═'*70}")
        for inc in incidents:
            icon = RISK_ICON.get(inc["risk_level"], "  ")
            ip_str = f"  IP: {inc['primary_ip']}" if inc["primary_ip"] else ""
            print(
                f"\n  [INC-{inc['incident_id']:03d}] {icon} {inc['risk_level']}"
                f"  |  {inc['num_logs']} log(s){ip_str}"
            )
            print(f"            Type    : {inc['attack_type']}")
            print(f"            Summary : {inc['summary']}")
            print(f"            Max score: {inc['max_score']*100:.1f}%")

    # ── Temporal / suspicious IPs ──────────────────────────────────────────────
    flagged_ips = {ip: info for ip, info in temporal.items() if info["flagged"]}
    if flagged_ips:
        print(f"\n{'═'*70}")
        print(f"  SUSPICIOUS IPs  ({len(flagged_ips)} flagged by temporal analysis)")
        print(f"{'═'*70}")
        for ip, info in sorted(flagged_ips.items(),
                                key=lambda x: x[1]["attack_count"], reverse=True):
            print(f"  🚨 {ip:<18}  attacks={info['attack_count']}  "
                  f"total_events={info.get('total_events', '?')}")

    # ── Suspicious IP summary ─────────────────────────────────────────────────
    suspicious_ips = [
        ip for ip, data in temporal.items()
        if isinstance(data, dict) and data.get("flagged") is True
    ]
    # Sort by attack_count descending, deduplicate preserving order
    seen: set = set()
    suspicious_ips = sorted(
        [ip for ip in suspicious_ips if not (ip in seen or seen.add(ip))],
        key=lambda ip: temporal[ip].get("attack_count", 0),
        reverse=True,
    )

    print(f"\n{'═'*70}")
    print(f"  Analysis complete. {len(incidents)} incident(s) | "
          f"{len(suspicious_ips)} suspicious IP(s)")

    if suspicious_ips:
        print("\n  Suspicious IPs:")
        for ip in suspicious_ips[:10]:
            info = temporal[ip]
            print(f"    🚨 {ip:<18}  attacks={info['attack_count']}"
                  f"  total_events={info.get('total_events', '?')}"
                  f"  burst={info.get('burst', False)}")

    print(f"{'═'*70}\n")

    # ── DEBUG: temporal sample (first 6 IPs) ──────────────────────────────────
    print("\nDEBUG TEMPORAL SAMPLE:")
    for i, (ip, data) in enumerate(temporal.items()):
        print(f"  {ip}  →  {data}")
        if i >= 5:
            break


# ─────────────────────────────────────────────────────────────────────────────
# DEMO LOGS  (used when no file is supplied)
# ─────────────────────────────────────────────────────────────────────────────

DEMO_LOGS = [
    "INFO 2024-01-15 10:00:01 User alice logged in from 192.168.1.10",
    "INFO 2024-01-15 10:00:03 GET /api/status 200 OK from 10.0.0.1",
    "WARN 2024-01-15 10:01:00 Multiple failed SSH login attempts from 10.0.0.5 — possible brute force",
    "ERROR 2024-01-15 10:01:02 Authentication failure for user root from 10.0.0.5 (attempt 4)",
    "ERROR 2024-01-15 10:01:05 Authentication failure for user root from 10.0.0.5 (attempt 5)",
    "081109 204000 35 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating",
    "081109 204005 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.73.220:50010 is added to blk_-2641638092720413974",
    "081109 204655 556 INFO dfs.DataNode$PacketResponder: Received block blk_3587508140051953248 of size 67108864 from /10.251.30.6",
    "081109 204903 678 ERROR dfs.DataNode$DataXceiver: writeBlock blk_-7049066296714776736 received exception java.io.IOException: Broken pipe",
    "INFO 2024-01-15 10:02:00 Backup job completed successfully — 4.2 GB written",
    "ERROR 2024-01-15 10:03:00 SQL injection detected in request from 192.168.1.99: ' OR 1=1--",
    "CRITICAL 2024-01-15 10:03:01 Malware detected in upload: ransomware.wannacry variant",
    "INFO 2024-01-15 10:04:00 TLS certificate renewed — valid for 90 days",
    "WARN 2024-01-15 10:05:00 Disk usage at 87% on /dev/sda1 — approaching limit",
]


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SOC log anomaly detection pipeline.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--logs",      type=str,   default=None,
                        help="Path to log file. Reads stdin if omitted.")
    parser.add_argument("--model",     type=str,   default=DEFAULT_MODEL,
                        help=f"Model .pkl path (default: {DEFAULT_MODEL})")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override hybrid score threshold (default: from model)")
    parser.add_argument("--show",      type=int,   default=DEFAULT_SHOW,
                        help=f"Lines to print in report (default: {DEFAULT_SHOW})")
    parser.add_argument("--output",    type=str,   default=None,
                        help="Save full results as JSON to this path.")
    args = parser.parse_args()

    print("=" * 70)
    print("  SOC Log Analyzer — Detection Pipeline")
    print("=" * 70)

    # Load model
    model, default_threshold = load_model(args.model)
    threshold = args.threshold if args.threshold is not None else default_threshold
    print(f"  Using hybrid threshold: {threshold}")

    # Load logs
    if args.logs:
        log_lines = load_logs(args.logs)
    elif not sys.stdin.isatty():
        log_lines = load_logs(None)  # stdin
    else:
        print("  No log file supplied — running on built-in demo logs.")
        log_lines = DEMO_LOGS

    # Run detection
    print("\n  Running detection pipeline...")
    results, incidents, temporal = run(log_lines, model, threshold)

    # Print report
    print_report(results, incidents, temporal, show=args.show)

    # Save JSON
    if args.output:
        output = {
            "results":   results,
            "incidents": incidents,
            "temporal":  {ip: info for ip, info in temporal.items()},
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  💾 Results saved to: {args.output}")