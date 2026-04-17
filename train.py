"""
train.py — SOC Model Training
==============================
Trains a RandomForest classifier on log-derived features.
Labels are generated automatically from heuristic rules.

Usage:
    python train.py                            # train on HDFS_2k.log (default)
    python train.py --logs path/to/file.log   # custom log file
    python train.py --logs logs/ --augment    # include built-in synthetic logs

Output:
    model.pkl  — packaged model dict with threshold, accuracy, recall
"""

import argparse
import os
import glob
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# Local engine — single source of truth for features & labelling
from soc_engine import (
    FEATURE_NAMES,
    extract_features,
    extract_features_df,
    heuristic_label,
)


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC LOG BANK  (covers rare attack patterns not always in HDFS logs)
# ─────────────────────────────────────────────────────────────────────────────

SYNTHETIC_LOGS = {
    # label 0 — NORMAL
    0: [
        "INFO 2024-01-01 10:00:00 User login successful from 192.168.1.10",
        "INFO System health check passed all services running",
        "INFO Connection established to database on port 5432",
        "INFO Scheduled backup completed successfully at 03:00",
        "INFO HTTP 200 OK GET /api/status from 10.0.0.1",
        "INFO File read successfully /var/log/app.log",
        "INFO Service started on port 8080",
        "DEBUG Kernel module loaded successfully",
        "INFO Cron job completed exit 0",
        "INFO Process started by user appuser pid=1234",
        "081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating",
        "081109 204005 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.73.220:50010 is added to blk_-2641638092720413974",
        "081109 204655 556 INFO dfs.DataNode$PacketResponder: Received block blk_3587508140051953248 of size 67108864 from /10.251.30.6:50010",
        "081109 204815 653 INFO dfs.DataNode$DataXceiver: Receiving block blk_5792489080791696128 src: /10.251.30.6:33145 dest: /10.251.30.6:50010",
        "081109 205200 720 INFO dfs.DataNode$DataXceiver: Serving block blk_9876543210987654321 to /10.250.11.85",
        "INFO Disk usage at 43% — within normal thresholds",
        "INFO Snapshot created for volume /dev/sda1",
        "INFO TLS handshake completed successfully with peer 10.0.5.3",
        "INFO Auth token refreshed for user alice session valid 3600s",
        "INFO Metrics exported to monitoring endpoint — 42 series",
    ],
    # label 1 — ANOMALY
    1: [
        # Security attacks
        "ERROR Failed login attempt from 192.168.1.5 — authentication failure after 3 tries",
        "WARN Multiple failed SSH login attempts from 10.0.0.3 — possible brute force",
        "ERROR SQL injection detected in request: ' OR 1=1; DROP TABLE users;--",
        "ERROR Unauthorized access attempt to /admin/config from 192.168.1.99",
        "CRITICAL Malware signature detected in file upload: trojan.win32.banker",
        "ERROR Buffer overflow detected in process memory region 0xDEADBEEF",
        "WARN Port scan detected from 192.168.1.200 — 300 ports probed in 2s",
        "ERROR Root login attempt via sudo -i from non-admin user bob",
        "ERROR Brute force attack detected: 50 login failures in 30 seconds from 10.0.0.5",
        "CRITICAL Ransomware behavior detected: mass file encryption event started",
        "ERROR XSS payload found in request parameter q=<script>alert(1)</script>",
        "ERROR Connection refused to 10.0.0.1:4444 — known C2 port",
        "WARN Authentication failure for user root from 192.168.1.100",
        "ERROR Command injection attempt detected in HTTP parameter cmd=; cat /etc/passwd",
        "ERROR 500 Internal Server Error on POST /api/execute — 47 consecutive failures",
        "WARN Access denied for user guest to resource /etc/shadow — privilege escalation",
        # HDFS anomalies
        "081109 203818 35 ERROR dfs.DataNode$DataXceiver: 10.251.42.220:50010:Got exception while serving blk_-1608999687919862906 to /10.251.203.80:\njava.io.IOException: Connection reset by peer",
        "081109 204544 403 WARN dfs.DataNode$PacketResponder: PacketResponder 2 for block blk_8229193803249955061 Interrupted",
        "081109 204903 678 ERROR dfs.DataNode$DataXceiver: writeBlock blk_-7049066296714776736 received exception java.io.IOException: Broken pipe",
        "081109 205134 710 ERROR dfs.FSDataset: failed to delete /data/blocks/blk_2345678901234567",
        "081109 205312 735 WARN dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_1234567890123456 pipeline error: java.io.IOException",
        "081109 205845 800 ERROR dfs.DataNode: Got exception while receiving block blk_9999999999999999: java.io.IOException: Pipe closed",
        "FATAL DataNode service crashed — disk failure on /data/hdfs/dn — data loss possible",
        "ERROR missing block blk_0000000000000001 during replication — 0 replicas found",
        "WARN HDFS NameNode lost contact with 3 DataNodes — possible network partition",
        "ERROR corrupt block blk_1111111111111111 detected during checksum verification",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# LOG LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_log_file(path: str) -> list[str]:
    """Load a single .log or .txt file into a list of lines."""
    encodings = ["utf-8", "latin-1", "utf-8-sig"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="ignore") as fh:
                return [ln.strip() for ln in fh if ln.strip()]
        except Exception:
            continue
    raise IOError(f"Cannot read file: {path}")


def load_logs_from_path(path: str) -> list[str]:
    """
    Load logs from:
      - a single file (e.g. HDFS_2k.log)
      - a directory  (all *.log and *.txt files)
    """
    if os.path.isfile(path):
        lines = load_log_file(path)
        print(f"  Loaded {len(lines):,} lines from {path}")
        return lines
    elif os.path.isdir(path):
        all_lines = []
        for pattern in ("*.log", "*.txt"):
            for fpath in glob.glob(os.path.join(path, pattern)):
                lines = load_log_file(fpath)
                print(f"  Loaded {len(lines):,} lines from {fpath}")
                all_lines.extend(lines)
        return all_lines
    else:
        raise FileNotFoundError(f"Path not found: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    log_lines: list[str],
    augment: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build (X, y) from raw log lines.
    Labels are heuristic-assigned. Synthetic logs optionally appended.
    """
    all_logs   = list(log_lines)
    all_labels = [heuristic_label(l) for l in all_logs]

    if augment:
        for lbl, logs in SYNTHETIC_LOGS.items():
            all_logs.extend(logs)
            all_labels.extend([lbl] * len(logs))
        print(f"  Augmented with {sum(len(v) for v in SYNTHETIC_LOGS.values())} synthetic logs")

    X = extract_features_df(all_logs)
    y = np.array(all_labels)

    n_normal  = int((y == 0).sum())
    n_anomaly = int((y == 1).sum())
    print(f"  Dataset: {len(y):,} samples  |  normal={n_normal:,}  anomaly={n_anomaly:,}")

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(log_path: str, augment: bool, output_path: str) -> None:
    print("=" * 60)
    print("  SOC Model Training")
    print("=" * 60)

    # ── Load logs ──────────────────────────────────────────────────────────────
    log_lines = load_logs_from_path(log_path)

    # ── Build dataset ──────────────────────────────────────────────────────────
    print("\n  Building dataset...")
    X, y = build_dataset(log_lines, augment=augment)

    # ── Train / val split ──────────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train: {len(y_train):,}  |  Val: {len(y_val):,}")

    # ── SMOTE: balance training split only ────────────────────────────────────
    print("\n  Applying SMOTE to balance training data...")
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    n_normal  = int((y_train == 0).sum())
    n_anomaly = int((y_train == 1).sum())
    print(f"  After SMOTE: normal={n_normal:,}  anomaly={n_anomaly:,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n  Training RandomForest...")
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── Cross-validation ───────────────────────────────────────────────────────
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1", n_jobs=-1)
    print(f"  CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Fixed threshold — no automatic optimization ───────────────────────────
    threshold = 0.4
    print(f"  Using fixed threshold: {threshold}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred = (model.predict_proba(X_val)[:, 1] >= threshold).astype(int)
    print("\n  Classification Report (validation set):")
    print(classification_report(y_val, y_pred, target_names=["NORMAL", "ANOMALY"]))

    acc    = round(accuracy_score(y_val, y_pred) * 100, 1)
    recall = round(recall_score(y_val, y_pred) * 100, 1)

    # ── Feature importance (top 10) ───────────────────────────────────────────
    fi = sorted(zip(FEATURE_NAMES, model.feature_importances_),
                key=lambda x: x[1], reverse=True)
    print("\n  Top 10 feature importances:")
    for feat, imp in fi[:10]:
        bar = "█" * int(imp * 50)
        print(f"    {feat:<25} {imp:.4f} {bar}")

    # ── Save model package ────────────────────────────────────────────────────
    pkg = {
        "model":          model,
        "feature_names":  FEATURE_NAMES,
        "threshold":      threshold,
        "accuracy":       acc,
        "recall":         recall,
    }
    joblib.dump(pkg, output_path)
    print(f"\n  ✅ Model saved to: {output_path}")
    print(f"     accuracy={acc}%  recall={recall}%  threshold={threshold}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SOC anomaly detection model.")
    parser.add_argument(
        "--logs", type=str, default="HDFS_2k.log",
        help="Path to log file or directory (default: HDFS_2k.log)"
    )
    parser.add_argument(
        "--augment", action="store_true", default=True,
        help="Append synthetic logs to training data (default: True)"
    )
    parser.add_argument(
        "--no-augment", dest="augment", action="store_false",
        help="Disable synthetic log augmentation"
    )
    parser.add_argument(
        "--output", type=str, default="model.pkl",
        help="Output model path (default: model.pkl)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.logs):
        print(f"  ⚠️  Log file not found: {args.logs}")
        print("  Training on synthetic logs only (use --logs to specify a file).")
        print("=" * 60)
        print("  SOC Model Training  (synthetic-only mode)")
        print("=" * 60)
        X, y = build_dataset([], augment=True)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        threshold = 0.4
        print(f"  Using fixed threshold: {threshold}")
        y_pred = (model.predict_proba(X_val)[:, 1] >= threshold).astype(int)
        print(classification_report(y_val, y_pred, target_names=["NORMAL", "ANOMALY"]))
        acc    = round(accuracy_score(y_val, y_pred) * 100, 1)
        recall = round(recall_score(y_val, y_pred) * 100, 1)
        pkg = {
            "model": model, "feature_names": FEATURE_NAMES,
            "threshold": threshold, "accuracy": acc, "recall": recall,
        }
        joblib.dump(pkg, args.output)
        print(f"\n  ✅ Model saved to: {args.output}")
        sys.exit(0)

    train(args.logs, augment=args.augment, output_path=args.output)