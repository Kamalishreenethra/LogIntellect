"""
soc_engine.py — Unified SOC Feature Extraction & Scoring Engine
================================================================
Single source of truth for:
  - Feature extraction  (used by BOTH train.py and predict.py)
  - Rule-based scoring
  - Risk level mapping
  - Temporal analysis
  - Incident correlation
  - Log file parsing
"""

import re
import io
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE NAMES  (must stay in sync — never add/remove without retraining)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    # Log level
    "level_debug", "level_info", "level_warn", "level_error", "level_fatal",
    # Generic security keywords
    "kw_unauthorized", "kw_denied", "kw_brute", "kw_attack", "kw_malware",
    "kw_overflow", "kw_injection", "kw_scan", "kw_root_esc", "kw_auth_fail",
    "kw_timeout", "kw_refused", "kw_suspicious_port", "kw_http_error",
    "kw_multi_attempt", "kw_failed",
    # HDFS-specific
    "hdfs_has_block", "hdfs_component", "hdfs_normal_op", "hdfs_exception",
    "hdfs_interrupted", "hdfs_failed_op", "hdfs_pipeline_err",
    "hdfs_write_err", "hdfs_close_err", "hdfs_is_clean", "hdfs_is_anomaly",
    # Structural metadata
    "log_length", "digit_ratio", "ip_present", "repeat_fail",
]


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  (one function — called identically in train & predict)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(log_text: str) -> dict:
    """
    Extract a fixed-length numeric feature vector from a raw log line.
    Always returns all keys in FEATURE_NAMES — no extras, no missing.
    """
    text = str(log_text)
    tl   = text.lower()

    # ── Log level ─────────────────────────────────────────────────────────────
    level_debug = int(bool(re.search(r'\bDEBUG\b', text)))
    level_info  = int(bool(re.search(r'\bINFO\b',  text)))
    level_warn  = int(bool(re.search(r'\bWARN(?:ING)?\b', text)))
    level_error = int(bool(re.search(r'\bERROR\b', text)))
    level_fatal = int(bool(re.search(r'\bFATAL\b|\bSEVERE\b|\bCRITICAL\b', text)))

    # ── Generic security keywords ─────────────────────────────────────────────
    kw_unauthorized    = int(bool(re.search(r'\bunauthorized\b', tl)))
    kw_denied          = int(bool(re.search(r'\baccess denied\b|\bdenied\b', tl)))
    kw_brute           = int(bool(re.search(r'\bbrute.?force\b', tl)))
    kw_attack          = int(bool(re.search(r'\battack\b|\bexploit\b', tl)))
    kw_malware         = int(bool(re.search(r'\bmalware\b|\bvirus\b|\bransomware\b|\btrojan\b', tl)))
    kw_overflow        = int(bool(re.search(r'\bbuffer overflow\b|\bstack overflow\b', tl)))
    kw_injection       = int(bool(re.search(r'\bsql injection\b|\bxss\b|\bcommand inject', tl)))
    kw_scan            = int(bool(re.search(r'\bport scan\b|\bscan detect\b', tl)))
    kw_root_esc        = int(bool(re.search(r'\broot login\b|\bprivilege escal\b|\bsudo -i\b', tl)))
    kw_auth_fail       = int(bool(re.search(r'\bauthentication fail|\bauth fail|\blogin fail', tl)))
    kw_timeout         = int(bool(re.search(r'\btimeout\b', tl)))
    kw_refused         = int(bool(re.search(r'\bconnection refused\b|\brefused\b', tl)))
    kw_suspicious_port = int(bool(re.search(r'\b(4444|1337|31337|6666|9001|4321)\b', text)))
    kw_http_error      = int(bool(re.search(r'\b[45]\d{2}\b', text)))
    kw_multi_attempt   = int(bool(re.search(r'attempt[s]?\s*[#:]?\s*\d+|\d+\s*attempt', tl)))
    kw_failed          = int(bool(re.search(r'\bfailed\b|\bfailure\b', tl)))

    # ── HDFS-specific features ────────────────────────────────────────────────
    hdfs_has_block  = int(bool(re.search(r'\bblk_[+-]?\d+', text)))
    hdfs_component  = int(bool(re.search(
        r'\bDataNode\b|\bNameNode\b|\bFSNamesystem\b|\bPacketResponder\b'
        r'|\bDataXceiver\b|\bFSDataset\b', text)))
    hdfs_normal_op  = int(bool(re.search(
        r'\bterminating\b|\bReceived block\b|\baddStoredBlock\b|\bblockMap updated\b'
        r'|\bReceiving block\b|\bServing block\b|\ballocateBlock\b'
        r'|\badded to blk_\b|\bis added\b', text, re.IGNORECASE)))
    hdfs_exception  = int(bool(re.search(
        r'\bIOException\b|\bjava\.io\b|\bBroken pipe\b|\bConnection reset\b'
        r'|\bPipe closed\b|\bSocket closed\b|\bException\b', text)))
    hdfs_interrupted = int(bool(re.search(r'\bInterrupted\b', text)))
    hdfs_failed_op  = int(bool(re.search(
        r'\bfailed to delete\b|\bfailed to write\b|\bfailed to replicate\b'
        r'|\blost block\b|\bcorrupt block\b|\bmissing block\b|\bunknown block\b', tl)))
    hdfs_pipeline_err = int(bool(re.search(
        r'\bpipeline\b.*\bfail\b|\bpipeline\b.*\berror\b|\bpipeline\b.*\bclose\b', tl)))
    hdfs_write_err  = int(bool(re.search(
        r'\bwriteBlock\b.*\bexception\b|\bwriteBlock\b.*\berror\b'
        r'|\breceived exception\b', text, re.IGNORECASE)))
    hdfs_close_err  = int(bool(re.search(
        r'\bexceptions\.close\b|\bclose.*exception\b|\bfinalizeBlock\b.*\bfail', tl)))

    hdfs_anomaly_count = (
        hdfs_exception + hdfs_interrupted + hdfs_failed_op +
        hdfs_pipeline_err + hdfs_write_err + hdfs_close_err +
        level_error + level_warn + level_fatal
    )
    hdfs_is_clean   = int(hdfs_has_block and level_info and hdfs_anomaly_count == 0)
    hdfs_is_anomaly = int(hdfs_has_block and hdfs_anomaly_count >= 1)

    # ── Structural metadata ───────────────────────────────────────────────────
    log_length  = min(len(text) / 300.0, 1.0)
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    ip_present  = int(bool(re.search(r'/?\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?', text)))
    repeat_fail = int(len(re.findall(r'fail', tl)) > 1)

    return {
        "level_debug": level_debug, "level_info": level_info,
        "level_warn": level_warn,   "level_error": level_error,
        "level_fatal": level_fatal,
        "kw_unauthorized": kw_unauthorized, "kw_denied": kw_denied,
        "kw_brute": kw_brute,       "kw_attack": kw_attack,
        "kw_malware": kw_malware,   "kw_overflow": kw_overflow,
        "kw_injection": kw_injection, "kw_scan": kw_scan,
        "kw_root_esc": kw_root_esc, "kw_auth_fail": kw_auth_fail,
        "kw_timeout": kw_timeout,   "kw_refused": kw_refused,
        "kw_suspicious_port": kw_suspicious_port,
        "kw_http_error": kw_http_error,
        "kw_multi_attempt": kw_multi_attempt, "kw_failed": kw_failed,
        "hdfs_has_block": hdfs_has_block, "hdfs_component": hdfs_component,
        "hdfs_normal_op": hdfs_normal_op, "hdfs_exception": hdfs_exception,
        "hdfs_interrupted": hdfs_interrupted, "hdfs_failed_op": hdfs_failed_op,
        "hdfs_pipeline_err": hdfs_pipeline_err, "hdfs_write_err": hdfs_write_err,
        "hdfs_close_err": hdfs_close_err,
        "hdfs_is_clean": hdfs_is_clean, "hdfs_is_anomaly": hdfs_is_anomaly,
        "log_length": log_length, "digit_ratio": digit_ratio,
        "ip_present": ip_present,  "repeat_fail": repeat_fail,
    }


def extract_features_df(logs: list) -> pd.DataFrame:
    """Batch extraction — returns DataFrame with columns in FEATURE_NAMES order."""
    rows = [extract_features(log) for log in logs]
    df   = pd.DataFrame(rows, columns=FEATURE_NAMES)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# HEURISTIC LABELLER  (used only in train.py to auto-label raw logs)
# ─────────────────────────────────────────────────────────────────────────────

def heuristic_label(log_text: str) -> int:
    """
    Return 1 (anomaly) or 0 (normal) using deterministic rules.
    This is the ground-truth substitute when no human labels exist.
    """
    f = extract_features(log_text)
    tl = log_text.lower()

    # Strong anomaly signals
    if any([
        f["level_fatal"],
        f["level_error"],
        f["hdfs_exception"],
        f["hdfs_failed_op"],
        f["hdfs_write_err"],
        f["hdfs_close_err"],
        f["kw_unauthorized"],
        f["kw_brute"],
        f["kw_injection"],
        f["kw_malware"],
        f["kw_overflow"],
        f["kw_attack"],
        f["kw_root_esc"],
    ]):
        return 1

    # Warn + extra signal → anomaly
    if f["level_warn"] and any([
        f["hdfs_interrupted"],
        f["hdfs_pipeline_err"],
        f["kw_failed"],
        f["kw_refused"],
        f["repeat_fail"],
    ]):
        return 1

    # Clean HDFS block operation
    if f["hdfs_is_clean"]:
        return 0

    # Plain INFO / DEBUG → normal
    if f["level_info"] or f["level_debug"]:
        return 0

    # No level tag but has anomaly HDFS signal
    if f["hdfs_is_anomaly"]:
        return 1

    # Fallback: check warn alone
    if f["level_warn"]:
        return 1

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED SCORER  (deterministic, independent of ML)
# ─────────────────────────────────────────────────────────────────────────────

# Each tuple: (feature_key, weight, label)
RULE_TABLE = [
    # HDFS anomalies
    ("hdfs_exception",    0.40, "HDFS IOException / Broken Pipe"),
    ("hdfs_failed_op",    0.30, "HDFS block operation failure"),
    ("hdfs_write_err",    0.25, "HDFS writeBlock exception"),
    ("hdfs_close_err",    0.20, "HDFS abnormal block close"),
    ("hdfs_pipeline_err", 0.20, "HDFS pipeline error"),
    ("hdfs_interrupted",  0.15, "HDFS pipeline interrupted"),
    ("hdfs_is_anomaly",   0.10, "HDFS anomaly composite"),
    # Log levels
    ("level_fatal",  0.40, "FATAL/CRITICAL log level"),
    ("level_error",  0.30, "ERROR level log"),
    ("level_warn",   0.20, "WARN level log"),
    # Security keywords
    ("kw_brute",          0.35, "Brute-force pattern"),
    ("kw_injection",      0.35, "Injection attack (SQLi/XSS)"),
    ("kw_malware",        0.40, "Malware/ransomware signature"),
    ("kw_overflow",       0.35, "Buffer overflow attempt"),
    ("kw_attack",         0.30, "Explicit attack keyword"),
    ("kw_root_esc",       0.30, "Root/privilege escalation"),
    ("kw_unauthorized",   0.25, "Unauthorized access attempt"),
    ("kw_denied",         0.15, "Access denied"),
    ("kw_scan",           0.20, "Port scan / recon"),
    ("kw_suspicious_port",0.25, "Known C2 port referenced"),
    ("kw_auth_fail",      0.20, "Authentication failure"),
    ("kw_failed",         0.10, "Operation failure"),
    ("kw_refused",        0.10, "Connection refused"),
    ("kw_timeout",        0.08, "Connection timeout"),
    ("kw_http_error",     0.08, "HTTP 4xx/5xx error"),
    ("kw_multi_attempt",  0.15, "Multiple attempt counter"),
    ("repeat_fail",       0.10, "Repeated failures in line"),
]


def rule_score(features: dict) -> tuple[float, list[dict]]:
    """
    Compute a rule-based risk score in [0, 1] and return fired rules.

    Returns
    -------
    (score, fired_rules)
    score       : float in [0.0, 1.0]
    fired_rules : list of {"label": str, "weight": float}
    """
    raw   = 0.0
    fired = []
    for key, weight, label in RULE_TABLE:
        if features.get(key, 0):
            raw += weight
            fired.append({"label": label, "weight": weight})

    # Penalty: clean HDFS ops lower the score
    if features.get("hdfs_is_clean", 0):
        raw = max(0.0, raw - 0.3)

    score = min(raw, 1.0)
    return score, fired


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID SCORER  (combines ML probability + rule score)
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_score(ml_prob: float, r_score: float) -> float:
    """If strong rule signal, override ML. Otherwise take the stronger signal."""
    if r_score >= 0.6:
        return r_score
    return max(ml_prob, r_score)


# ─────────────────────────────────────────────────────────────────────────────
# RISK LEVEL MAPPING
# ─────────────────────────────────────────────────────────────────────────────

def map_risk_level(score: float) -> str:
    """Map a continuous score in [0,1] to a categorical risk level."""
    if score >= 0.9:
        return "CRITICAL"
    elif score >= 0.7:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# ATTACK TYPE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_attack_type(features: dict, is_attack: bool) -> str:
    if not is_attack:
        return "—"
    f = features

    if f.get("hdfs_has_block"):
        if f.get("hdfs_exception") and f.get("hdfs_failed_op"):
            return "HDFS Block Corruption / IO Failure"
        if f.get("hdfs_exception"):
            return "HDFS Write Exception (IOException / Broken Pipe)"
        if f.get("hdfs_interrupted"):
            return "HDFS Pipeline Interruption"
        if f.get("hdfs_failed_op"):
            return "HDFS Block Operation Failure"
        if f.get("hdfs_close_err") or f.get("hdfs_pipeline_err"):
            return "HDFS Pipeline / Close Error"
        if f.get("level_error") or f.get("level_warn"):
            return "HDFS Infrastructure Anomaly"

    if f.get("kw_brute") or (f.get("kw_auth_fail") and f.get("repeat_fail")):
        return "Brute Force"
    if f.get("kw_injection"):
        return "Injection Attack (SQLi / XSS)"
    if f.get("kw_scan"):
        return "Port Scan / Reconnaissance"
    if f.get("kw_malware") or f.get("kw_overflow"):
        return "Malware / Exploit"
    if f.get("kw_unauthorized") or f.get("kw_denied") or f.get("kw_root_esc"):
        return "Unauthorized Access"
    if f.get("level_fatal"):
        return "System Failure / Crash"
    if f.get("kw_refused") or f.get("kw_timeout"):
        return "Network Anomaly / DoS"

    return "Anomalous Behavior"


# ─────────────────────────────────────────────────────────────────────────────
# EXPLANATION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary(features: dict, fired_rules: list, is_attack: bool) -> str:
    if features.get("hdfs_is_clean", 0):
        return "Normal HDFS block operation — no errors detected."
    if not is_attack:
        return "No significant threat indicators. Log matches normal patterns."
    if fired_rules:
        top = fired_rules[:2]
        return " → ".join(r["label"] for r in top)
    return "Statistical anomaly: pattern deviates from normal baseline."


# ─────────────────────────────────────────────────────────────────────────────
# IP EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

_IP_RE = re.compile(r'(?<!\d)(\d{1,3}(?:\.\d{1,3}){3})(?!\d)')

def extract_ip(log_text: str) -> str | None:
    m = _IP_RE.search(log_text)
    return m.group(1) if m else None


# ─────────────────────────────────────────────────────────────────────────────
# TIMESTAMP EXTRACTION  (best-effort)
# ─────────────────────────────────────────────────────────────────────────────

# Common timestamp patterns in log files
_TS_PATTERNS = [
    # ISO-8601: 2024-01-15T10:30:00 / 2024-01-15 10:30:00
    (re.compile(r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})'), "%Y-%m-%dT%H:%M:%S"),
    # HDFS: 081109 203615
    (re.compile(r'^(\d{6} \d{6})'), "%y%m%d %H%M%S"),
    # syslog: Jan 15 10:30:00
    (re.compile(r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})'), "%b %d %H:%M:%S"),
]

def extract_timestamp(log_text: str) -> datetime | None:
    for pattern, fmt in _TS_PATTERNS:
        m = pattern.search(log_text)
        if m:
            ts_str = m.group(1).replace("T", " ")
            try:
                dt = datetime.strptime(ts_str, fmt.replace("T", " "))
                # HDFS logs have 2-digit year — assume 2000s
                if dt.year < 100:
                    dt = dt.replace(year=dt.year + 2000)
                return dt
            except ValueError:
                continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def temporal_analysis(results: list[dict], window_seconds: int = 60) -> dict:
    """
    Detect repeated failures from the same IP within a time window.

    Parameters
    ----------
    results        : list of prediction records (with 'log', 'is_attack', 'ip')
    window_seconds : sliding window size in seconds

    Returns
    -------
    dict keyed by IP → {attack_count, total_events, flagged}
    """
    ip_events = defaultdict(list)  # ip → list of (timestamp or int index, is_attack)

    for i, r in enumerate(results):
        ip = r.get("ip")
        if not ip:
            continue
        ts = r.get("timestamp")
        # Only use timestamp if it is a real datetime object; otherwise fall back to
        # the loop index (guaranteed int) so the index branch always gets integers.
        key = ts if isinstance(ts, datetime) else i
        ip_events[ip].append((key, r.get("is_attack", False)))

    temporal = {}
    for ip, events in ip_events.items():
        if not ip:
            continue

        # ── Attack event filtering ────────────────────────────────────────────
        # Accept strict booleans, integer 1, or float scores >= 0.5
        attack_events = []
        for k, a in events:
            try:
                if (
                    a is True
                    or a == 1
                    or (isinstance(a, (int, float)) and not isinstance(a, bool) and a >= 0.3)
                ):
                    attack_events.append((k, a))
            except Exception:
                continue

        attack_count = len(attack_events)
        flagged = False
        burst_detected = False

        # ── RULE 1: Frequency detection ───────────────────────────────────────
        # Threshold = 2: flag any IP responsible for 2+ distinct attack logs.
        # Threshold of 5 was too strict for this dataset — many HDFS logs lack
        # an extractable IP, so a 7-log incident may only contribute 2-3 counts
        # here even though the IP is genuinely malicious.
        if attack_count >= 2:
            flagged = True

        # ── RULE 2: Burst detection ───────────────────────────────────────────
        keys = [e[0] for e in attack_events]

        if keys:
            # TIMESTAMP MODE
            if isinstance(keys[0], datetime):
                try:
                    span = (max(keys) - min(keys)).total_seconds()
                    if span <= window_seconds and attack_count >= 2:
                        flagged = True
                        burst_detected = True
                except Exception:
                    pass

            # INDEX MODE
            else:
                int_indices = []
                for k in keys:
                    try:
                        int_indices.append(int(k))
                    except (TypeError, ValueError):
                        continue

                if len(int_indices) >= 2:
                    indices = sorted(int_indices)
                    # Window of 100 lines: HDFS logs for the same host are
                    # spread across hundreds of lines in the file
                    for j in range(len(indices) - 1):
                        if indices[j + 1] - indices[j] <= 100:
                            flagged = True
                            burst_detected = True
                            break

        # ── Final output ──────────────────────────────────────────────────────
        temporal[ip] = {
            "attack_count": attack_count,
            "total_events": len(events),
            "flagged": flagged,
            "burst": burst_detected,
        }

    return temporal
# ─────────────────────────────────────────────────────────────────────────────
# INCIDENT CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

def correlate_incidents(results: list[dict], proximity: int = 5) -> list[dict]:
    """
    Group related attack logs into incidents.

    Grouping logic:
      - Same IP  → same incident (regardless of distance)
      - Adjacent attacks within `proximity` lines → same incident

    Returns
    -------
    list of incidents: {incident_id, log_indices, ip, num_logs, risk_level, summary}
    """
    attack_indices = [i for i, r in enumerate(results) if r.get("is_attack")]
    if not attack_indices:
        return []

    # Group by proximity
    groups: list[list[int]] = []
    current = [attack_indices[0]]

    for idx in attack_indices[1:]:
        if idx - current[-1] <= proximity:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)

    # Merge groups sharing the same IP
    ip_to_group: dict[str, int] = {}
    merged: list[list[int]] = []

    for group in groups:
        ips = {results[i].get("ip") for i in group if results[i].get("ip")}
        found = None
        for ip in ips:
            if ip in ip_to_group:
                found = ip_to_group[ip]
                break
        if found is not None:
            merged[found].extend(group)
            for ip in ips:
                ip_to_group[ip] = found
        else:
            gidx = len(merged)
            merged.append(group[:])
            for ip in ips:
                ip_to_group[ip] = gidx

    # Build incident records
    incidents = []
    for inc_id, indices in enumerate(merged, start=1):
        indices = sorted(set(indices))
        scores  = [results[i].get("final_score", 0.0) for i in indices]
        max_score = max(scores) if scores else 0.0
        risk = map_risk_level(max_score)

        ips = [results[i].get("ip") for i in indices if results[i].get("ip")]
        primary_ip = max(set(ips), key=ips.count) if ips else None

        types = [results[i].get("attack_type", "—") for i in indices
                 if results[i].get("attack_type", "—") != "—"]
        primary_type = max(set(types), key=types.count) if types else "Unknown"

        incidents.append({
            "incident_id": inc_id,
            "log_indices": indices,
            "num_logs": len(indices),
            "primary_ip": primary_ip,
            "risk_level": risk,
            "max_score": round(max_score, 3),
            "attack_type": primary_type,
            "summary": (
                f"{primary_type} involving {len(indices)} log(s)"
                + (f" from {primary_ip}" if primary_ip else "")
            ),
        })

    return incidents


# ─────────────────────────────────────────────────────────────────────────────
# LOG FILE PARSER  (Streamlit upload helper)
# ─────────────────────────────────────────────────────────────────────────────

def parse_upload(file) -> list[str]:
    """Parse an uploaded file (Streamlit UploadedFile) into a list of log lines."""
    raw = file.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("latin-1")

    fname = getattr(file, "name", "")
    if fname.endswith(".csv"):
        df = pd.read_csv(io.StringIO(content))
        log_col = None
        for col in df.columns:
            if any(k in col.lower() for k in ["log", "message", "content", "event", "text", "line"]):
                log_col = col
                break
        log_col = log_col or df.columns[0]
        return df[log_col].dropna().astype(str).tolist()

    return [ln.strip() for ln in content.splitlines() if ln.strip()]