"""
app.py — SOC Log Analyzer — Streamlit Dashboard
================================================
Run: streamlit run app.py
"""

import os
import traceback

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from soc_engine import (
    FEATURE_NAMES,
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
    parse_upload,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SOC Log Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# MINIMAL CSS  (dark theme, readable)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg: #0d1117; --surf: #161b22; --border: #30363d;
  --accent: #58a6ff; --green: #3fb950; --yellow: #d29922;
  --red: #f85149; --orange: #db6d28; --text: #e6edf3; --muted: #8b949e;
  --mono: 'JetBrains Mono', monospace;
}

html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1400px !important; }

.metric-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.6rem; margin-bottom: 1rem; }
.metric-tile { background: var(--surf); border: 1px solid var(--border); border-radius: 8px; padding: 0.9rem; text-align: center; }
.metric-val  { font-family: var(--mono); font-size: 1.8rem; font-weight: 700; display: block; }
.metric-lbl  { font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }

.badge { display: inline-block; padding: 2px 8px; border-radius: 20px; font-family: var(--mono); font-size: 0.67rem; font-weight: 600; }
.b-atk  { background: rgba(248,81,73,.12);  color: var(--red);    border: 1px solid rgba(248,81,73,.3); }
.b-nrm  { background: rgba(63,185,80,.12);  color: var(--green);  border: 1px solid rgba(63,185,80,.3); }
.b-crit { background: rgba(248,81,73,.18);  color: var(--red);    border: 1px solid rgba(248,81,73,.4); }
.b-high { background: rgba(219,109,40,.12); color: var(--orange); border: 1px solid rgba(219,109,40,.3); }
.b-med  { background: rgba(210,153,34,.12); color: var(--yellow); border: 1px solid rgba(210,153,34,.3); }
.b-low  { background: rgba(63,185,80,.12);  color: var(--green);  border: 1px solid rgba(63,185,80,.3); }

.log-row { background: var(--surf); border: 1px solid var(--border); border-radius: 6px; padding: 0.65rem 1rem; margin-bottom: 0.35rem; }
.log-atk { border-left: 3px solid var(--red); }
.log-nrm { border-left: 3px solid var(--green); }
.log-meta { display: flex; gap: 0.4rem; align-items: center; flex-wrap: wrap; margin-bottom: 0.25rem; }
.log-text { font-family: var(--mono); font-size: 0.70rem; color: var(--muted); word-break: break-all; }
.log-why  { font-size: 0.73rem; color: #8b949e; margin-top: 0.2rem; font-style: italic; }

.score-bar { display: inline-flex; align-items: center; gap: 0.3rem; margin-left: auto; }
.bar-bg    { background: var(--border); border-radius: 3px; height: 5px; width: 55px; }
.bar-fill  { height: 5px; border-radius: 3px; }

.incident-card { background: var(--surf); border: 1px solid var(--border); border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.5rem; }
.section-label { font-family: var(--mono); font-size: 0.70rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1.5px; border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; margin: 1rem 0 0.6rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH   = "model.pkl"
DEFAULT_THR  = 0.40
PAGE_SIZE    = 30

RISK_ICON   = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "✅"}
RISK_CLASS  = {"CRITICAL": "b-crit", "HIGH": "b-high", "MEDIUM": "b-med", "LOW": "b-low"}
RISK_COLOR  = {"CRITICAL": "#f85149", "HIGH": "#db6d28", "MEDIUM": "#d29922", "LOW": "#3fb950"}

DEMO_LOGS = [
    "INFO 2024-01-15 10:00:01 User alice logged in from 192.168.1.10",
    "INFO 2024-01-15 10:00:03 GET /api/status 200 OK from 10.0.0.1",
    "WARN 2024-01-15 10:01:00 Multiple failed SSH login attempts from 10.0.0.5",
    "ERROR 2024-01-15 10:01:02 Authentication failure for user root from 10.0.0.5 (attempt 4)",
    "ERROR 2024-01-15 10:01:05 Authentication failure for user root from 10.0.0.5 (attempt 5)",
    "081109 204000 35 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating",
    "081109 204005 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.73.220:50010 is added to blk_-2641638092720413974",
    "081109 204655 556 INFO dfs.DataNode$PacketResponder: Received block blk_3587508140051953248 of size 67108864 from /10.251.30.6",
    "081109 204903 678 ERROR dfs.DataNode$DataXceiver: writeBlock blk_-7049066296714776736 received exception java.io.IOException: Broken pipe",
    "INFO 2024-01-15 10:02:00 Backup job completed successfully",
    "ERROR 2024-01-15 10:03:00 SQL injection detected from 192.168.1.99: ' OR 1=1--",
    "CRITICAL 2024-01-15 10:03:01 Malware detected: ransomware.wannacry variant from 192.168.1.99",
    "INFO 2024-01-15 10:04:00 TLS certificate renewed — valid for 90 days",
    "WARN 2024-01-15 10:05:00 Disk usage at 87% on /dev/sda1 — approaching limit",
]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CACHE
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, DEFAULT_THR
    pkg = joblib.load(MODEL_PATH)
    return pkg["model"], pkg["threshold"]


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION PIPELINE  (matches predict.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(log_lines: list[str], model, threshold: float) -> tuple:
    feat_df  = extract_features_df(log_lines)
    ml_proba = model.predict_proba(feat_df)[:, 1] if model else np.zeros(len(log_lines))

    results = []
    for i, line in enumerate(log_lines):
        features  = feat_df.iloc[i].to_dict()
        ml_prob   = float(ml_proba[i])
        r_score, fired = rule_score(features)
        h_score   = hybrid_score(ml_prob, r_score)
        is_attack = h_score >= threshold
        risk      = map_risk_level(h_score)
        atype     = detect_attack_type(features, is_attack)
        summary   = generate_summary(features, fired, is_attack)
        ip        = extract_ip(line)
        ts        = extract_timestamp(line)

        results.append({
            "index":       i,
            "log":         line,
            "is_attack":   is_attack,
            "ml_prob":     round(ml_prob, 4),
            "rule_score":  round(r_score, 4),
            "final_score": round(h_score, 4),
            "risk_level":  risk,
            "attack_type": atype,
            "summary":     summary,
            "fired_rules": fired,
            "ip":          ip,
            "timestamp":   ts,
        })

    temporal  = temporal_analysis(results)
    suspicious_ips = [
    ip for ip, data in temporal.items()
    if data.get("flagged")
]
    print(f"{len(suspicious_ips)} suspicious IP(s)")
    incidents = correlate_incidents(results)
    return results, incidents, temporal


# ─────────────────────────────────────────────────────────────────────────────
# HTML HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def esc(s: str) -> str:
    return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")


def render_log_row(r: dict) -> str:
    is_atk  = r["is_attack"]
    risk    = r["risk_level"]
    pct     = int(r["final_score"] * 100)
    color   = RISK_COLOR.get(risk, "#8b949e")
    row_cls = "log-atk" if is_atk else "log-nrm"
    lbl_cls = "b-atk" if is_atk else "b-nrm"
    lbl_txt = "ATTACK" if is_atk else "NORMAL"
    risk_cls = RISK_CLASS.get(risk, "b-low")

    atype_html = ""
    if is_atk and r["attack_type"] != "—":
        atype_html = (
            f'<span style="font-size:0.68rem;color:var(--orange);font-family:var(--mono)">'
            f'⚡ {esc(r["attack_type"])}</span>'
        )

    ip_html = ""
    if r["ip"]:
        ip_html = (
            f'<span style="font-size:0.67rem;color:var(--muted);font-family:var(--mono)">'
            f'📍 {esc(r["ip"])}</span>'
        )

    return (
        f'<div class="log-row {row_cls}">'
        f'<div class="log-meta">'
        f'<span class="badge {lbl_cls}">{lbl_txt}</span>'
        f'<span class="badge {risk_cls}">{RISK_ICON.get(risk,"")} {risk}</span>'
        f'{atype_html}{ip_html}'
        f'<div class="score-bar" style="margin-left:auto">'
        f'<span style="font-family:var(--mono);font-size:0.67rem;color:var(--muted)">{pct}%</span>'
        f'<div class="bar-bg"><div class="bar-fill" style="width:{pct}%;background:{color}"></div></div>'
        f'</div>'
        f'</div>'
        f'<div class="log-text">{esc(r["log"][:200])}</div>'
        f'<div class="log-why">💬 {esc(r["summary"])}</div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="display:flex;align-items:center;gap:1rem;padding:0.8rem 0;
  border-bottom:1px solid #30363d;margin-bottom:1.2rem">
  <span style="font-size:2rem">🛡️</span>
  <div>
    <h1 style="font-family:'JetBrains Mono',monospace;font-size:1.5rem;
      color:#58a6ff;margin:0">SOC Log Analyzer</h1>
    <p style="color:#8b949e;font-size:0.75rem;margin:0;text-transform:uppercase;
      letter-spacing:0.5px">Hybrid AI · Temporal Correlation · Incident Detection</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    threshold = st.slider("Detection Threshold", 0.20, 0.80, DEFAULT_THR, 0.05,
                          help="Hybrid score cutoff for ATTACK classification")
    page_size = st.selectbox("Rows per page", [20, 30, 50, 100], index=1)

    st.markdown("---")
    st.markdown("### 📂 Model Status")
    if os.path.exists(MODEL_PATH):
        pkg = joblib.load(MODEL_PATH)
        st.success(f"Model loaded ✅")
        st.caption(f"Accuracy: {pkg['accuracy']}%  |  Recall: {pkg['recall']}%")
        st.caption(f"Default threshold: {pkg['threshold']}")
    else:
        st.warning("No model found.\nRun: `python train.py`")

    st.markdown("---")
    st.markdown("### ℹ️ Scoring Formula")
    st.caption("final = 0.6 × ML_prob + 0.4 × rule_score")
    st.caption("CRITICAL ≥ 0.75 | HIGH ≥ 0.50")
    st.caption("MEDIUM ≥ 0.25  | LOW < 0.25")


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD + ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

col_up, col_btn = st.columns([3, 1])
with col_up:
    uploaded = st.file_uploader(
        "Upload log file (.log / .txt / .csv)",
        type=["log", "txt", "csv"],
        label_visibility="collapsed",
    )
with col_btn:
    analyze = st.button("🔍 Analyze Logs", use_container_width=True)

if "results"   not in st.session_state: st.session_state.results   = None
if "incidents" not in st.session_state: st.session_state.incidents = None
if "temporal"  not in st.session_state: st.session_state.temporal  = None
if "log_lines" not in st.session_state: st.session_state.log_lines = None

if analyze:
    model, default_thr = load_model()
    if model is None:
        st.error(
            "⚠️ No trained model found. Run `python train.py` first.\n\n"
            "The dashboard will still show rule-based scores while ML prob = 0."
        )

    with st.spinner("Running detection pipeline..."):
        try:
            if uploaded:
                log_lines = parse_upload(uploaded)
            else:
                st.info("No file uploaded — running on built-in demo logs.")
                log_lines = DEMO_LOGS

            # If no model, use a stub
            if model is None:
                from sklearn.dummy import DummyClassifier
                import numpy as np
                class RuleOnlyModel:
                    def predict_proba(self, X):
                        return np.zeros((len(X), 2))
                model = RuleOnlyModel()

            results, incidents, temporal = run_detection(log_lines, model, threshold)
            st.session_state.results   = results
            st.session_state.incidents = incidents
            st.session_state.temporal  = temporal
            st.session_state.log_lines = log_lines

        except Exception:
            st.error(f"Detection error:\n```\n{traceback.format_exc()}\n```")
            st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.results:
    results   = st.session_state.results
    incidents = st.session_state.incidents
    temporal  = st.session_state.temporal

    total    = len(results)
    n_atk    = sum(1 for r in results if r["is_attack"])
    n_nrm    = total - n_atk
    by_risk  = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in results:
        by_risk[r["risk_level"]] += 1

    # ── Metrics ────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="metric-grid">'
        f'<div class="metric-tile"><span class="metric-val" style="color:#58a6ff">{total}</span><span class="metric-lbl">Total Logs</span></div>'
        f'<div class="metric-tile"><span class="metric-val" style="color:#f85149">{n_atk}</span><span class="metric-lbl">Anomalies</span></div>'
        f'<div class="metric-tile"><span class="metric-val" style="color:#3fb950">{n_nrm}</span><span class="metric-lbl">Normal</span></div>'
        f'<div class="metric-tile"><span class="metric-val" style="color:#f85149">{by_risk["CRITICAL"] + by_risk["HIGH"]}</span><span class="metric-lbl">High+ Risk</span></div>'
        f'<div class="metric-tile"><span class="metric-val" style="color:#58a6ff">{len(incidents)}</span><span class="metric-lbl">Incidents</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Risk distribution bar ─────────────────────────────────────────────────
    pct_atk = round(n_atk / total * 100) if total else 0
    pct_nrm = 100 - pct_atk
    st.markdown(
        f'<div style="background:var(--surf);border:1px solid var(--border);border-radius:8px;padding:0.8rem 1rem;margin-bottom:0.8rem">'
        f'<div style="font-family:var(--mono);font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:0.5rem">Threat Distribution</div>'
        f'<div style="display:flex;gap:0.5rem;align-items:center;font-size:0.74rem;margin-bottom:0.3rem">'
        f'<span style="font-family:var(--mono);width:100px;color:#f85149">ATTACK {pct_atk}%</span>'
        f'<div style="flex:1;background:var(--border);border-radius:4px;height:7px">'
        f'<div style="width:{pct_atk}%;height:7px;border-radius:4px;background:#f85149"></div></div></div>'
        f'<div style="display:flex;gap:0.5rem;align-items:center;font-size:0.74rem">'
        f'<span style="font-family:var(--mono);width:100px;color:#3fb950">NORMAL {pct_nrm}%</span>'
        f'<div style="flex:1;background:var(--border);border-radius:4px;height:7px">'
        f'<div style="width:{pct_nrm}%;height:7px;border-radius:4px;background:#3fb950"></div></div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Incidents ─────────────────────────────────────────────────────────────
    if incidents:
        st.markdown('<div class="section-label">📋 Incident Summary</div>', unsafe_allow_html=True)
        inc_cols = st.columns(min(len(incidents), 3))
        for idx, inc in enumerate(incidents[:6]):
            with inc_cols[idx % 3]:
                icon  = RISK_ICON.get(inc["risk_level"], "")
                color = RISK_COLOR.get(inc["risk_level"], "#8b949e")
                ip_html = f'<div style="font-size:0.68rem;color:#58a6ff">📍 {inc["primary_ip"]}</div>' if inc["primary_ip"] else ""
                st.markdown(
                    f'<div class="incident-card" style="border-left:3px solid {color}">'
                    f'<div style="font-family:var(--mono);font-size:0.75rem;font-weight:600">'
                    f'{icon} INC-{inc["incident_id"]:03d} · {inc["risk_level"]}</div>'
                    f'<div style="font-size:0.70rem;color:var(--muted);margin-top:0.2rem">'
                    f'{inc["num_logs"]} log(s) · {inc["attack_type"]}</div>'
                    f'{ip_html}'
                    f'<div style="font-size:0.68rem;color:var(--muted);margin-top:0.2rem;font-style:italic">'
                    f'Score: {inc["max_score"]*100:.0f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Suspicious IPs ────────────────────────────────────────────────────────
    flagged = {ip: info for ip, info in temporal.items() if info["flagged"]}
    if flagged:
        st.markdown('<div class="section-label">🚨 Suspicious IPs (Temporal Analysis)</div>', unsafe_allow_html=True)
        ip_data = sorted(flagged.items(), key=lambda x: x[1]["attack_count"], reverse=True)
        ip_cols = st.columns(min(len(ip_data), 4))
        for idx, (ip, info) in enumerate(ip_data[:4]):
            with ip_cols[idx % 4]:
                st.markdown(
                    f'<div class="metric-tile">'
                    f'<span class="metric-val" style="color:#f85149;font-size:1.1rem">{ip}</span>'
                    f'<span class="metric-lbl">{info["attack_count"]} attacks detected</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Filters ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">📋 Log Analysis Results</div>', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4 = st.columns([1, 1, 1, 1])
    with fc1:
        f_label = st.selectbox("Classification", ["All", "ATTACK", "NORMAL"])
    with fc2:
        f_risk  = st.selectbox("Risk Level", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    with fc3:
        all_types = sorted(set(r["attack_type"] for r in results))
        f_type  = st.selectbox("Attack Type", ["All"] + all_types)
    with fc4:
        sort_by = st.selectbox("Sort By", ["Index", "Score ↓", "Risk ↓"])

    # Apply filters
    filtered = results[:]
    if f_label != "All":
        filtered = [r for r in filtered if (r["is_attack"] and f_label == "ATTACK") or
                                           (not r["is_attack"] and f_label == "NORMAL")]
    if f_risk != "All":
        filtered = [r for r in filtered if r["risk_level"] == f_risk]
    if f_type != "All":
        filtered = [r for r in filtered if r["attack_type"] == f_type]

    # Sort
    if sort_by == "Score ↓":
        filtered = sorted(filtered, key=lambda r: r["final_score"], reverse=True)
    elif sort_by == "Risk ↓":
        order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        filtered = sorted(filtered, key=lambda r: order.get(r["risk_level"], 9))

    st.caption(f"Showing {len(filtered)} of {total} entries")

    # ── Pagination ────────────────────────────────────────────────────────────
    total_pages = max(1, (len(filtered) - 1) // page_size + 1)
    if total_pages > 1:
        page = st.number_input("Page", 1, total_pages, 1, step=1)
    else:
        page = 1
    start   = (page - 1) * page_size
    page_rs = filtered[start: start + page_size]

    for r in page_rs:
        st.markdown(render_log_row(r), unsafe_allow_html=True)

    if total_pages > 1:
        st.caption(f"Page {page} of {total_pages}")

    # ── Export ────────────────────────────────────────────────────────────────
    with st.expander("📊 Export / Raw Table", expanded=False):
        df_export = pd.DataFrame([{
            "index":       r["index"],
            "log":         r["log"],
            "is_attack":   r["is_attack"],
            "ml_prob":     r["ml_prob"],
            "rule_score":  r["rule_score"],
            "final_score": r["final_score"],
            "risk_level":  r["risk_level"],
            "attack_type": r["attack_type"],
            "summary":     r["summary"],
            "ip":          r["ip"],
        } for r in results])

        st.dataframe(df_export, use_container_width=True, height=280)
        st.download_button(
            "⬇️ Download CSV",
            df_export.to_csv(index=False).encode("utf-8"),
            "soc_results.csv", "text/csv",
        )

else:
    # ── Landing state ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:4rem 0;color:#8b949e">
      <div style="font-size:3rem;margin-bottom:1rem">🛡️</div>
      <p style="font-size:1rem;font-family:'JetBrains Mono',monospace;color:#e6edf3">
        Upload a log file and click <span style="color:#58a6ff">Analyze Logs</span>
      </p>
      <p style="font-size:0.8rem;margin-top:0.4rem">
        Supports .log / .txt (raw logs) and .csv (with a log/message column)
      </p>
      <p style="font-size:0.78rem;margin-top:0.3rem;color:#58a6ff">
        No file? Click Analyze Logs to run on built-in demo data.
      </p>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:2rem;padding-top:0.8rem;border-top:1px solid #30363d;
  text-align:center;font-size:0.67rem;color:#8b949e;font-family:'JetBrains Mono',monospace">
  SOC Log Analyzer · Hybrid Detection (RF + Rules) · Temporal Analysis · Incident Correlation
</div>
""", unsafe_allow_html=True)