"""
dashboard.py — SOC Threat Intelligence Dashboard
==================================================
High-end Streamlit UI for the SOC Log Analyzer system.

Run:
    streamlit run dashboard.py

Requires:
    streamlit, pandas, matplotlib, joblib
    soc_engine.py, predict.py, model.pkl  (in same directory)
"""

import io
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from datetime import datetime

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="SOC Threat Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Try importing backend modules ─────────────────────────────────────────────
try:
    from soc_engine import (
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
    from predict import run as run_pipeline, DEMO_LOGS
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Dark Cyberpunk Theme
# ─────────────────────────────────────────────────────────────────────────────
DARK_CSS = """
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@400;700;900&display=swap');

/* ── Root Variables ──────────────────────────────────────────────────────── */
:root {
    --bg-base:       #020b18;
    --bg-panel:      #050f1f;
    --bg-card:       #071526;
    --bg-row-alt:    #081930;
    --border-dim:    #0d2a45;
    --border-bright: #0e4a7a;
    --accent-blue:   #00aaff;
    --accent-cyan:   #00f5ff;
    --glow-blue:     0 0 8px #00aaff88, 0 0 20px #00aaff33;
    --glow-cyan:     0 0 8px #00f5ff88, 0 0 20px #00f5ff33;
    --clr-critical:  #ff2244;
    --clr-high:      #ff7700;
    --clr-medium:    #ffcc00;
    --clr-low:       #00ee88;
    --glow-critical: 0 0 8px #ff224488, 0 0 18px #ff224433;
    --glow-high:     0 0 8px #ff770088, 0 0 18px #ff770033;
    --glow-medium:   0 0 8px #ffcc0088, 0 0 18px #ffcc0033;
    --glow-low:      0 0 8px #00ee8888, 0 0 18px #00ee8833;
    --text-primary:  #c8e8ff;
    --text-dim:      #4a7a9b;
    --text-label:    #2a6a9a;
    --font-mono:     'Share Tech Mono', monospace;
    --font-ui:       'Rajdhani', sans-serif;
    --font-display:  'Orbitron', sans-serif;
}

/* ── App Shell ───────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-ui) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border-dim) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Main block padding ──────────────────────────────────────────────────── */
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 100% !important; }

/* ── Header ──────────────────────────────────────────────────────────────── */
.soc-header {
    background: linear-gradient(135deg, #020d1f 0%, #030e22 50%, #020c1d 100%);
    border: 1px solid var(--border-bright);
    border-radius: 4px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 30px #001a3388, inset 0 1px 0 #0e4a7a55;
}
.soc-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
    animation: scan-line 3s linear infinite;
}
@keyframes scan-line {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}
.soc-title {
    font-family: var(--font-display) !important;
    font-size: 1.8rem !important;
    font-weight: 900 !important;
    color: var(--accent-cyan) !important;
    text-shadow: var(--glow-cyan);
    letter-spacing: 3px !important;
    margin: 0 !important;
    line-height: 1.2 !important;
}
.soc-subtitle {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    color: var(--text-dim) !important;
    letter-spacing: 2px !important;
    margin-top: 4px !important;
}
.soc-status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--clr-low);
    box-shadow: var(--glow-low);
    animation: pulse 1.8s ease-in-out infinite;
    margin-right: 6px;
    vertical-align: middle;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.85); }
}
.soc-ts {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--text-dim) !important;
    margin-top: 8px !important;
}

/* ── Metric Cards ────────────────────────────────────────────────────────── */
.metric-card {
    background: var(--bg-card);
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    border: 1px solid var(--border-dim);
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0; height: 2px;
}
.metric-card.blue::after  { background: var(--accent-blue); box-shadow: var(--glow-blue); }
.metric-card.red::after   { background: var(--clr-critical); box-shadow: var(--glow-critical); }
.metric-card.orange::after { background: var(--clr-high); box-shadow: var(--glow-high); }
.metric-card.green::after { background: var(--clr-low); box-shadow: var(--glow-low); }

.metric-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-dim);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-value {
    font-family: var(--font-display);
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-value.blue   { color: var(--accent-blue);  text-shadow: var(--glow-blue);  }
.metric-value.red    { color: var(--clr-critical); text-shadow: var(--glow-critical); }
.metric-value.orange { color: var(--clr-high);     text-shadow: var(--glow-high);   }
.metric-value.green  { color: var(--clr-low);      text-shadow: var(--glow-low);    }
.metric-sub {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--text-dim);
}

/* ── Section Headers ─────────────────────────────────────────────────────── */
.section-header {
    font-family: var(--font-display);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 3px;
    color: var(--accent-blue);
    text-shadow: var(--glow-blue);
    text-transform: uppercase;
    padding: 0.6rem 0 0.6rem 0;
    border-bottom: 1px solid var(--border-dim);
    margin-bottom: 1rem;
}
.section-header .icon { margin-right: 8px; }

/* ── Incident Table ──────────────────────────────────────────────────────── */
.inc-table { width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: 0.76rem; }
.inc-table th {
    background: #03101f;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 0.62rem;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border-bright);
    text-align: left;
    font-weight: 400;
}
.inc-table td {
    padding: 7px 12px;
    border-bottom: 1px solid var(--border-dim);
    color: var(--text-primary);
    vertical-align: middle;
}
.inc-table tr:nth-child(even) td { background: var(--bg-row-alt); }
.inc-table tr.critical-row td { background: #1a020622 !important; border-left: 2px solid var(--clr-critical); }
.inc-table tr.critical-row td:first-child { padding-left: 10px; }
.inc-table tr:hover td { background: #0a1f35 !important; }

.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 2px;
    font-size: 0.65rem;
    letter-spacing: 1px;
    font-weight: 600;
    font-family: var(--font-mono);
}
.badge.CRITICAL { background: #ff224420; color: var(--clr-critical); border: 1px solid var(--clr-critical); box-shadow: var(--glow-critical); }
.badge.HIGH     { background: #ff770020; color: var(--clr-high);     border: 1px solid var(--clr-high);     box-shadow: var(--glow-high);     }
.badge.MEDIUM   { background: #ffcc0020; color: var(--clr-medium);   border: 1px solid var(--clr-medium);   box-shadow: var(--glow-medium);   }
.badge.LOW      { background: #00ee8820; color: var(--clr-low);      border: 1px solid var(--clr-low);      box-shadow: var(--glow-low);      }

/* ── Suspicious IP Cards ─────────────────────────────────────────────────── */
.ip-card {
    background: var(--bg-card);
    border: 1px solid var(--border-dim);
    border-left: 3px solid var(--clr-critical);
    border-radius: 3px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-family: var(--font-mono);
    position: relative;
}
.ip-card.burst { border-left-color: var(--clr-high); box-shadow: -4px 0 12px #ff770033; }
.ip-card .ip-addr { color: var(--clr-critical); font-size: 0.85rem; font-weight: 700; }
.ip-card.burst .ip-addr { color: var(--clr-high); }
.ip-card .ip-meta { font-size: 0.68rem; color: var(--text-dim); margin-top: 3px; }
.ip-card .burst-tag {
    position: absolute;
    right: 10px; top: 50%;
    transform: translateY(-50%);
    font-size: 0.6rem;
    background: #ff770022;
    color: var(--clr-high);
    border: 1px solid var(--clr-high);
    padding: 2px 6px;
    border-radius: 2px;
    letter-spacing: 1px;
}
.ip-alert-icon { color: var(--clr-critical); margin-right: 6px; }

/* ── Log Table ───────────────────────────────────────────────────────────── */
.log-table { width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: 0.72rem; }
.log-table th {
    background: #03101f;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 0.6rem;
    padding: 6px 10px;
    border-bottom: 1px solid var(--border-bright);
    text-align: left;
    font-weight: 400;
}
.log-table td {
    padding: 5px 10px;
    border-bottom: 1px solid #0a1f30;
    color: #8aadcc;
    vertical-align: top;
    max-width: 360px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.log-table tr:hover td { background: #071525 !important; }
.log-text { color: #4a7a9b; font-size: 0.68rem; }
.score-bar-wrap { display: inline-block; width: 60px; height: 4px; background: #0d1f2e; border-radius: 2px; vertical-align: middle; }
.score-bar      { height: 4px; border-radius: 2px; display: block; }

/* ── Streamlit native overrides ──────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #001a33, #002a50) !important;
    color: var(--accent-cyan) !important;
    border: 1px solid var(--accent-blue) !important;
    border-radius: 3px !important;
    font-family: var(--font-display) !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    padding: 0.55rem 1.4rem !important;
    box-shadow: var(--glow-blue) !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #002a50, #003a70) !important;
    box-shadow: 0 0 16px #00aaff99, 0 0 32px #00aaff44 !important;
}
.stFileUploader > div {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border-bright) !important;
    border-radius: 4px !important;
}
.stSelectbox > div > div, .stSlider > div { background: transparent !important; }
.stSelectbox label, .stSlider label, .stFileUploader label {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    color: var(--text-dim) !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricLabel"]  { color: var(--text-dim) !important; }
[data-testid="stMetricValue"]  { color: var(--accent-cyan) !important; font-family: var(--font-display) !important; }
.stAlert { background: var(--bg-card) !important; border-radius: 3px !important; }
div[data-testid="stExpander"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 3px !important;
}
.stSpinner > div { border-top-color: var(--accent-cyan) !important; }
hr { border-color: var(--border-dim) !important; }
p, li { color: var(--text-primary) !important; font-family: var(--font-ui) !important; }
h1, h2, h3 { font-family: var(--font-display) !important; color: var(--accent-cyan) !important; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

RISK_COLOR = {
    "CRITICAL": "#ff2244",
    "HIGH":     "#ff7700",
    "MEDIUM":   "#ffcc00",
    "LOW":      "#00ee88",
}

RISK_ICON = {
    "CRITICAL": "🔴",
    "HIGH":     "🟠",
    "MEDIUM":   "🟡",
    "LOW":      "🟢",
}

MPL_BG  = "#020b18"
MPL_TEXT = "#4a7a9b"
MPL_GRID = "#0d2a45"


def risk_badge(level: str) -> str:
    return f'<span class="badge {level}">{level}</span>'


def score_bar(score: float, color: str) -> str:
    pct = int(score * 100)
    return (
        f'<span class="score-bar-wrap">'
        f'<span class="score-bar" style="width:{pct}%;background:{color};'
        f'box-shadow:0 0 4px {color}66;"></span></span> '
        f'<span style="font-size:0.68rem;color:{color};">{pct}%</span>'
    )


def _mpl_dark_axes(ax, title=""):
    ax.set_facecolor(MPL_BG)
    ax.tick_params(colors=MPL_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(MPL_GRID)
    ax.yaxis.label.set_color(MPL_TEXT)
    ax.xaxis.label.set_color(MPL_TEXT)
    if title:
        ax.set_title(title, color="#00aaff", fontsize=9, fontweight="bold",
                     fontfamily="monospace", pad=10)
    ax.grid(True, color=MPL_GRID, linewidth=0.5, linestyle="--", alpha=0.6)


def build_risk_chart(by_risk: dict):
    labels = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    values = [by_risk.get(l, 0) for l in labels]
    colors = [RISK_COLOR[l] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), facecolor=MPL_BG)
    fig.subplots_adjust(wspace=0.35, left=0.07, right=0.97, top=0.85, bottom=0.12)

    # Bar chart
    ax = axes[0]
    ax.set_facecolor(MPL_BG)
    bars = ax.bar(labels, values, color=colors, width=0.55,
                  linewidth=0, zorder=3)
    for bar, col, val in zip(bars, colors, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=8,
                color=col, fontfamily="monospace", fontweight="bold")
    _mpl_dark_axes(ax, "RISK LEVEL DISTRIBUTION")
    ax.set_xlabel("")
    ax.set_ylabel("Log Count", fontsize=7)
    ax.tick_params(axis="x", colors=MPL_TEXT, labelsize=8)

    # Add glow effect approximation with a lighter bar behind
    for bar, col in zip(bars, colors):
        gbar = ax.bar(bar.get_x() + bar.get_width()/2, bar.get_height(),
                      width=bar.get_width() * 1.5, color=col, alpha=0.07,
                      zorder=2, linewidth=0)

    # Donut
    ax2 = axes[1]
    ax2.set_facecolor(MPL_BG)
    non_zero = [(v, c, l) for v, c, l in zip(values, colors, labels) if v > 0]
    if non_zero:
        vals2, cols2, labs2 = zip(*non_zero)
        wedges, _ = ax2.pie(
            vals2, colors=cols2, startangle=90,
            wedgeprops=dict(width=0.55, edgecolor=MPL_BG, linewidth=2),
        )
        total = sum(vals2)
        ax2.text(0, 0, f"{total}\nLOGS", ha="center", va="center",
                 fontsize=9, color="#00f5ff", fontfamily="monospace",
                 fontweight="bold")
        legend_patches = [
            mpatches.Patch(color=c, label=f"{l}  {v}")
            for v, c, l in zip(vals2, cols2, labs2)
        ]
        ax2.legend(handles=legend_patches, loc="lower center",
                   bbox_to_anchor=(0.5, -0.18), ncol=2,
                   fontsize=7, framealpha=0,
                   labelcolor=MPL_TEXT,
                   handlelength=1, handleheight=0.9)
        ax2.set_title("THREAT COMPOSITION", color="#00aaff", fontsize=9,
                      fontweight="bold", fontfamily="monospace", pad=10)
    return fig


def build_score_dist_chart(results: list):
    scores = [r["final_score"] for r in results]
    colors_list = [RISK_COLOR[r["risk_level"]] for r in results]

    fig, ax = plt.subplots(figsize=(9, 2.8), facecolor=MPL_BG)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.18)
    ax.set_facecolor(MPL_BG)

    n = len(scores)
    x = list(range(n))
    ax.scatter(x, scores, c=colors_list, s=4, alpha=0.6, linewidths=0, zorder=3)

    # Rolling mean line
    window = max(1, n // 50)
    series = pd.Series(scores).rolling(window, min_periods=1).mean()
    ax.plot(x, series, color="#00aaff", linewidth=0.9, alpha=0.8, zorder=4)
    ax.axhline(0.5, color="#ff2244", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(n * 0.02, 0.52, "THRESHOLD", color="#ff2244", fontsize=6,
            fontfamily="monospace", alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Hybrid Score", fontsize=7)
    ax.set_xlabel("Log Index", fontsize=7)
    _mpl_dark_axes(ax, "HYBRID ANOMALY SCORE — ALL LOGS")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DEMO DATA  (used when model is not available)
# ─────────────────────────────────────────────────────────────────────────────

def generate_demo_results(n=200):
    rng = np.random.default_rng(42)
    attack_types = ["Brute Force", "SQL Injection", "Port Scan",
                    "Malware", "Privilege Escalation", "DoS", "Data Exfil", "Normal"]
    ips = [f"10.0.{rng.integers(0,5)}.{rng.integers(1,255)}" for _ in range(n)]
    results = []
    for i in range(n):
        ml  = float(np.clip(rng.beta(1.5, 5) if rng.random() > 0.15 else rng.beta(5, 1.5), 0, 1))
        rs  = float(np.clip(rng.beta(1.2, 5) if rng.random() > 0.12 else rng.beta(4, 1.2), 0, 1))
        hs  = round(0.6 * ml + 0.4 * rs, 4)
        is_atk = hs >= 0.45
        atk_type = rng.choice(attack_types[:-1]) if is_atk else "Normal"
        rl = map_risk_level(hs) if BACKEND_AVAILABLE else (
            "CRITICAL" if hs > 0.75 else
            "HIGH"     if hs > 0.55 else
            "MEDIUM"   if hs > 0.35 else "LOW"
        )
        results.append({
            "index":       i,
            "log":         f"[{'ERROR' if is_atk else 'INFO'}] event #{i} from {ips[i]}",
            "is_attack":   is_atk,
            "ml_prob":     round(ml, 4),
            "rule_score":  round(rs, 4),
            "final_score": hs,
            "risk_level":  rl,
            "attack_type": atk_type,
            "summary":     f"{atk_type} activity detected" if is_atk else "Normal operation",
            "fired_rules": [],
            "ip":          ips[i] if is_atk else None,
            "timestamp":   None,
        })
    return results


def generate_demo_incidents(results):
    attack_res = [r for r in results if r["is_attack"]]
    if not attack_res:
        return []
    groups = {}
    for r in attack_res:
        ip = r.get("ip") or "unknown"
        groups.setdefault(ip, []).append(r)
    incidents = []
    for idx, (ip, logs) in enumerate(list(groups.items())[:20], start=1):
        max_score = max(r["final_score"] for r in logs)
        rl = ("CRITICAL" if max_score > 0.75 else
              "HIGH"     if max_score > 0.55 else
              "MEDIUM"   if max_score > 0.35 else "LOW")
        atk = logs[0]["attack_type"]
        incidents.append({
            "incident_id": idx,
            "num_logs":    len(logs),
            "primary_ip":  ip,
            "risk_level":  rl,
            "max_score":   round(max_score, 3),
            "attack_type": atk,
            "summary":     f"{atk} from {ip} ({len(logs)} events)",
        })
    return sorted(incidents, key=lambda x: x["max_score"], reverse=True)


def generate_demo_temporal(results):
    ip_map = {}
    for r in results:
        ip = r.get("ip")
        if not ip:
            continue
        if ip not in ip_map:
            ip_map[ip] = {"attack_count": 0, "total_events": 0,
                          "flagged": False, "burst": False}
        ip_map[ip]["total_events"] += 1
        if r["is_attack"]:
            ip_map[ip]["attack_count"] += 1
    for ip, d in ip_map.items():
        if d["attack_count"] >= 2:
            d["flagged"] = True
        if d["attack_count"] >= 4:
            d["burst"] = True
    return ip_map


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div style="font-family:\'Orbitron\',sans-serif;font-size:0.75rem;'
            'color:#00aaff;letter-spacing:3px;padding:0.5rem 0 1rem 0;'
            'text-shadow:0 0 8px #00aaff88;">⬡ SOC ANALYZER</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="font-family:monospace;font-size:0.62rem;color:#2a6a9a;'
            'letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">'
            'Upload Log File</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            label="log_upload",
            type=["log", "txt", "csv"],
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("⚡ RUN ANALYSIS", use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown(
            '<div style="font-family:monospace;font-size:0.62rem;color:#2a6a9a;'
            'letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">'
            'Filter by Risk Level</div>',
            unsafe_allow_html=True,
        )
        risk_filter = st.selectbox(
            "risk_filter", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
            label_visibility="collapsed",
        )

        st.markdown(
            '<div style="font-family:monospace;font-size:0.62rem;color:#2a6a9a;'
            'letter-spacing:2px;text-transform:uppercase;margin-top:1rem;margin-bottom:6px;">'
            'Logs to Display</div>',
            unsafe_allow_html=True,
        )
        log_limit = st.slider("log_limit", 10, 200, 50, label_visibility="collapsed")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-family:monospace;font-size:0.6rem;color:#1a4a6a;">'
            f'<span style="color:#00ee8888;">●</span> SYSTEM ONLINE<br>'
            f'<span style="color:#2a6a9a;">BUILD</span> v2.4.1<br>'
            f'<span style="color:#2a6a9a;">ENGINE</span> RF+RULES HYBRID<br>'
            f'<span style="color:#2a6a9a;">MODEL</span> {"LOADED ✓" if BACKEND_AVAILABLE and os.path.exists("model.pkl") else "DEMO MODE"}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── HEADER ────────────────────────────────────────────────────────────────
    now_ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S  UTC")
    st.markdown(
        f"""
        <div class="soc-header">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                    <div class="soc-title">🛡 SOC THREAT INTELLIGENCE DASHBOARD</div>
                    <div class="soc-subtitle">◈ HYBRID ML + RULE-BASED DETECTION SYSTEM  ◈  RANDOMFOREST ENGINE</div>
                    <div class="soc-ts">
                        <span class="soc-status-dot"></span>
                        LIVE MONITORING  |  {now_ts}
                    </div>
                </div>
                <div style="text-align:right;font-family:monospace;font-size:0.62rem;color:#1a4a6a;line-height:1.8;">
                    <span style="color:#00f5ff88;">DETECTION ENGINE</span><br>
                    TEMPORAL ANALYSIS ✓<br>
                    INCIDENT CORRELATION ✓<br>
                    IP THREAT TRACKING ✓
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── SESSION STATE ─────────────────────────────────────────────────────────
    if "results" not in st.session_state:
        st.session_state["results"]      = None
        st.session_state["incidents"]    = None
        st.session_state["temporal"]     = None
        st.session_state["analyzed"]     = False

    # ── RUN PIPELINE ──────────────────────────────────────────────────────────
    if run_btn:
        with st.spinner("🔍  Running detection pipeline..."):
            if uploaded_file and BACKEND_AVAILABLE:
                try:
                    pkg = joblib.load("model.pkl")
                    model, threshold = pkg["model"], pkg["threshold"]
                    log_lines = parse_upload(uploaded_file)
                    results, incidents, temporal = run_pipeline(log_lines, model, threshold)
                    st.session_state["results"]   = results
                    st.session_state["incidents"] = incidents
                    st.session_state["temporal"]  = temporal
                    st.session_state["analyzed"]  = True
                    st.success(f"✅  Analysis complete — {len(results):,} logs processed.")
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.info("Falling back to demo data.")
                    results = generate_demo_results()
                    st.session_state["results"]   = results
                    st.session_state["incidents"] = generate_demo_incidents(results)
                    st.session_state["temporal"]  = generate_demo_temporal(results)
                    st.session_state["analyzed"]  = True
            else:
                # Demo mode
                results = generate_demo_results()
                st.session_state["results"]   = results
                st.session_state["incidents"] = generate_demo_incidents(results)
                st.session_state["temporal"]  = generate_demo_temporal(results)
                st.session_state["analyzed"]  = True
                if not BACKEND_AVAILABLE:
                    st.info("ℹ️  Backend not detected — displaying demo data. Plug in soc_engine.py + model.pkl to analyze real logs.")

    # ── RENDER DASHBOARD ──────────────────────────────────────────────────────
    if not st.session_state["analyzed"]:
        # Landing state
        st.markdown(
            """
            <div style="
                text-align:center;
                padding:5rem 2rem;
                font-family:'Share Tech Mono',monospace;
            ">
                <div style="font-size:3.5rem;margin-bottom:1rem;opacity:0.3;">🛡</div>
                <div style="font-size:0.9rem;color:#0e4a7a;letter-spacing:3px;text-transform:uppercase;">
                    Upload a log file and click RUN ANALYSIS<br>or click RUN ANALYSIS to see a demo
                </div>
                <div style="margin-top:2rem;font-size:0.65rem;color:#0a2a40;letter-spacing:2px;">
                    SUPPORTS: .log  .txt  .csv
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    results   = st.session_state["results"]
    incidents = st.session_state["incidents"]
    temporal  = st.session_state["temporal"]

    # Apply risk filter
    filtered_results = results
    if risk_filter != "ALL":
        filtered_results = [r for r in results if r["risk_level"] == risk_filter]

    # ── COMPUTE STATS ─────────────────────────────────────────────────────────
    total_logs  = len(results)
    n_anomalies = sum(1 for r in results if r["is_attack"])
    n_incidents = len(incidents)
    suspicious_ips = {ip: d for ip, d in temporal.items()
                      if isinstance(d, dict) and d.get("flagged")}
    n_susp_ips = len(suspicious_ips)
    by_risk = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in results:
        rl = r.get("risk_level", "LOW")
        by_risk[rl] = by_risk.get(rl, 0) + 1
    anomaly_pct = n_anomalies / total_logs * 100 if total_logs else 0

    # ── METRIC CARDS ──────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header"><span class="icon">◈</span>SYSTEM OVERVIEW</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)

    def metric_card(container, color_cls, label, value, sub):
        container.markdown(
            f"""
            <div class="metric-card {color_cls}">
                <div class="metric-label">{label}</div>
                <div class="metric-value {color_cls}">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    metric_card(c1, "blue",   "◈ TOTAL LOGS",      f"{total_logs:,}",   "lines ingested")
    metric_card(c2, "red",    "⚠ ANOMALIES",       f"{n_anomalies:,}",  f"{anomaly_pct:.1f}% threat rate")
    metric_card(c3, "orange", "◉ INCIDENTS",        f"{n_incidents:,}",  "correlated groups")
    metric_card(c4, "green",  "⬡ SUSPICIOUS IPs",  f"{n_susp_ips:,}",   "flagged attackers")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CHARTS ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header"><span class="icon">◈</span>THREAT ANALYTICS</div>',
        unsafe_allow_html=True,
    )

    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        fig_risk = build_risk_chart(by_risk)
        st.pyplot(fig_risk, use_container_width=True)
        plt.close(fig_risk)

    with col_chart2:
        fig_dist = build_score_dist_chart(results)
        st.pyplot(fig_dist, use_container_width=True)
        plt.close(fig_dist)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── INCIDENT PANEL + SUSPICIOUS IPs  (side by side) ───────────────────────
    inc_col, ip_col = st.columns([3, 1])

    with inc_col:
        st.markdown(
            '<div class="section-header"><span class="icon">◉</span>INCIDENT CORRELATION PANEL</div>',
            unsafe_allow_html=True,
        )
        if incidents:
            rows_html = ""
            for inc in incidents:
                rl   = inc["risk_level"]
                row_cls = "critical-row" if rl == "CRITICAL" else ""
                icon = RISK_ICON.get(rl, "")
                ip_disp = inc.get("primary_ip") or "—"
                inc_id_str = f"INC-{inc['incident_id']:03d}"
                rows_html += f"""
                <tr class="{row_cls}">
                    <td style="color:#00aaff;font-weight:600;">{inc_id_str}</td>
                    <td>{risk_badge(rl)}</td>
                    <td style="color:#c8e8ff;">{inc['num_logs']}</td>
                    <td style="color:#4a7a9b;">{ip_disp}</td>
                    <td style="color:#8aadcc;">{inc.get('attack_type','—')}</td>
                    <td style="color:#2a6a9a;font-size:0.65rem;max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                        {inc.get('summary','')[:70]}
                    </td>
                </tr>
                """
            st.markdown(
                f"""
                <div style="overflow-x:auto;max-height:380px;overflow-y:auto;">
                <table class="inc-table">
                <thead>
                    <tr>
                        <th>INC ID</th>
                        <th>RISK</th>
                        <th>LOGS</th>
                        <th>IP</th>
                        <th>ATTACK TYPE</th>
                        <th>SUMMARY</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
                </table>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-family:monospace;font-size:0.75rem;color:#1a4a6a;'
                'padding:2rem;text-align:center;">NO INCIDENTS DETECTED</div>',
                unsafe_allow_html=True,
            )

    with ip_col:
        st.markdown(
            '<div class="section-header"><span class="icon">⬡</span>THREAT INTEL — IPs</div>',
            unsafe_allow_html=True,
        )
        if suspicious_ips:
            sorted_ips = sorted(
                suspicious_ips.items(),
                key=lambda x: x[1].get("attack_count", 0),
                reverse=True,
            )
            cards_html = ""
            for ip_addr, info in sorted_ips[:12]:
                burst = info.get("burst", False)
                card_cls = "ip-card burst" if burst else "ip-card"
                burst_tag = '<span class="burst-tag">BURST</span>' if burst else ""
                cards_html += f"""
                <div class="{card_cls}">
                    {burst_tag}
                    <div class="ip-addr">🚨 {ip_addr}</div>
                    <div class="ip-meta">
                        attacks={info.get('attack_count',0)} &nbsp;|&nbsp;
                        total={info.get('total_events',0)}
                    </div>
                </div>
                """
            st.markdown(
                f'<div style="max-height:380px;overflow-y:auto;">{cards_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-family:monospace;font-size:0.72rem;color:#1a4a6a;'
                'padding:1.5rem;text-align:center;">NO FLAGGED IPs</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── LOG INSPECTION PANEL ──────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header"><span class="icon">◈</span>'
        f'LOG INSPECTION  '
        f'<span style="font-size:0.6rem;color:#1a4a6a;letter-spacing:1px;">'
        f'[ SHOWING {min(log_limit, len(filtered_results))} / {len(filtered_results)} '
        f'{"(FILTERED)" if risk_filter != "ALL" else ""} ]'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    display_logs = filtered_results[:log_limit]

    if display_logs:
        rows_html = ""
        for r in display_logs:
            rl    = r["risk_level"]
            rc    = RISK_COLOR.get(rl, "#4a7a9b")
            icon  = RISK_ICON.get(rl, "")
            atk_badge = (
                f'<span style="color:#ff2244;font-size:0.6rem;">⚠ ATTACK</span>'
                if r["is_attack"] else
                '<span style="color:#1a4a6a;font-size:0.6rem;">NORMAL</span>'
            )
            log_text = str(r.get("log", ""))[:90]
            ml_s  = score_bar(r.get("ml_prob",    0), "#00aaff")
            rs_s  = score_bar(r.get("rule_score", 0), "#7755ff")
            hs_s  = score_bar(r.get("final_score",0), rc)
            rows_html += f"""
            <tr>
                <td style="color:#1a4a6a;font-size:0.65rem;">{r['index']+1}</td>
                <td>{risk_badge(rl)}</td>
                <td>{atk_badge}</td>
                <td class="log-text">{log_text}</td>
                <td style="color:#8aadcc;font-size:0.68rem;">{r.get('attack_type','—')}</td>
                <td>{ml_s}</td>
                <td>{rs_s}</td>
                <td>{hs_s}</td>
                <td style="color:#1a4a6a;font-size:0.65rem;">{r.get('ip') or '—'}</td>
            </tr>
            """
        st.markdown(
            f"""
            <div style="overflow-x:auto;max-height:480px;overflow-y:auto;">
            <table class="log-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>RISK</th>
                    <th>STATUS</th>
                    <th>LOG</th>
                    <th>TYPE</th>
                    <th>ML SCORE</th>
                    <th>RULE SCORE</th>
                    <th>HYBRID SCORE</th>
                    <th>IP</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-family:monospace;font-size:0.75rem;color:#1a4a6a;'
            'padding:2rem;text-align:center;">NO LOGS MATCH CURRENT FILTER</div>',
            unsafe_allow_html=True,
        )

    # ── FOOTER ────────────────────────────────────────────────────────────────
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.6rem;'
        'color:#0a2a40;text-align:center;letter-spacing:2px;padding:0.5rem 0;">'
        'SOC THREAT INTELLIGENCE DASHBOARD  ◈  HYBRID ML ENGINE  ◈  ALL RIGHTS RESERVED'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()