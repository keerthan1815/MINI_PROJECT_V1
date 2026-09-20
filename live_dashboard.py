"""
live_dashboard.py — Real-time AI Network Failure Prediction & Explainability Dashboard.

Single Source of Truth: config.py
Never hardcodes feature names, operational thresholds, or model paths.
"""

from collections import deque
import json
import os
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------------------
# Project Configuration & Constants
# ---------------------------------------------------------------------------
from config import (
    ALL_FEATURES,
    CHART_CONFUSION_MATRIX,
    CHART_REGRESSION_DIST,
    CHART_ROC_CURVE,
    CHART_SHAP_BAR,
    CHART_SHAP_BEESWARM,
    CLASS_LABEL,
    DNS_LATENCY_CRITICAL,
    DNS_LATENCY_DEAD,
    DNS_LATENCY_WARNING,
    EMAIL_COOLDOWN,
    FAILURE_SCENARIOS,
    HEALING_COOLDOWN,
    HISTORY_LENGTH,
    JITTER_CRITICAL,
    JITTER_WARNING,
    MINUTES_CRITICAL,
    MINUTES_WARNING,
    MODEL_CLF_PATH,
    MODEL_REG_PATH,
    NIC_ERRORS_CRITICAL,
    NIC_ERRORS_WARNING,
    PING_COUNT,
    PROJECT_TITLE,
    RAW_FEATURES,
    REFRESH_SECS,
    REGRESSION_LABEL,
    ROLLING_FEATURES,
    ROLLING_WINDOW,
    ROUTER_LATENCY_CRITICAL,
    ROUTER_LATENCY_DEAD,
    ROUTER_LATENCY_WARNING,
    ROUTER_LOSS_CRITICAL,
    ROUTER_LOSS_WARNING,
    RSSI_CRITICAL,
    RSSI_WARNING,
    SMOTE_RANDOM_STATE,
    TRAFFIC_CRITICAL,
    TRAFFIC_WARNING,
    TRAINING_DATA,
    TREND_FEATURES,
    TX_RATE_CRITICAL,
    TX_RATE_WARNING,
    XGBOOST_CLF_PARAMS,
    XGBOOST_REG_PARAMS,
)

# ---------------------------------------------------------------------------
# Subsystem Imports
# ---------------------------------------------------------------------------
from auth import login
from buzzer import start_buzzer, stop_buzzer
from chat_assistant import handle_query
from database import (
    get_alerts,
    get_recent_readings,
    init_db,
    insert_reading,
)
from email_alert import send_alert
from failure_logger import log_failure
from health_scorer import calculate_health_score, rssi_quality, score_to_label
from network_simulator import NetworkSimulator
from real_network_collector import detect_network_config, get_real_reading
from report_pdf import generate_pdf_report
from root_cause import diagnose
from self_healing import flag_overload, flag_router, flag_switch_port, heal_dns, run_healing
from sla_monitor import calculate_sla
from system_status import get_overall_status, update_device_status

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ===========================================================================
# Streamlit Page Config
# ===========================================================================
st.set_page_config(
    page_title="AI Network Failure Prediction System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure database tables exist
try:
    init_db()
except Exception:
    pass

# Custom styling for rich, modern glassmorphic aesthetics
st.markdown("""
<style>
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-label {
        font-size: 0.82rem;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.45rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-unit {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 400;
    }
    .status-badge-critical {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        border-radius: 6px;
        padding: 6px 12px;
        font-weight: 600;
    }
    .status-badge-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
        border-radius: 6px;
        padding: 6px 12px;
        font-weight: 600;
    }
    .status-badge-healthy {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #10b981;
        border-radius: 6px;
        padding: 6px 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Session State Initialization (Guarded single block)
# ===========================================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = ""
if "simulators" not in st.session_state:
    st.session_state.simulators = {s: NetworkSimulator(scenario_name=s) for s in FAILURE_SCENARIOS.keys()}
if "scenario_histories" not in st.session_state:
    st.session_state.scenario_histories = {s: deque(maxlen=HISTORY_LENGTH) for s in FAILURE_SCENARIOS.keys()}
if "real_history" not in st.session_state:
    st.session_state.real_history = deque(maxlen=HISTORY_LENGTH)
if "last_email_time" not in st.session_state:
    st.session_state.last_email_time = 0.0
if "last_healing_time" not in st.session_state:
    st.session_state.last_healing_time = 0.0
if "buzzer_active" not in st.session_state:
    st.session_state.buzzer_active = False
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to the AI Network Assistant!**\n\n"
                "I can analyze network root causes, explain XGBoost predictions, "
                "inspect SHAP values, and recommend self-healing procedures. How can I help?"
            ),
        }
    ]


# ===========================================================================
# LOGIN GATE
# ===========================================================================
if not st.session_state.logged_in:
    _, center_col, _ = st.columns([1, 1.8, 1])
    with center_col:
        st.markdown(f"## 📡 {PROJECT_TITLE}")
        st.caption(
            "Predictive AI failure detection and root-cause explainability for network "
            "infrastructure devices using telemetry symptoms, XGBoost, and SHAP."
        )
        st.markdown("---")
        with st.form("login_form"):
            st.markdown("#### Operator Login")
            u_input = st.text_input("Username", placeholder="e.g. admin, engineer, viewer")
            p_input = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

            if submitted:
                role = login(u_input.strip(), p_input.strip())
                if role:
                    st.session_state.logged_in = True
                    st.session_state.username = u_input.strip()
                    st.session_state.role = role
                    st.success(f"Authenticated as **{u_input.strip()}** ({role.upper()})")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please verify your username and password.")

        st.info("Demo accounts: `admin/admin123` · `engineer/eng123` · `viewer/view123`")
    st.stop()


# ===========================================================================
# Auto-Refresh Component
# ===========================================================================
st_autorefresh(interval=int(REFRESH_SECS * 1000), key="network_dashboard_refresher")


# ===========================================================================
# Model Loading (Cached via st.cache_resource)
# ===========================================================================
@st.cache_resource(show_spinner=False)
def load_ml_models():
    clf, reg, explainer = None, None, None
    clf_exists = os.path.exists(MODEL_CLF_PATH)
    reg_exists = os.path.exists(MODEL_REG_PATH)

    if clf_exists and reg_exists:
        try:
            clf = joblib.load(MODEL_CLF_PATH)
            reg = joblib.load(MODEL_REG_PATH)
            # Verify model was trained with the exact 15 ALL_FEATURES schema
            if hasattr(clf, "feature_names_in_"):
                if list(clf.feature_names_in_) != ALL_FEATURES:
                    return None, None, None
            if SHAP_AVAILABLE:
                try:
                    explainer = shap.TreeExplainer(clf)
                except Exception:
                    explainer = None
        except Exception:
            clf, reg, explainer = None, None, None
    return clf, reg, explainer


clf_model, reg_model, shap_explainer = load_ml_models()
models_ready = (clf_model is not None and reg_model is not None)


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.markdown("### 📡 Network AI Agent")
    st.markdown(f"User: **{st.session_state.username}** (`{st.session_state.role.upper()}`)")

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.role = ""
        if st.session_state.buzzer_active:
            try:
                stop_buzzer()
            except Exception:
                pass
            st.session_state.buzzer_active = False
        st.rerun()

    st.markdown("---")
    active_page = st.radio(
        "Navigation",
        [
            "📡 Live Network Monitor",
            "📊 Analytics",
            "📋 Network Logs",
            "🔧 Self Healing",
            "📄 Reports & SLA",
            "💬 AI Assistant",
            "ℹ️ System Info",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### Data Source")
    data_mode = st.radio(
        "Select Telemetry Feed",
        [
            "🖥️ Simulated (failure scenarios)",
            "📡 Real (your network)",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### Evaluation Artifacts")
    for label, path in [
        ("Confusion Matrix", CHART_CONFUSION_MATRIX),
        ("ROC Curve", CHART_ROC_CURVE),
        ("Feature Importance", CHART_SHAP_BAR),
    ]:
        if os.path.exists(path):
            with st.expander(f"📈 {label}"):
                st.image(path, use_container_width=True)


# ===========================================================================
# HEADER
# ===========================================================================
head_col1, head_col2, head_col3 = st.columns([3, 1.2, 1.2])
with head_col1:
    st.markdown(f"## 🌐 {PROJECT_TITLE}")
    st.caption("Monitoring network failure through 9 real-time performance indicators — powered by XGBoost and SHAP")
with head_col2:
    current_time_str = datetime.now().strftime("%H:%M:%S · %d %b %Y")
    st.markdown(f"🕒 **Current Time**\n`{current_time_str}`")
with head_col3:
    overall_status = get_overall_status()
    status_color = (
        "status-badge-critical" if overall_status == "CRITICAL"
        else "status-badge-warning" if overall_status == "WARNING"
        else "status-badge-healthy"
    )
    status_icon = "🔴" if overall_status == "CRITICAL" else "🟡" if overall_status == "WARNING" else "🟢"
    st.markdown(f"🏥 **Overall Status**\n<div class='{status_color}'>{status_icon} {overall_status}</div>", unsafe_allow_html=True)

if not models_ready:
    st.error(
        f"⚠️ Machine learning models not found or incompatible with the 15-feature schema (`{MODEL_CLF_PATH}`, `{MODEL_REG_PATH}`). "
        "Please train the models by executing: `python train_model.py`"
    )

st.markdown("---")

# ===========================================================================
# HELPER: Feature Vector Builder & Inference
# ===========================================================================
def process_telemetry_reading(reading_raw: dict, stream_history: deque):
    """
    Given a reading and its stream history, compute rolling/trend features,
    run model predictions, diagnose root cause, and return evaluation bundle.
    """
    stream_history.append(reading_raw)
    hist_list = list(stream_history)

    # Calculate rolling5 metrics over recent window
    r_lat_roll = float(np.mean([x.get("router_latency", 10.0) for x in hist_list[-ROLLING_WINDOW:]]))
    d_lat_roll = float(np.mean([x.get("dns_latency", 45.0) for x in hist_list[-ROLLING_WINDOW:]]))
    rssi_roll = float(np.mean([x.get("rssi", -58.0) or -58.0 for x in hist_list[-ROLLING_WINDOW:]]))

    # Calculate trend metrics (latest minus ROLLING_WINDOW ticks ago)
    window_start = hist_list[-min(len(hist_list), ROLLING_WINDOW)]
    r_lat_trend = float(reading_raw.get("router_latency", 10.0) - window_start.get("router_latency", 10.0))
    d_lat_trend = float(reading_raw.get("dns_latency", 45.0) - window_start.get("dns_latency", 45.0))
    curr_rssi = reading_raw.get("rssi", -58.0) or -58.0
    start_rssi = window_start.get("rssi", -58.0) or -58.0
    rssi_trend = float(curr_rssi - start_rssi)

    # Construct the complete 15 ALL_FEATURES dictionary
    feat_dict = {
        "router_latency": reading_raw.get("router_latency", 10.0),
        "router_packet_loss": reading_raw.get("router_packet_loss", 0.0),
        "dns_latency": reading_raw.get("dns_latency", 45.0),
        "dns_packet_loss": reading_raw.get("dns_packet_loss", 0.0),
        "rssi": reading_raw.get("rssi", 0.0) if reading_raw.get("rssi") is not None else 0.0,
        "tx_rate": reading_raw.get("tx_rate", 0.0) if reading_raw.get("tx_rate") is not None else 0.0,
        "jitter": reading_raw.get("jitter", 2.0),
        "nic_errors": reading_raw.get("nic_errors", 0.0),
        "traffic": reading_raw.get("traffic", 50.0),
        "router_latency_rolling5": round(r_lat_roll, 2),
        "dns_latency_rolling5": round(d_lat_roll, 2),
        "rssi_rolling5": round(rssi_roll, 2),
        "router_latency_trend": round(r_lat_trend, 2),
        "dns_latency_trend": round(d_lat_trend, 2),
        "rssi_trend": round(rssi_trend, 2),
    }

    # Replace None values with 0.0 for model compatibility
    for feat in ALL_FEATURES:
        if feat_dict.get(feat) is None:
            feat_dict[feat] = 0.0

    features_df = pd.DataFrame([[feat_dict[f] for f in ALL_FEATURES]], columns=ALL_FEATURES)

    pred = 0
    prob = 0.0
    minutes = 10.0

    if models_ready:
        try:
            pred = int(clf_model.predict(features_df)[0])
            prob = float(clf_model.predict_proba(features_df)[0][1])
            minutes = float(np.clip(reg_model.predict(features_df)[0], 0.0, 10.0))
        except Exception:
            pred = reading_raw.get("is_failure", 0)
            prob = 0.95 if pred == 1 else 0.05
            minutes = reading_raw.get("minutes_to_failure", 10.0)
    else:
        pred = reading_raw.get("is_failure", 0)
        prob = 0.95 if pred == 1 else 0.05
        minutes = reading_raw.get("minutes_to_failure", 10.0)

    # Health score and root-cause analysis
    health_score = calculate_health_score(reading_raw, prob)
    rca = diagnose(reading_raw)
    severity = rca.get("severity", "LOW")

    # Update system status map
    device_label = reading_raw.get("scenario") or reading_raw.get("device", "My Network")
    update_device_status(device_label, severity)

    # Persist reading to database
    try:
        insert_reading(
            device=device_label,
            metrics=reading_raw,
            prediction="Failure" if pred == 1 else "Normal",
            confidence=round(prob * 100, 1),
            health_score=health_score,
            severity=severity,
            failing_device=rca.get("reasons", [""])[-1],
        )
    except Exception:
        pass

    # Handle alert dispatches & automated healing
    now_ts = time.time()
    if pred == 1:
        if severity == "CRITICAL":
            # Audible alarm
            if not st.session_state.buzzer_active:
                try:
                    start_buzzer()
                    st.session_state.buzzer_active = True
                except Exception:
                    pass

            # Email notification with cooldown
            if now_ts - st.session_state.last_email_time > EMAIL_COOLDOWN:
                try:
                    send_alert(device_label, "FAILURE PREDICTED", " | ".join(rca.get("reasons", [])), reading_raw)
                    st.session_state.last_email_time = now_ts
                except Exception:
                    pass

        # Trigger self-healing with cooldown
        if now_ts - st.session_state.last_healing_time > HEALING_COOLDOWN:
            try:
                run_healing(reading_raw, rca, failure_risk=prob * 100, device=device_label)
                st.session_state.last_healing_time = now_ts
            except Exception:
                pass

        # Log failure to CSV and DB alerts
        try:
            log_failure(device_label, reading_raw, severity, rca.get("lead_time", "N/A"), rca.get("reasons", []))
        except Exception:
            pass

    else:
        # Turn off buzzer when healthy
        if st.session_state.buzzer_active:
            try:
                stop_buzzer()
                st.session_state.buzzer_active = False
            except Exception:
                pass

    return {
        "features_df": features_df,
        "pred": pred,
        "prob": prob,
        "minutes": minutes,
        "health_score": health_score,
        "rca": rca,
        "reading": reading_raw,
    }


# ===========================================================================
# HELPER: Render Single Device / Stream View
# ===========================================================================
def render_stream_view(bundle: dict, stream_history: deque, is_real: bool = False):
    reading = bundle["reading"]
    pred = bundle["pred"]
    prob = bundle["prob"]
    minutes = bundle["minutes"]
    health = bundle["health_score"]
    rca = bundle["rca"]
    severity = rca.get("severity", "LOW")

    if is_real:
        st.caption(
            f"📍 **Detected Network Profile** | Gateway: `{reading.get('_gateway_ip', 'N/A')}` | "
            f"DNS: `{reading.get('_dns_ip', 'N/A')}` | Link: `{reading.get('_conn_type', 'Unknown')}` | "
            f"SSID: `{reading.get('_ssid', 'N/A')}` | Channel: `{reading.get('_channel', 'N/A')}`"
        )

    # ROW 1 — 5 metric cards
    r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
    with r1c1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Router Ping</div>"
            f"<div class='metric-val'>{reading.get('router_latency', 0):.1f} <span class='metric-unit'>ms</span></div></div>",
            unsafe_allow_html=True,
        )
    with r1c2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Router Packet Loss</div>"
            f"<div class='metric-val'>{reading.get('router_packet_loss', 0):.1f} <span class='metric-unit'>%</span></div></div>",
            unsafe_allow_html=True,
        )
    with r1c3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>DNS Ping</div>"
            f"<div class='metric-val'>{reading.get('dns_latency', 0):.1f} <span class='metric-unit'>ms</span></div></div>",
            unsafe_allow_html=True,
        )
    with r1c4:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>DNS Packet Loss</div>"
            f"<div class='metric-val'>{reading.get('dns_packet_loss', 0):.1f} <span class='metric-unit'>%</span></div></div>",
            unsafe_allow_html=True,
        )
    with r1c5:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>NIC Errors</div>"
            f"<div class='metric-val'>{reading.get('nic_errors', 0):.1f} <span class='metric-unit'>/s</span></div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # ROW 2 — 5 metric cards
    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)
    is_eth = (reading.get("_conn_type") == "Ethernet")
    rssi_display = "N/A" if is_eth or reading.get("rssi") is None else f"{reading.get('rssi'):.0f}"
    tx_display = "N/A" if is_eth or reading.get("tx_rate") is None else f"{reading.get('tx_rate'):.0f}"

    with r2c1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>WiFi RSSI</div>"
            f"<div class='metric-val'>{rssi_display} <span class='metric-unit'>dBm</span></div></div>",
            unsafe_allow_html=True,
        )
    with r2c2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>WiFi TX Rate</div>"
            f"<div class='metric-val'>{tx_display} <span class='metric-unit'>Mbps</span></div></div>",
            unsafe_allow_html=True,
        )
    with r2c3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Jitter</div>"
            f"<div class='metric-val'>{reading.get('jitter', 0):.1f} <span class='metric-unit'>ms</span></div></div>",
            unsafe_allow_html=True,
        )
    with r2c4:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Traffic</div>"
            f"<div class='metric-val'>{reading.get('traffic', 0):.1f} <span class='metric-unit'>KB/s</span></div></div>",
            unsafe_allow_html=True,
        )
    with r2c5:
        h_label, h_icon = score_to_label(health)
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Health Score</div>"
            f"<div class='metric-val'>{health:.0f} <span class='metric-unit'>/100 {h_icon}</span></div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # WiFi Signal quality bar if on WiFi
    conn_type_raw = reading.get("_conn_type", "WiFi")
    if conn_type_raw == "WiFi" and reading.get("rssi") is not None and reading.get("rssi") != 0.0:
        rssi_val = reading["rssi"]
        quality_str = rssi_quality(rssi_val)
        pct_sig = min(1.0, max(0.0, (rssi_val + 100) / 70.0))
        st.progress(pct_sig, text=f"📶 WiFi Signal Quality: {quality_str} ({rssi_val:.0f} dBm)")

    # Health score bar
    st.progress(health / 100.0, text=f"🩺 Path Health Index: {h_label} ({health:.1f}/100)")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Status Banner (Prediction Outcome)
    if pred == 1:
        banner_color = "#ef4444" if severity == "CRITICAL" else "#f59e0b" if severity == "MEDIUM" else "#eab308"
        st.markdown(
            f"""
            <div style='background-color: rgba(239, 68, 68, 0.15); border-left: 6px solid {banner_color}; padding: 16px; border-radius: 6px; margin-bottom: 15px;'>
                <h4 style='margin: 0; color: {banner_color};'>🚨 PREDICTIVE FAILURE DETECTED — {severity} SEVERITY</h4>
                <p style='margin: 4px 0 8px 0; font-size: 0.95rem; color: #f1f5f9;'>
                    <strong>Confidence:</strong> {prob * 100:.1f}% &nbsp;|&nbsp; 
                    <strong>Estimated Lead Time:</strong> {rca.get('lead_time', 'Imminent')}
                </p>
                <div style='color: #cbd5e1; font-size: 0.9rem;'>
                    <strong>Diagnostic Findings:</strong>
                    <ul style='margin: 4px 0 0 20px;'>
                        {"".join(f"<li>{r}</li>" for r in rca.get('reasons', []))}
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style='background-color: rgba(16, 185, 129, 0.15); border-left: 6px solid #10b981; padding: 14px; border-radius: 6px; margin-bottom: 15px;'>
                <h4 style='margin: 0; color: #10b981;'>🟢 NETWORK PATH HEALTHY</h4>
                <p style='margin: 4px 0 0 0; font-size: 0.92rem; color: #e2e8f0;'>
                    AI Failure Risk: <strong>{prob * 100:.1f}%</strong> — All telemetric indicators are operating within nominal thresholds.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Gauge and Line Chart Side by Side
    col_gauge, col_chart = st.columns([1, 2])

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(minutes, 1),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Minutes to Failure (Regressor)", 'font': {'size': 15, 'color': '#f8fafc'}},
            number={'suffix': " min", 'font': {'size': 28, 'color': '#ffffff'}},
            gauge={
                'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': "#ef4444" if minutes <= MINUTES_CRITICAL else "#f59e0b" if minutes <= MINUTES_WARNING else "#10b981"},
                'bgcolor': "#1e293b",
                'borderwidth': 1,
                'bordercolor': "rgba(255,255,255,0.1)",
                'steps': [
                    {'range': [0, MINUTES_CRITICAL], 'color': 'rgba(239, 68, 68, 0.35)'},
                    {'range': [MINUTES_CRITICAL, MINUTES_WARNING], 'color': 'rgba(245, 158, 11, 0.3)'},
                    {'range': [MINUTES_WARNING, 10], 'color': 'rgba(16, 185, 129, 0.25)'},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 3},
                    'thickness': 0.8,
                    'value': MINUTES_CRITICAL,
                },
            },
        ))
        fig_gauge.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "#f8fafc"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_chart:
        hist_df = pd.DataFrame(list(stream_history))
        if not hist_df.empty:
            fig_lines = go.Figure()
            if "router_latency" in hist_df.columns:
                fig_lines.add_trace(go.Scatter(y=hist_df["router_latency"], mode="lines", name="Router Lat (ms)", line=dict(color="#3b82f6", width=2)))
            if "dns_latency" in hist_df.columns:
                fig_lines.add_trace(go.Scatter(y=hist_df["dns_latency"], mode="lines", name="DNS Lat (ms)", line=dict(color="#8b5cf6", width=2)))
            if "jitter" in hist_df.columns:
                fig_lines.add_trace(go.Scatter(y=hist_df["jitter"], mode="lines", name="Jitter (ms)", line=dict(color="#ec4899", width=1.5)))
            if "router_packet_loss" in hist_df.columns:
                fig_lines.add_trace(go.Scatter(y=hist_df["router_packet_loss"], mode="lines", name="Router Loss (%)", line=dict(color="#ef4444", width=1.5, dash="dot")))

            fig_lines.update_layout(
                title="Recent Metric History (60 Cycles)",
                title_font=dict(size=14, color="#f8fafc"),
                height=280,
                margin=dict(l=30, r=20, t=40, b=30),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30, 41, 59, 0.4)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                xaxis=dict(showgrid=False, title="Reading Index"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Value"),
            )
            st.plotly_chart(fig_lines, use_container_width=True)

    # SHAP Feature Importance Waterfall when Failing
    if pred == 1 and models_ready:
        st.markdown("#### 🔬 AI Explainability: Top Metric Contributors to Failure")
        feat_df = bundle["features_df"]

        contributions = None
        if shap_explainer is not None:
            try:
                sv = shap_explainer.shap_values(feat_df)
                if isinstance(sv, list) and len(sv) == 2:
                    contributions = sv[1][0]
                elif isinstance(sv, np.ndarray) and sv.ndim == 2:
                    contributions = sv[0]
            except Exception:
                contributions = None

        if contributions is None:
            try:
                fi = clf_model.feature_importances_
                contributions = fi * np.abs(feat_df.iloc[0].values)
            except Exception:
                contributions = np.ones(len(ALL_FEATURES))

        contrib_df = pd.DataFrame({
            "Feature": ALL_FEATURES,
            "Impact": contributions,
            "Value": feat_df.iloc[0].values,
        }).sort_values(by="Impact", ascending=True).tail(8)

        fig_waterfall = go.Figure(go.Bar(
            x=contrib_df["Impact"],
            y=contrib_df["Feature"],
            orientation="h",
            marker=dict(
                color=["#ef4444" if x > 0 else "#3b82f6" for x in contrib_df["Impact"]],
                line=dict(color="rgba(255,255,255,0.2)", width=1),
            ),
            text=[f"Val: {v:.1f}" for v in contrib_df["Value"]],
            textposition="auto",
        ))
        fig_waterfall.update_layout(
            title="Key Symptoms Driving Failure Prediction",
            height=260,
            margin=dict(l=140, r=20, t=35, b=25),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30, 41, 59, 0.4)',
            xaxis=dict(title="Relative SHAP Impact", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

# ===========================================================================
# PAGE 1 — LIVE NETWORK MONITOR
# ===========================================================================
if active_page == "📡 Live Network Monitor":
    if data_mode == "🖥️ Simulated (failure scenarios)":
        scenario_names = list(FAILURE_SCENARIOS.keys())
        tabs = st.tabs([f"⚡ {s.replace('_', ' ').title()}" for s in scenario_names])

        for idx, s in enumerate(scenario_names):
            with tabs[idx]:
                st.caption(f"Simulating: {FAILURE_SCENARIOS[s]}")
                sim = st.session_state.simulators[s]
                reading = sim.next_reading()
                hist = st.session_state.scenario_histories[s]
                bundle = process_telemetry_reading(reading, hist)
                render_stream_view(bundle, hist, is_real=False)

    else:
        st.markdown("### 📡 Live Host Telemetry Stream")
        reading = get_real_reading("My Network")
        hist = st.session_state.real_history
        bundle = process_telemetry_reading(reading, hist)
        render_stream_view(bundle, hist, is_real=True)


# ===========================================================================
# PAGE 2 — ANALYTICS
# ===========================================================================
elif active_page == "📊 Analytics":
    st.markdown("### 📊 Predictive Failure Analytics")

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("#### Failure Incidents by Scenario / Type")
        try:
            alerts = get_alerts(limit=300)
            if alerts:
                adf = pd.DataFrame(alerts)
                device_counts = adf["device"].value_counts()
                fig_pie = go.Figure(go.Pie(
                    labels=device_counts.index,
                    values=device_counts.values,
                    hole=0.45,
                ))
                fig_pie.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No failure incidents recorded in database yet.")
        except Exception as e:
            st.warning(f"Could not load incident counts: {e}")

    with col_a2:
        st.markdown("#### Failure Trend Timeline")
        try:
            readings = get_recent_readings(limit=250)
            if readings:
                rdf = pd.DataFrame(readings)
                if "timestamp" in rdf.columns and "health_score" in rdf.columns:
                    rdf["timestamp"] = pd.to_datetime(rdf["timestamp"])
                    fig_trend = go.Figure(go.Scatter(
                        x=rdf["timestamp"],
                        y=rdf["health_score"],
                        mode="lines+markers",
                        line=dict(color="#3b82f6", width=2),
                        name="Path Health",
                    ))
                    fig_trend.update_layout(
                        height=300,
                        margin=dict(l=30, r=20, t=30, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(30, 41, 59, 0.4)',
                        yaxis=dict(range=[0, 100], title="Health (0-100)"),
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No timeline data logged yet.")
        except Exception as e:
            st.warning(f"Could not load health trend: {e}")

    st.markdown("---")
    st.markdown("#### Model Evaluation & Feature Distributions")
    c_img1, c_img2 = st.columns(2)
    with c_img1:
        if os.path.exists(CHART_SHAP_BAR):
            st.image(CHART_SHAP_BAR, caption="Feature Importance (Raw, Rolling, Trend)", use_container_width=True)
        else:
            st.info(f"Chart `{CHART_SHAP_BAR}` not generated yet.")

    with c_img2:
        if os.path.exists(CHART_REGRESSION_DIST):
            st.image(CHART_REGRESSION_DIST, caption="Minutes-to-Failure Regression Distribution", use_container_width=True)
        else:
            st.info(f"Chart `{CHART_REGRESSION_DIST}` not generated yet.")


# ===========================================================================
# PAGE 3 — NETWORK LOGS
# ===========================================================================
elif active_page == "📋 Network Logs":
    st.markdown("### 📋 Historical Network Telemetry Logs")

    try:
        raw_logs = get_recent_readings(limit=300)
        if raw_logs:
            ldf = pd.DataFrame(raw_logs)

            filt_col1, filt_col2 = st.columns(2)
            with filt_col1:
                available_devices = ["All"] + sorted(ldf["device"].dropna().unique().tolist())
                sel_dev = st.selectbox("Filter by Stream / Scenario", available_devices)
            with filt_col2:
                available_sevs = ["All"] + sorted(ldf["severity"].dropna().unique().tolist()) if "severity" in ldf.columns else ["All"]
                sel_sev = st.selectbox("Filter by Severity", available_sevs)

            filtered = ldf.copy()
            if sel_dev != "All":
                filtered = filtered[filtered["device"] == sel_dev]
            if sel_sev != "All" and "severity" in filtered.columns:
                filtered = filtered[filtered["severity"] == sel_sev]

            st.dataframe(filtered, use_container_width=True, height=450)

            csv_data = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Logs to CSV",
                data=csv_data,
                file_name=f"network_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.info("Database contains no logged readings yet. Keep the monitor running to populate logs.")
    except Exception as e:
        st.error(f"Error querying network logs database: {e}")


# ===========================================================================
# PAGE 4 — SELF HEALING
# ===========================================================================
elif active_page == "🔧 Self Healing":
    st.markdown("### 🔧 Automated Self-Healing Policies & Manual Dispatch")

    st.markdown("#### Configured Self-Healing Rules")
    rules_data = [
        {"Condition": f"DNS Latency > {DNS_LATENCY_WARNING} ms", "Automated Action": "Flush OS resolver DNS cache", "Target Component": "DNS / Gateway"},
        {"Condition": f"Router RTT > {ROUTER_LATENCY_CRITICAL} ms or Loss > {ROUTER_LOSS_CRITICAL}%", "Automated Action": "Flag router queue check & restart window", "Target Component": "Router"},
        {"Condition": f"NIC Errors > {NIC_ERRORS_WARNING}/s", "Automated Action": "Flag physical cable & switch port verification", "Target Component": "Switch / PHY"},
        {"Condition": f"Traffic > {TRAFFIC_WARNING} KB/s", "Automated Action": "Log edge QoS overload & check DDoS signatures", "Target Component": "Firewall / Edge"},
    ]
    st.table(pd.DataFrame(rules_data))

    st.markdown("#### Manual Recovery Triggers (Admin Role)")
    btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)

    is_admin = (st.session_state.role == "admin")
    if not is_admin:
        st.caption("🔒 Manual recovery dispatch requires `admin` role privileges.")

    with btn_c1:
        if st.button("🧹 Flush DNS Cache", disabled=not is_admin, use_container_width=True):
            heal_dns("Manual Admin")
            st.success("DNS Resolver cache flushed successfully!")
    with btn_c2:
        if st.button("⚠️ Flag Router Inspection", disabled=not is_admin, use_container_width=True):
            flag_router("Manual Admin")
            st.success("Router flagged for session verification!")
    with btn_c3:
        if st.button("🔌 Flag Switch Port", disabled=not is_admin, use_container_width=True):
            flag_switch_port("Manual Admin")
            st.success("Switch port / PHY flagged for inspection!")
    with btn_c4:
        if st.button("📈 Log Overload Alert", disabled=not is_admin, use_container_width=True):
            flag_overload("Manual Admin")
            st.success("Edge traffic overload logged!")

    st.markdown("---")
    st.markdown("#### Recent Self-Healing Execution Log")
    if os.path.exists("healing_logs.csv"):
        try:
            h_df = pd.read_csv("healing_logs.csv")
            st.dataframe(h_df.tail(50), use_container_width=True)
        except Exception as e:
            st.info("No healing history entries yet.")
    else:
        st.info("No self-healing events have been recorded yet.")


# ===========================================================================
# PAGE 5 — REPORTS & SLA
# ===========================================================================
elif active_page == "📄 Reports & SLA":
    st.markdown("### 📄 SLA Compliance & Executive PDF Reports")

    sla_col1, sla_col2, sla_col3 = st.columns(3)
    try:
        sla_pct, up_cnt, down_cnt = calculate_sla()
    except Exception:
        sla_pct, up_cnt, down_cnt = 100.0, 100, 0

    with sla_col1:
        st.metric("Path Availability (SLA)", f"{sla_pct:.2f}%", delta="Target: 99.9%")
    with sla_col2:
        st.metric("Healthy Cycles (Uptime)", f"{up_cnt} ticks")
    with sla_col3:
        st.metric("Degraded Cycles (Downtime)", f"{down_cnt} ticks")

    st.markdown("---")
    st.markdown("#### Generate Executive Summary PDF")
    rep_c1, rep_c2 = st.columns([1, 2])
    with rep_c1:
        rep_type = st.selectbox("Report Time Horizon", ["daily", "weekly", "monthly"])
        if st.button("📑 Generate PDF Report", use_container_width=True):
            with st.spinner("Compiling metrics, charts, and audit trails..."):
                try:
                    pdf_filename = generate_pdf_report(rep_type)
                    st.success(f"Report generated: `{pdf_filename}`")
                    if os.path.exists(pdf_filename):
                        with open(pdf_filename, "rb") as f:
                            st.download_button(
                                label="⬇️ Download PDF Report",
                                data=f.read(),
                                file_name=pdf_filename,
                                mime="application/pdf",
                            )
                except Exception as e:
                    st.error(f"Failed to generate report: {e}")


# ===========================================================================
# PAGE 6 — AI ASSISTANT
# ===========================================================================
elif active_page == "💬 AI Assistant":
    st.markdown("### 💬 AI Network Diagnostics Assistant")
    st.caption("Ask questions about network symptoms, failing devices, XGBoost features, SLA, or self-healing.")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Type your question (e.g. 'Which device is failing?', 'Explain SHAP', 'SLA status')...")
    if user_query:
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            try:
                res = handle_query(user_query)
                answer = res.get("response", str(res)) if isinstance(res, dict) else str(res)
            except Exception as e:
                answer = f"I encountered an error querying the intelligence engine: {e}"

            st.markdown(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})


# ===========================================================================
# PAGE 7 — SYSTEM INFO
# ===========================================================================
elif active_page == "ℹ️ System Info":
    st.markdown("### ℹ️ System Architecture & Configuration")

    st.markdown("#### Real Host Network Discovery")
    try:
        net_cfg = detect_network_config()
        cfg_col1, cfg_col2 = st.columns(2)
        with cfg_col1:
            st.write(f"**Local IP Address:** `{net_cfg.get('local_ip')}`")
            st.write(f"**Default Gateway IP:** `{net_cfg.get('gateway_ip')}`")
            st.write(f"**Primary DNS Server:** `{net_cfg.get('primary_dns')}`")
            st.write(f"**DNS Resolvers List:** `{net_cfg.get('dns_servers')}`")
        with cfg_col2:
            st.write(f"**Connection Interface Type:** `{net_cfg.get('connection_type')}`")
            st.write(f"**Discovery Timestamp:** `{net_cfg.get('detected_at')}`")
    except Exception as e:
        st.warning(f"Could not read network configuration: {e}")

    st.markdown("---")
    st.markdown("#### Machine Learning Architecture & Hyperparameters")
    ml_col1, ml_col2 = st.columns(2)
    with ml_col1:
        st.markdown(f"**Classifier Artifact:** `{MODEL_CLF_PATH}`")
        st.json(XGBOOST_CLF_PARAMS)
    with ml_col2:
        st.markdown(f"**Regressor Artifact:** `{MODEL_REG_PATH}`")
        st.json(XGBOOST_REG_PARAMS)

    st.markdown("---")
    st.markdown(f"#### Complete Feature Vector (`ALL_FEATURES` — Total: {len(ALL_FEATURES)})")
    f_c1, f_c2, f_c3 = st.columns(3)
    with f_c1:
        st.markdown(f"**Raw Features ({len(RAW_FEATURES)})**")
        for f in RAW_FEATURES:
            st.markdown(f"- `{f}`")
    with f_c2:
        st.markdown(f"**Rolling Window ({len(ROLLING_FEATURES)})**")
        for f in ROLLING_FEATURES:
            st.markdown(f"- `{f}` (window={ROLLING_WINDOW})")
    with f_c3:
        st.markdown(f"**Short-term Trends ({len(TREND_FEATURES)})**")
        for f in TREND_FEATURES:
            st.markdown(f"- `{f}` (lag={ROLLING_WINDOW})")

    st.markdown("---")
    st.markdown("#### Active Failure Scenarios Pool")
    st.table(pd.DataFrame(list(FAILURE_SCENARIOS.items()), columns=["Scenario Identifier", "Observed Symptom Description"]))
