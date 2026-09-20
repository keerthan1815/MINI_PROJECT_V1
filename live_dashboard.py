"""
Live dashboard: predict NETWORK DEVICE failure from network symptoms.

Simulated: Router / Switch / Firewall each emit that device's failure signs.
Real: one "My Network" view of the actual gateway, DNS, WiFi, and NIC.
"""

import os
import time
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from auth import login
from buzzer import start_buzzer, stop_buzzer
from chat_assistant import handle_query
from database import (
    get_alerts, get_failure_count_by_device, get_healing_logs,
    get_recent_readings, insert_reading,
)
from email_alert import send_alert
from failure_logger import log_failure
from features import (
    ALL_FEATURES, FEATURE_HELP, METRIC_TO_DEVICE, RAW_FEATURES,
    REAL_DEVICE, SIM_DEVICES,
)
from health_scorer import calculate_health_score, rssi_quality, score_to_label
from logger import log_data
from network_simulator import NetworkSimulator
from real_network_collector import detect_all, get_real_reading, _measure_wifi
from report_pdf import generate_pdf_report
from root_cause import diagnose
from self_healing import run_healing
from sla_monitor import calculate_sla
from system_status import get_overall_status, status, update_device_status

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

CLF_FILE = "failure_model.pkl"
REG_FILE = "regression_model.pkl"
HISTORY_LEN = 60
EMAIL_COOL = 120
HEAL_COOL = 90
SEV_COLOR = {"LOW": "🟡", "MEDIUM": "🟠", "CRITICAL": "🔴"}

st.set_page_config(
    page_title="Network Device Failure — XGBoost + SHAP",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    st.markdown("# Network Device Failure Prediction")
    st.caption("XGBoost + SHAP · Router / Switch / Firewall symptoms — not PC health")
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        with st.form("lf"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            s = st.form_submit_button("Login", use_container_width=True)
        if s:
            role = login(u, p)
            if role:
                st.session_state.update({"logged_in": True, "username": u, "role": role})
                st.rerun()
            else:
                st.error("Invalid credentials")
        st.info("admin/admin123 · engineer/eng123 · viewer/view123")
    st.stop()

st_autorefresh(interval=2000, key="noc")


@st.cache_resource
def load_clf():
    return joblib.load(CLF_FILE) if os.path.exists(CLF_FILE) else None


@st.cache_resource
def load_reg():
    return joblib.load(REG_FILE) if os.path.exists(REG_FILE) else None


@st.cache_resource
def load_explainer(_clf):
    if _clf is None or not SHAP_AVAILABLE:
        return None
    try:
        return shap.TreeExplainer(_clf)
    except Exception:
        return None


clf = load_clf()
reg = load_reg()
explainer = load_explainer(clf)


def _empty_df():
    df = pd.DataFrame(columns=["timestamp"] + RAW_FEATURES + ["status"])
    return df


if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.simulators = {d: NetworkSimulator(d) for d in SIM_DEVICES}
    st.session_state.histories = {d: _empty_df() for d in SIM_DEVICES + [REAL_DEVICE]}
    st.session_state.last_email = {}
    st.session_state.last_heal = {}
    st.session_state.chat_history = []


def _rolling_features(device):
    hist = st.session_state.histories.get(device, _empty_df())
    w = 5

    def _roll(col):
        if col not in hist.columns:
            return 0.0
        vals = pd.to_numeric(hist[col], errors="coerce").dropna().tail(w)
        return round(float(vals.mean()), 2) if len(vals) else 0.0

    def _trend(col):
        if col not in hist.columns:
            return 0.0
        vals = pd.to_numeric(hist[col], errors="coerce").dropna().tail(w).tolist()
        if len(vals) >= 2:
            return round(float(vals[-1] - vals[0]), 2)
        return 0.0

    return {
        "router_latency_rolling5": _roll("router_latency_ms"),
        "dns_latency_rolling5": _roll("dns_latency_ms"),
        "rssi_rolling5": _roll("rssi_dbm"),
        "router_trend": _trend("router_latency_ms"),
        "dns_trend": _trend("dns_latency_ms"),
        "rssi_trend": _trend("rssi_dbm"),
    }


def _fmt_ms(v):
    if v is None:
        return "N/A"
    if v >= 9000:
        return "Timeout"
    return f"{v:.0f} ms"


def _explain_chart(feat_15):
    if SHAP_AVAILABLE and explainer is not None:
        try:
            X1 = pd.DataFrame([feat_15])[ALL_FEATURES]
            sv = explainer.shap_values(X1)[0]
            idx = np.argsort(np.abs(sv))[::-1]
            fig, ax = plt.subplots(figsize=(7, 3.6))
            ax.barh([ALL_FEATURES[i] for i in idx],
                    [sv[i] for i in idx],
                    color=["#ff4444" if v > 0 else "#4488ff" for v in sv[idx]])
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("SHAP (red → device failure)")
            ax.set_title("Why this looks like a failing network device", fontweight="bold")
            plt.tight_layout()
            return fig
        except Exception:
            pass
    if clf is not None:
        try:
            fi = clf.feature_importances_
            X1 = pd.DataFrame([feat_15])[ALL_FEATURES]
            scaled = [float(X1.iloc[0][f]) * fi[i] for i, f in enumerate(ALL_FEATURES)]
            idx = np.argsort(np.abs(scaled))[::-1]
            fig, ax = plt.subplots(figsize=(7, 3.6))
            ax.barh([ALL_FEATURES[i] for i in idx], [scaled[i] for i in idx],
                    color=["#ff4444" if scaled[i] > 0 else "#4488ff" for i in idx])
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("Contribution (red → failure)")
            ax.set_title("Feature contribution to device-failure score", fontweight="bold")
            plt.tight_layout()
            return fig
        except Exception:
            return None
    return None


def _minutes_gauge(minutes):
    capped = min(max(float(minutes), 0), 10)
    color = "#ff0000" if capped < 2 else "#ff8800" if capped < 5 else "#00cc44"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(capped, 1),
        title={"text": "Minutes to device failure", "font": {"size": 13}},
        number={"suffix": " min", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 10]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 2], "color": "#ffcccc"},
                {"range": [2, 5], "color": "#fff0cc"},
                {"range": [5, 10], "color": "#ccffcc"},
            ],
        },
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10),
                      paper_bgcolor="#0e1117", font=dict(color="white"))
    return fig


with st.sidebar:
    st.markdown("### Control")
    st.markdown(f"**{st.session_state['username']}** · `{st.session_state['role']}`")
    if st.button("Logout", use_container_width=True):
        st.session_state["logged_in"] = False
        st.rerun()
    st.divider()
    page = st.radio("Pages", [
        "Live Monitor", "Topology", "Analytics",
        "Logs", "Healing", "Reports", "Assistant", "How it works",
    ])
    st.divider()
    mode = st.radio("Data source", [
        "Simulated (Router / Switch / Firewall)",
        "Real (this host observing the LAN)",
    ])
    st.caption("Real mode pings your gateway and DNS. It does not score PC CPU/RAM.")
    st.divider()
    for img, cap in [
        ("confusion_matrix.png", "Confusion matrix"),
        ("roc_curve.png", "ROC"),
        ("shap_summary_bar.png", "Feature importance"),
    ]:
        if os.path.exists(img):
            st.image(img, caption=cap, use_container_width=True)

st.markdown("# Explainable AI prediction of network **device** failure")
st.caption("XGBoost + SHAP · 15 network symptoms · Router / Switch / WiFi AP / Firewall-ISP")

overall = get_overall_status()
st.info(
    f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}  ·  "
    f"Path: **{overall}**  ·  "
    f"Classifier {'ready' if clf else 'missing'}  ·  "
    f"Regressor {'ready' if reg else 'missing'}  ·  "
    f"SHAP {'on' if SHAP_AVAILABLE else 'importance fallback'}"
)
if overall == "CRITICAL":
    st.error("Critical network-device failure predicted")
    start_buzzer()
elif overall == "WARNING":
    st.warning("Warning signs on a network device")
else:
    stop_buzzer()

use_real = mode.startswith("Real")
devices = [REAL_DEVICE] if use_real else SIM_DEVICES


def page_live_monitor():
    if clf is None:
        st.error("Train the model first: `python generate_training_data.py` then `python train_model.py`")
        return

    tabs = st.tabs([f"{d}" for d in devices])
    for i, device in enumerate(devices):
        with tabs[i]:
            reading = get_real_reading(device_name=device) if use_real \
                else st.session_state.simulators[device].next_reading()

            roll = _rolling_features(device)
            feat_15 = {**{k: reading.get(k, 0) for k in RAW_FEATURES}, **roll}
            if use_real:
                for k in ["router_latency_rolling5", "dns_latency_rolling5",
                          "rssi_rolling5", "router_trend", "dns_trend", "rssi_trend"]:
                    if k in reading:
                        feat_15[k] = reading[k]

            X15 = pd.DataFrame([feat_15])[ALL_FEATURES]
            prob = float(clf.predict_proba(X15)[0][1])
            pred = int(clf.predict(X15)[0])
            status_str = "FAILURE PREDICTED" if pred else "Normal"
            minutes_to_fail = 10.0
            if reg is not None:
                try:
                    minutes_to_fail = float(np.clip(reg.predict(X15)[0], 0, 10))
                except Exception:
                    pass

            health = calculate_health_score(reading, prob)
            h_label, h_icon = score_to_label(health)
            rca = diagnose(reading)
            failing = rca["failing_device"] if pred else device
            update_device_status(device, rca["severity"] if pred else "Normal")
            if pred and failing in status:
                update_device_status(failing, rca["severity"])

            new_row = {
                "timestamp": reading["timestamp"],
                **{k: reading.get(k, 0) for k in RAW_FEATURES},
                "status": status_str,
            }
            hist = st.session_state.histories[device]
            st.session_state.histories[device] = pd.concat(
                [hist, pd.DataFrame([new_row])], ignore_index=True
            ).tail(HISTORY_LEN)

            try:
                insert_reading(
                    device, reading, status_str, round(prob * 100, 2),
                    health, rca["severity"] if pred else "Normal", failing)
            except Exception:
                pass
            try:
                log_data(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "DOWN" if pred and rca["severity"] == "CRITICAL" else "UP",
                    status_str, round(prob * 100, 2), failing)
            except Exception:
                pass

            wifi = reading.get("_wifi_available", reading.get("_conn_type") != "Ethernet")
            r1 = st.columns(5)
            r1[0].metric("Router RTT", _fmt_ms(reading["router_latency_ms"]),
                         help=FEATURE_HELP["router_latency_ms"])
            r1[1].metric("Router loss", f"{reading['router_packet_loss']:.0f}%")
            r1[2].metric("DNS / WAN RTT", _fmt_ms(reading["dns_latency_ms"]))
            r1[3].metric("DNS loss", f"{reading['dns_packet_loss']:.0f}%")
            r1[4].metric("Jitter", f"{reading['jitter_ms']:.1f} ms")

            r2 = st.columns(5)
            r2[0].metric("NIC errors/s", f"{reading['nic_errors_per_sec']:.1f}")
            r2[1].metric("Traffic", f"{reading['traffic_kbps']:.0f} KB/s")
            if wifi:
                r2[2].metric("RSSI", f"{reading['rssi_dbm']:.0f} dBm",
                             help=rssi_quality(reading["rssi_dbm"]))
                r2[3].metric("TX rate", f"{reading['tx_rate_mbps']:.0f} Mbps")
            else:
                r2[2].metric("RSSI", "N/A (Ethernet)")
                r2[3].metric("TX rate", "N/A (Ethernet)")
            r2[4].metric("Device health", f"{health}/100")

            if use_real:
                st.caption(
                    f"Gateway `{reading.get('_router_ip')}`  ·  "
                    f"DNS `{reading.get('_dns_ip')}`  ·  "
                    f"{reading.get('_conn_type')}"
                    + (f"  ·  SSID `{reading.get('_ssid')}`" if reading.get("_ssid") else "")
                )

            if wifi and reading["rssi_dbm"] < 0:
                st.progress(max(0.0, min((reading["rssi_dbm"] + 100) / 70, 1.0)))

            col_h, col_g = st.columns(2)
            with col_h:
                st.markdown(f"**{h_icon} {h_label}** — network device health {health}/100")
                st.progress(health / 100)
            with col_g:
                if reg is not None:
                    st.plotly_chart(_minutes_gauge(minutes_to_fail),
                                    use_container_width=True)

            chart_cols = [c for c in [
                "router_latency_ms", "dns_latency_ms", "jitter_ms",
                "router_packet_loss", "nic_errors_per_sec", "rssi_dbm",
            ] if c in st.session_state.histories[device].columns]
            cdf = st.session_state.histories[device].set_index("timestamp")[chart_cols]
            st.line_chart(cdf.astype(float), height=220)

            if pred:
                sev = rca["severity"]
                st.error(
                    f"{SEV_COLOR.get(sev, '')} **{sev}** — suspected **{failing}** "
                    f"({prob*100:.1f}% confidence, ~{minutes_to_fail:.1f} min)"
                )
                ca, cb = st.columns(2)
                with ca:
                    st.write(f"**Lead:** {rca['lead_time']}")
                    st.write("**Why this device (not your PC):**")
                    for r in rca["reasons"]:
                        st.write(f"- {r}")
                    if explainer is not None:
                        try:
                            sv = explainer.shap_values(
                                pd.DataFrame([feat_15])[ALL_FEATURES])[0]
                            top_i = int(np.argmax(np.abs(sv)))
                            fname = ALL_FEATURES[top_i]
                            st.info(
                                f"SHAP top driver: `{fname}` → "
                                f"{METRIC_TO_DEVICE.get(fname, 'network path')}"
                            )
                        except Exception:
                            pass
                with cb:
                    fig = _explain_chart(feat_15)
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)

                now_ts = time.time()
                if sev == "CRITICAL" and (now_ts - st.session_state.last_email.get(device, 0)) > EMAIL_COOL:
                    try:
                        send_alert(failing, status_str, " | ".join(rca["reasons"]), reading)
                        st.session_state.last_email[device] = now_ts
                    except Exception:
                        pass
                if (now_ts - st.session_state.last_heal.get(device, 0)) > HEAL_COOL:
                    try:
                        run_healing(reading, rca, failure_risk=prob * 100, device=failing)
                        st.session_state.last_heal[device] = now_ts
                    except Exception:
                        pass
                try:
                    log_failure(failing, reading, sev, rca["lead_time"], rca["reasons"])
                except Exception:
                    pass
            else:
                st.success(
                    f"**{device}** path looks normal "
                    f"(failure probability {prob*100:.1f}%)"
                )


def page_topology():
    st.markdown("## Observed network path")
    pos = {"Internet": (0, 2), "Firewall": (2, 2), "Router": (4, 2),
           "Switch": (6, 2), "WiFi AP": (6, 0.7)}
    edges = [("Internet", "Firewall"), ("Firewall", "Router"),
             ("Router", "Switch"), ("Router", "WiFi AP")]
    cmap = {"healthy": "#00cc44", "warning": "#ffaa00", "failure": "#ff3333"}
    ex, ey = [], []
    for s, d in edges:
        x0, y0 = pos[s]
        x1, y1 = pos[d]
        ex += [x0, x1, None]
        ey += [y0, y1, None]
    nx_, ny_, nc_, nl_ = [], [], [], []
    alias = {"WiFi AP": "Router"}
    for node, (x, y) in pos.items():
        nx_.append(x)
        ny_.append(y)
        key = alias.get(node, node)
        s = status.get(key, "healthy")
        nc_.append(cmap.get(s, "#aaa"))
        nl_.append(f"{node}\n{s.upper()}")
    fig = go.Figure(
        data=[
            go.Scatter(x=ex, y=ey, mode="lines",
                       line=dict(width=2, color="#555"), hoverinfo="none"),
            go.Scatter(x=nx_, y=ny_, mode="markers+text", text=nl_,
                       textposition="top center",
                       marker=dict(size=42, color=nc_,
                                   line=dict(width=2, color="#222"))),
        ],
        layout=go.Layout(
            showlegend=False, height=360,
            margin=dict(b=20, l=5, r=5, t=30),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="white"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Colours come from predicted device class, not from PC health.")


def page_analytics():
    st.markdown("## Model and alert analytics")
    counts = get_failure_count_by_device()
    if counts:
        cols = st.columns(len(counts))
        for i, (d, c) in enumerate(counts.items()):
            cols[i].metric(d, c)
    if os.path.exists("failure_history.csv"):
        try:
            df = pd.read_csv("failure_history.csv")
            df["Time"] = pd.to_datetime(df["Time"])
            st.line_chart(df.groupby(df["Time"].dt.date).size().rename("Failures"))
        except Exception:
            pass
    recs = get_recent_readings(limit=150)
    if recs:
        df2 = pd.DataFrame(recs)
        df2["timestamp"] = pd.to_datetime(df2["timestamp"])
        if "confidence" in df2.columns:
            st.line_chart(df2.set_index("timestamp")["confidence"])
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("shap_summary_bar.png"):
            st.image("shap_summary_bar.png", use_container_width=True)
    with c2:
        if os.path.exists("regression_distribution.png"):
            st.image("regression_distribution.png", use_container_width=True)


def page_logs():
    st.markdown("## Readings and alerts")
    t1, t2 = st.tabs(["Readings", "Alerts"])
    with t1:
        recs = get_recent_readings(limit=200)
        if recs:
            df = pd.DataFrame(recs)
            show = [c for c in [
                "timestamp", "device", "failing_device",
                "router_latency_ms", "router_packet_loss",
                "dns_latency_ms", "dns_packet_loss", "jitter_ms",
                "nic_errors_per_sec", "rssi_dbm", "traffic_kbps",
                "prediction", "health_score", "severity",
            ] if c in df.columns]
            st.dataframe(df[show], use_container_width=True)
        else:
            st.info("No readings yet.")
    with t2:
        alerts = get_alerts(limit=100)
        if alerts:
            st.dataframe(pd.DataFrame(alerts)[
                ["timestamp", "device", "severity", "lead_time", "reasons"]],
                use_container_width=True)
        else:
            st.info("No alerts yet.")


def page_healing():
    st.markdown("## Network-side response")
    st.write(
        "Actions target the **path**, not the PC: DNS cache flush, "
        "switch-port flag, router-check recommendation, overload log."
    )
    logs = get_healing_logs(limit=50)
    if logs:
        st.dataframe(pd.DataFrame(logs)[
            ["timestamp", "device", "issue", "action", "result"]],
            use_container_width=True)
    else:
        st.info("No healing actions yet.")


def page_reports():
    sla, up, dn = calculate_sla()
    c1, c2, c3 = st.columns(3)
    c1.metric("SLA", f"{sla}%")
    c2.metric("Up samples", up)
    c3.metric("Down samples", dn)
    st.progress(sla / 100)
    rtype = st.selectbox("Report", ["daily", "weekly", "monthly"])
    if st.button("Generate PDF", type="primary"):
        fname = generate_pdf_report(rtype)
        with open(fname, "rb") as f:
            st.download_button("Download", f, file_name=fname, mime="application/pdf")


def page_chat():
    st.markdown("## Assistant")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    user_input = st.chat_input("Ask which device is failing, or why...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        result = handle_query(user_input, context={"network_status": "UP"})
        st.session_state.chat_history.append(
            {"role": "assistant", "content": result.get("response", "")})
        st.rerun()


def page_how():
    st.markdown("## What is being predicted?")
    st.markdown("""
**Network device failure**, not PC failure.

A **device** here is a router, switch, WiFi AP, or firewall — hardware that
forwards traffic. This laptop is only the **observer**.

We cannot read a typical home router's CPU without SNMP. We *can* see the
job fail:

| Device | Symptom | Measurement |
|---|---|---|
| Router | Gateway RTT / loss / jitter | ping default gateway |
| WiFi AP | RSSI drop, TX-rate collapse | `netsh wlan` |
| Switch | PHY/NIC errors | `psutil` NIC counters (delta/s) |
| Firewall / ISP | DNS/WAN RTT and loss | ping configured DNS |

XGBoost classifies **failure vs normal**. A second model estimates
**minutes to failure**. SHAP names the symptom, which maps to the device.
""")
    cfg = detect_all()
    st.markdown("### This host's observed path")
    c1, c2, c3 = st.columns(3)
    c1.metric("Observer IP", cfg.get("local_ip") or "—")
    c2.metric("Router (gateway)", cfg.get("gateway_ip") or "—")
    c3.metric("DNS", cfg.get("primary_dns") or "—")
    wifi = _measure_wifi()
    if wifi:
        st.caption(f"WiFi SSID `{wifi.get('ssid')}`  RSSI {wifi.get('rssi')} dBm  "
                   f"TX {wifi.get('tx_rate')} Mbps")
    else:
        st.caption("Ethernet (or no WiFi radio) — RSSI/TX are not measured.")
    st.markdown("**Features used by the model:** `" + "`, `".join(ALL_FEATURES) + "`")


if page == "Live Monitor":
    page_live_monitor()
elif page == "Topology":
    page_topology()
elif page == "Analytics":
    page_analytics()
elif page == "Logs":
    page_logs()
elif page == "Healing":
    page_healing()
elif page == "Reports":
    page_reports()
elif page == "Assistant":
    page_chat()
else:
    page_how()
