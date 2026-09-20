"""
Explainable AI-Based Predictive Failure Detection for
Network Devices Using XGBoost and SHAP

System layer: Scoring Layer (interpretable device-health index).

Algorithms / techniques:
    - Weighted penalty scoring on nine network symptoms
    - Blend with XGBoost failure probability
    - RSSI quality bands for Wi-Fi AP health

Inputs:
    - Metric reading dict (gateway RTT, loss, DNS, RSSI, TX, jitter, NIC errors, traffic)
    - failure_prob from the XGBoost classifier (0–1)

Outputs:
    - Health score 0–100 (device path, not PC)
    - Qualitative label + RSSI quality string for the dashboard

Research reference:
    Alghamdi et al. (2025), IJISRT,
    "Artificial Intelligence for Predictive Failures of Network Devices"
"""

from features import TIMEOUT_MS


# Convert live network-device symptoms + XGBoost risk into a 0–100 health score.
def calculate_health_score(reading, failure_prob):
    score = 100.0
    r_lat = reading.get("router_latency_ms", 0)
    r_loss = reading.get("router_packet_loss", 0)
    d_lat = reading.get("dns_latency_ms", 0)
    d_loss = reading.get("dns_packet_loss", 0)
    rssi = reading.get("rssi_dbm", -55)
    tx = reading.get("tx_rate_mbps", 200)
    jitter = reading.get("jitter_ms", 0)
    errs = reading.get("nic_errors_per_sec", 0)
    traffic = reading.get("traffic_kbps", 0)

    if r_lat >= TIMEOUT_MS:
        score -= 28
    elif r_lat > 20:
        score -= min((r_lat - 20) * 0.25, 20)

    if r_loss > 5:
        score -= min(r_loss * 0.45, 25)

    if d_lat >= TIMEOUT_MS:
        score -= 22
    elif d_lat > 80:
        score -= min((d_lat - 80) * 0.08, 18)

    if d_loss > 10:
        score -= min(d_loss * 0.3, 18)

    if rssi < -70:
        score -= min((-rssi - 70) * 1.4, 20)

    if tx < 50:
        score -= min((50 - tx) * 0.2, 12)

    if jitter > 20:
        score -= min((jitter - 20) * 0.25, 12)

    score -= min(errs * 2.5, 18)

    if traffic > 200:
        score -= min((traffic - 200) * 0.03, 10)

    score -= failure_prob * 30
    return max(0.0, min(100.0, round(score, 1)))


# Map the numeric path score to an operator-facing label (excellent → critical).
def score_to_label(score):
    if score >= 90:
        return "Excellent", "🟢"
    if score >= 70:
        return "Good", "🟢"
    if score >= 50:
        return "Fair", "🟡"
    if score >= 30:
        return "Poor", "🟠"
    return "Critical", "🔴"


# Interpret Wi-Fi RSSI as AP / radio health, not as laptop battery or CPU load.
def rssi_quality(rssi_dbm):
    if rssi_dbm >= -50:
        return "Excellent signal"
    if rssi_dbm >= -60:
        return "Good signal"
    if rssi_dbm >= -70:
        return "Fair signal"
    if rssi_dbm >= -80:
        return "Weak — AP struggling or far"
    return "Very weak — AP / RF failure likely"
