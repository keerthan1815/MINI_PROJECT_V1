"""
Explainable AI-Based Predictive Failure Detection for
Network Devices Using XGBoost and SHAP

System layer: Feature Schema Layer (shared contract for collection,
training, explainability, and the dashboard).

Algorithms / techniques:
    - Client-side symptom schema (not host CPU/RAM)
    - Metric-to-device mapping used by SHAP attribution and RCA
    - Rolling-window and trend feature names consumed by XGBoost

Inputs:
    - None at runtime (constants imported by other modules)

Outputs:
    - RAW_FEATURES (9 network metrics)
    - ROLLING_FEATURES / TREND_FEATURES (engineered names)
    - ALL_FEATURES (15-column XGBoost input order)
    - METRIC_TO_DEVICE, FEATURE_HELP, device name constants

Research reference:
    Alghamdi et al. (2025), IJISRT,
    "Artificial Intelligence for Predictive Failures of Network Devices"
"""

# Nine raw path symptoms that indicate router / switch / AP / firewall stress.
RAW_FEATURES = [
    "router_latency_ms",
    "router_packet_loss",
    "dns_latency_ms",
    "dns_packet_loss",
    "rssi_dbm",
    "tx_rate_mbps",
    "jitter_ms",
    "nic_errors_per_sec",
    "traffic_kbps",
]

# Short-horizon averages so XGBoost can see a degrading device, not a single spike.
ROLLING_FEATURES = [
    "router_latency_rolling5",
    "dns_latency_rolling5",
    "rssi_rolling5",
]

# Delta over the same window — early warning that a device is trending toward failure.
TREND_FEATURES = [
    "router_trend",
    "dns_trend",
    "rssi_trend",
]

ALL_FEATURES = RAW_FEATURES + ROLLING_FEATURES + TREND_FEATURES  # 15

SIM_DEVICES = ["Router", "Switch", "Firewall"]
REAL_DEVICE = "My Network"

TIMEOUT_MS = 9999.0

# Paper-facing mapping: which network device a SHAP-ranked metric implicates.
METRIC_TO_DEVICE = {
    "router_latency_ms": "Router",
    "router_packet_loss": "Router",
    "router_latency_rolling5": "Router",
    "router_trend": "Router",
    "jitter_ms": "Router",
    "rssi_dbm": "WiFi AP / Router",
    "tx_rate_mbps": "WiFi AP / Switch",
    "rssi_rolling5": "WiFi AP / Router",
    "rssi_trend": "WiFi AP / Router",
    "nic_errors_per_sec": "Switch",
    "dns_latency_ms": "Firewall / ISP",
    "dns_packet_loss": "Firewall / ISP",
    "dns_latency_rolling5": "Firewall / ISP",
    "dns_trend": "Firewall / ISP",
    "traffic_kbps": "Firewall / path overload",
}

FEATURE_HELP = {
    "router_latency_ms": "ICMP RTT to default gateway (router)",
    "router_packet_loss": "% of gateway pings that timed out",
    "dns_latency_ms": "ICMP RTT to configured DNS (WAN / ISP / firewall)",
    "dns_packet_loss": "% of DNS pings that timed out",
    "rssi_dbm": "WiFi received signal strength (AP health / distance)",
    "tx_rate_mbps": "WiFi link rate — AP lowers this when RF quality falls",
    "jitter_ms": "Std. dev. of recent gateway RTTs (instability)",
    "nic_errors_per_sec": "NIC / switch-port error counter rate",
    "traffic_kbps": "Throughput through the local NIC (path load)",
}
