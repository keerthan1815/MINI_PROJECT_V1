"""
Shared feature schema for network DEVICE failure prediction.

This project does not monitor PC CPU or RAM.
It observes symptoms of failing network devices (router, switch,
WiFi AP, firewall/ISP path) from client-side measurements.
"""

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

ROLLING_FEATURES = [
    "router_latency_rolling5",
    "dns_latency_rolling5",
    "rssi_rolling5",
]

TREND_FEATURES = [
    "router_trend",
    "dns_trend",
    "rssi_trend",
]

ALL_FEATURES = RAW_FEATURES + ROLLING_FEATURES + TREND_FEATURES  # 15

SIM_DEVICES = ["Router", "Switch", "Firewall"]
REAL_DEVICE = "My Network"

TIMEOUT_MS = 9999.0

# Paper-facing mapping: metric → suspected device
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
