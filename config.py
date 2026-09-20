"""
config.py — Single source of truth for:
    "Explainable AI-Based Predictive Failure Detection for
     Network Devices Using XGBoost and SHAP"

All project-wide constants are defined here. Import from this module
instead of hardcoding values anywhere else in the codebase.
"""

# ---------------------------------------------------------------------------
# Project identity
# ---------------------------------------------------------------------------
PROJECT_TITLE = (
    "Explainable AI-Based Predictive Failure Detection "
    "for Network Devices Using XGBoost and SHAP"
)

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

# Raw sensor / telemetry features collected per polling cycle
RAW_FEATURES = [
    "router_latency",
    "router_packet_loss",
    "dns_latency",
    "dns_packet_loss",
    "rssi",
    "tx_rate",
    "jitter",
    "nic_errors",
    "traffic",
]

# Rolling-window (window=5) smoothed versions of key metrics
ROLLING_FEATURES = [
    "router_latency_rolling5",
    "dns_latency_rolling5",
    "rssi_rolling5",
]

# Linear trend (slope) computed over the rolling window
TREND_FEATURES = [
    "router_latency_trend",
    "dns_latency_trend",
    "rssi_trend",
]

# Complete feature vector fed into the ML models
ALL_FEATURES = RAW_FEATURES + ROLLING_FEATURES + TREND_FEATURES

# ---------------------------------------------------------------------------
# Target / label columns
# ---------------------------------------------------------------------------

# Binary classification label  (1 = failure imminent, 0 = healthy)
CLASS_LABEL = "is_failure"

# Regression label — how many minutes until the next failure event
REGRESSION_LABEL = "minutes_to_failure"

# ---------------------------------------------------------------------------
# Thresholds — Router latency (ms)
# ---------------------------------------------------------------------------
ROUTER_LATENCY_WARNING  = 60      # Elevated — monitor closely
ROUTER_LATENCY_CRITICAL = 100     # Severe — intervention likely needed
ROUTER_LATENCY_DEAD     = 9999    # Sentinel value — router unreachable

# ---------------------------------------------------------------------------
# Thresholds — Router packet loss (%)
# ---------------------------------------------------------------------------
ROUTER_LOSS_WARNING  = 5          # Minor loss — acceptable short-term
ROUTER_LOSS_CRITICAL = 20         # Heavy loss — link likely degraded

# ---------------------------------------------------------------------------
# Thresholds — DNS latency (ms)
# ---------------------------------------------------------------------------
DNS_LATENCY_WARNING  = 200        # Sluggish DNS — possible ISP issue
DNS_LATENCY_CRITICAL = 500        # Severe DNS delay — near-outage
DNS_LATENCY_DEAD     = 9999       # Sentinel value — DNS unreachable

# ---------------------------------------------------------------------------
# Thresholds — WiFi signal strength (dBm, values are negative)
# ---------------------------------------------------------------------------
RSSI_WARNING  = -75               # Weak signal — throughput impacted
RSSI_CRITICAL = -85               # Very weak — imminent disconnection

# ---------------------------------------------------------------------------
# Thresholds — Transmit rate (Mbps)
# ---------------------------------------------------------------------------
TX_RATE_WARNING  = 80             # Below expected baseline
TX_RATE_CRITICAL = 40             # Severely degraded throughput

# ---------------------------------------------------------------------------
# Thresholds — Jitter (ms)
# ---------------------------------------------------------------------------
JITTER_WARNING  = 20              # Noticeable jitter — VoIP/video affected
JITTER_CRITICAL = 50              # Severe jitter — real-time traffic unusable

# ---------------------------------------------------------------------------
# Thresholds — NIC errors (count per interval)
# ---------------------------------------------------------------------------
NIC_ERRORS_WARNING  = 5           # Low error rate — watch trend
NIC_ERRORS_CRITICAL = 15          # High error rate — hardware fault likely

# ---------------------------------------------------------------------------
# Thresholds — Traffic volume (Mbps)
# ---------------------------------------------------------------------------
TRAFFIC_WARNING  = 300            # Elevated utilisation — approaching limits
TRAFFIC_CRITICAL = 500            # Bandwidth saturation — congestion likely

# ---------------------------------------------------------------------------
# Thresholds — Time-to-failure (minutes, used by regression model)
# ---------------------------------------------------------------------------
MINUTES_CRITICAL = 2.0            # Failure expected within 2 minutes
MINUTES_WARNING  = 5.0            # Failure expected within 5 minutes

# ---------------------------------------------------------------------------
# Monitoring / runtime parameters
# ---------------------------------------------------------------------------
PING_COUNT       = 4              # ICMP echo requests sent per probe
REFRESH_SECS     = 2              # Dashboard / polling refresh interval (s)
HISTORY_LENGTH   = 60             # Number of historical readings to retain
EMAIL_COOLDOWN   = 120            # Minimum seconds between alert e-mails
HEALING_COOLDOWN = 90             # Minimum seconds between auto-heal actions

# ---------------------------------------------------------------------------
# Data generation / simulation parameters
# ---------------------------------------------------------------------------
SMOTE_RANDOM_STATE      = 42      # Reproducibility seed for SMOTE oversampling
NUM_READINGS_PER_DEVICE = 3000    # Synthetic samples generated per device
SIMULATION_SPEED        = 0.02    # Sleep duration (s) between simulated ticks
ROLLING_WINDOW          = 5       # Window size for rolling stats / trend calc

# ---------------------------------------------------------------------------
# Persistence paths
# ---------------------------------------------------------------------------
MODEL_CLF_PATH = "failure_model.pkl"       # Serialised XGBoost classifier
MODEL_REG_PATH = "regression_model.pkl"    # Serialised XGBoost regressor
TRAINING_DATA  = "training_data.csv"       # Generated / cached training dataset

# Chart artifact output paths
CHART_CONFUSION_MATRIX = "confusion_matrix.png"
CHART_ROC_CURVE        = "roc_curve.png"
CHART_SHAP_BAR         = "shap_summary_bar.png"
CHART_SHAP_BEESWARM    = "shap_beeswarm.png"
CHART_REGRESSION_DIST  = "regression_distribution.png"

# ---------------------------------------------------------------------------
# XGBoost hyperparameters
# ---------------------------------------------------------------------------

# Binary classifier (predict is_failure)
XGBOOST_CLF_PARAMS = {
    "n_estimators":     200,
    "max_depth":        6,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "logloss",
    "random_state":     42,
}

# Regressor (predict minutes_to_failure)
XGBOOST_REG_PARAMS = {
    "n_estimators":     200,
    "max_depth":        6,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "objective":        "reg:squarederror",
    "random_state":     42,
}

# ---------------------------------------------------------------------------
# Failure scenario catalogue
# Used by the simulator and root-cause explainer.
# Keys   → internal scenario ID
# Values → human-readable description shown in alerts / dashboard
# ---------------------------------------------------------------------------
FAILURE_SCENARIOS = {
    "router_congestion":    "Network path to router is congested",
    "router_unreachable":   "Router not responding to any packets",
    "internet_outage":      "DNS unreachable — internet connection lost",
    "wifi_degradation":     "WiFi signal weakening — link quality dropping",
    "physical_layer_fault": "Cable or switch port errors increasing",
    "network_overload":     "Traffic spike — network bandwidth saturated",
    "dns_slowdown":         "DNS server responding slowly — ISP issue",
    "packet_storm":         "High packet loss across all paths",
}
