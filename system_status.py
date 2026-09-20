"""
Explainable AI-Based Predictive Failure Detection for
Network Devices Using XGBoost and SHAP

System layer: State Layer (live health of infrastructure devices).

Algorithms / techniques:
    - Severity-to-status mapping from RCA after an XGBoost prediction
    - Worst-device aggregation for overall path health

Inputs:
    - Device name and RCA severity (CRITICAL / MEDIUM / LOW / Normal)

Outputs:
    - In-memory status dict used by topology, chat, PDF, and buzzer
    - Overall HEALTHY / WARNING / CRITICAL for the observed path

Research reference:
    Alghamdi et al. (2025), IJISRT,
    "Artificial Intelligence for Predictive Failures of Network Devices"
"""

status = {
    "Internet": "healthy",
    "Router": "healthy",
    "Switch": "healthy",
    "Firewall": "healthy",
    "My Network": "healthy",
}

SEVERITY_TO_STATUS = {
    "CRITICAL": "failure",
    "MEDIUM": "warning",
    "LOW": "warning",
    "Normal": "healthy",
}


# Update topology colour for a router, switch, firewall, or live path after a prediction.
def update_device_status(device, severity):
    status[device] = SEVERITY_TO_STATUS.get(severity, "healthy")


# Roll up individual device states into one operator-facing path status.
def get_overall_status():
    watched = [status.get(d, "healthy") for d in
               ["Router", "Switch", "Firewall", "My Network"]]
    if any(v == "failure" for v in watched):
        return "CRITICAL"
    if any(v == "warning" for v in watched):
        return "WARNING"
    return "HEALTHY"


# Emoji used on topology / assistant views for a given infrastructure node.
def get_status_emoji(device):
    s = status.get(device, "healthy")
    return {"healthy": "🟢", "warning": "🟡", "failure": "🔴"}.get(s, "⚪")
