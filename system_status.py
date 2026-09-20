"""
system_status.py — live health of network devices (not PCs).
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


def update_device_status(device, severity):
    status[device] = SEVERITY_TO_STATUS.get(severity, "healthy")


def get_overall_status():
    watched = [status.get(d, "healthy") for d in
               ["Router", "Switch", "Firewall", "My Network"]]
    if any(v == "failure" for v in watched):
        return "CRITICAL"
    if any(v == "warning" for v in watched):
        return "WARNING"
    return "HEALTHY"


def get_status_emoji(device):
    s = status.get(device, "healthy")
    return {"healthy": "🟢", "warning": "🟡", "failure": "🔴"}.get(s, "⚪")
