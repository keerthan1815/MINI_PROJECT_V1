"""
Explainable AI-Based Predictive Failure Detection for
Network Devices Using XGBoost and SHAP

System layer: Self-Healing Layer (network-side recovery, not PC process kill).

Algorithms / techniques:
    - Symptom-triggered playbooks (gateway RTT, DNS latency, NIC errors, overload)
    - Windows ipconfig /flushdns or resolvectl flush-caches
    - Operator flags for switch-port and router checks
    - CSV + SQLite healing audit trail

Inputs:
    - Live metric reading, RCA dict, XGBoost failure_risk, device name

Outputs:
    - DNS cache flush when WAN/DNS looks failed
    - healing_logs.csv / SQLite healing_logs
    - healing_memory.csv updates

Research reference:
    Alghamdi et al. (2025), IJISRT,
    "Artificial Intelligence for Predictive Failures of Network Devices"
"""

import csv
import os
import platform
from datetime import datetime

from database import insert_healing_log
from healing_memory import init_memory, save_memory

HEALING_CSV = "healing_logs.csv"


# Audit a recommended or executed recovery so later reports show what was tried.
def _log(device, issue, action, result):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exists = os.path.exists(HEALING_CSV)
    with open(HEALING_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not exists or os.stat(HEALING_CSV).st_size == 0:
            w.writerow(["Time", "Device", "Issue", "Action", "Result"])
        w.writerow([now, device, issue, action, result])
    try:
        insert_healing_log(device, issue, action, result)
    except Exception:
        pass


# Clear the resolver cache when DNS/WAN latency implicates firewall or ISP path.
def heal_dns(device="Router"):
    try:
        if platform.system().lower() == "windows":
            os.system("ipconfig /flushdns >nul 2>&1")
        else:
            os.system("resolvectl flush-caches 2>/dev/null")
        save_memory("DNS_SLOW", "FLUSH_DNS", 1)
        _log(device, "High DNS/WAN latency", "Flush DNS resolver cache", "Completed")
    except Exception as e:
        _log(device, "High DNS/WAN latency", "Flush DNS", f"Failed: {e}")


# Flag a likely failing switch port when NIC/PHY error rate rises.
def flag_switch_port(device="Switch"):
    save_memory("NIC_ERRORS", "FLAG_PORT", 1)
    _log(device, "NIC / switch-port errors",
         "Recommend cable + switch-port inspection",
         "Flagged for operator")


# Recommend a router check when gateway RTT or loss shows the device is failing.
def flag_router(device="Router"):
    save_memory("ROUTER_RTT", "CHECK_GATEWAY", 1)
    _log(device, "High gateway RTT / loss",
         "Recommend router CPU/session check and reboot window",
         "Flagged for operator")


# Log path overload that typically stresses the firewall / edge device.
def flag_overload(device="Firewall"):
    save_memory("TRAFFIC_SPIKE", "LOG_OVERLOAD", 1)
    _log(device, "Path overload",
         "Log traffic spike — check firewall/QoS / DDoS",
         "Alert logged")


# Choose network-only recovery from RCA + metrics after XGBoost predicts failure.
def run_healing(reading, rca, failure_risk=0, device="Unknown"):
    try:
        init_memory()
        r_lat = reading.get("router_latency_ms", 0)
        r_loss = reading.get("router_packet_loss", 0)
        d_lat = reading.get("dns_latency_ms", 0)
        errs = reading.get("nic_errors_per_sec", 0)
        traffic = reading.get("traffic_kbps", 0)
        failing = rca.get("failing_device", device)

        if r_lat > 100 or r_loss > 10:
            flag_router(failing)
        if d_lat > 200:
            heal_dns(failing)
        if errs > 5:
            flag_switch_port(failing)
        if traffic > 300:
            flag_overload(failing)
        if failure_risk > 85 and not (r_lat > 100 or d_lat > 200 or errs > 5):
            flag_router(failing)
    except Exception as e:
        print("Healing error:", e)
