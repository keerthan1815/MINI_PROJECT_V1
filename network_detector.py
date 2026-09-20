"""
Explainable AI-Based Predictive Failure Detection for
Network Devices Using XGBoost and SHAP

System layer: Data Collection Layer (link-type and coarse path liveness).

Algorithms / techniques:
    - Windows netsh wlan (Wi-Fi vs Ethernet)
    - TCP probe to 8.8.8.8:53 as a coarse Internet-up check
    - psutil imported for host NIC context (not CPU/RAM scoring)

Inputs:
    - Local OS network configuration (Windows netsh / socket)

Outputs:
    - Connection type string (WiFi / LAN)
    - Internet UP/DOWN
    - Coarse healthy/failure flags for Router/Switch when the WAN probe fails

Research reference:
    Alghamdi et al. (2025), IJISRT,
    "Artificial Intelligence for Predictive Failures of Network Devices"
"""

import psutil
import socket
import platform
import subprocess


# Detect whether the observer host is on Wi-Fi (AP path) or Ethernet (switch path).
def get_network_type():
    """WiFi ya LAN detect karega (Windows)"""

    if platform.system().lower() == "windows":
        try:
            output = subprocess.check_output(
                "netsh wlan show interfaces",
                shell=True,
                text=True
            )

            if "SSID" in output and "State" in output:
                return "WiFi Connected"

        except:
            pass

        return "LAN / Ethernet"

    return "Unknown"


# Probe whether the WAN/firewall path can still reach the public Internet.
def check_internet():
    """Internet UP/DOWN check"""

    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return "UP"
    except:
        return "DOWN"


# Fallback device flags when live XGBoost inference is not yet running.
def get_device_status():
    """Server / Router / Switch simulation (basic)"""

    internet = check_internet()

    status = {
        "Server": "healthy",
        "Router": "healthy" if internet == "UP" else "failure",
        "Switch": "healthy" if internet == "UP" else "failure",
        "Internet": internet,
        "Network_Type": get_network_type()
    }

    return status
