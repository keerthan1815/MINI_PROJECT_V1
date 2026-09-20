import psutil
import socket
import platform
import subprocess


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


def check_internet():
    """Internet UP/DOWN check"""

    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return "UP"
    except:
        return "DOWN"


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