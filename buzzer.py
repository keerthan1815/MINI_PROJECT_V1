"""
Explainable AI-Based Predictive Failure Detection for
Network Devices Using XGBoost and SHAP

System layer: Alerting Layer (local audible alarm on critical device failure).

Algorithms / techniques:
    - Background daemon thread
    - Windows winsound.Beep (or idle wait on other OS)

Inputs:
    - start_buzzer() / stop_buzzer() calls from the live dashboard
    - Driven when overall path status is CRITICAL

Outputs:
    - Continuous beep while a predicted device failure remains critical

Research reference:
    Alghamdi et al. (2025), IJISRT,
    "Artificial Intelligence for Predictive Failures of Network Devices"
"""

import threading
import platform

running = False

IS_WINDOWS = platform.system().lower() == "windows"

if IS_WINDOWS:
    import winsound


# Loop an alarm so operators hear a predicted router/switch/firewall outage.
def _alarm():
    while running:
        if IS_WINDOWS:
            winsound.Beep(4000, 1500)
        else:
            import time
            time.sleep(1.5)


# Begin audible warning when XGBoost + RCA mark the path as CRITICAL.
def start_buzzer():
    global running
    if not running:
        running = True
        threading.Thread(target=_alarm, daemon=True).start()


# Silence the alarm once predicted device health returns to warning or healthy.
def stop_buzzer():
    global running
    running = False
