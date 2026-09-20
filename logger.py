"""
Explainable AI-Based Predictive Failure Detection for
Network Devices Using XGBoost and SHAP

System layer: Persistence Layer (runtime path-status log).

Algorithms / techniques:
    - Dual write: CSV append + SQLite insert
    - Records XGBoost prediction, risk, and suspected failing device

Inputs:
    - Timestamp, UP/DOWN path status, AI prediction text, risk %, device name

Outputs:
    - network_logs.csv
    - network_logs rows used later for SLA (Alghamdi-style availability tracking)

Research reference:
    Alghamdi et al. (2025), IJISRT,
    "Artificial Intelligence for Predictive Failures of Network Devices"
"""

import os
import pandas as pd
from database import insert_network_log


# Persist one monitoring tick so SLA and audits know if a device path was DOWN.
def log_data(time, network, prediction, risk, failing_device=""):
    file = "network_logs.csv"
    new_row = pd.DataFrame([{
        "Time": time,
        "Network Status": network,
        "AI Prediction": prediction,
        "Risk": risk,
        "Failing Device": failing_device,
    }])
    if os.path.exists(file):
        new_row.to_csv(file, mode="a", header=False, index=False)
    else:
        new_row.to_csv(file, index=False)
    try:
        insert_network_log(network, prediction, risk, failing_device)
    except Exception:
        pass
