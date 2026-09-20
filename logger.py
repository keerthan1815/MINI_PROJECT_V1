"""CSV + SQLite logger for overall network path status."""

import os
import pandas as pd
from database import insert_network_log


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
