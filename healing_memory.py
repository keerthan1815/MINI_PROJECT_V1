"""
Explainable AI-Based Predictive Failure Detection for
Network Devices Using XGBoost and SHAP

System layer: Self-Healing Layer (action memory for network recovery).

Algorithms / techniques:
    - CSV-backed case memory (issue → successful action)
    - First-success lookup for recommended healing steps

Inputs:
    - Issue code (e.g. DNS_SLOW, NIC_ERRORS, ROUTER_RTT)
    - Action name and success flag written after a healing attempt

Outputs:
    - healing_memory.csv
    - get_best_action(issue) → previously successful network-side action

Research reference:
    Alghamdi et al. (2025), IJISRT,
    "Artificial Intelligence for Predictive Failures of Network Devices"
"""

import csv
import os

MEMORY_FILE = "healing_memory.csv"


# Create the healing-memory store used after predicted device failures.
def init_memory():
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["issue", "action", "success"])


# Record whether a DNS flush, port flag, or router check was logged as useful.
def save_memory(issue, action, success):
    with open(MEMORY_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([issue, action, success])


# Recall a prior successful recovery for the same network-device symptom class.
def get_best_action(issue):

    if not os.path.exists(MEMORY_FILE):
        return None

    best_action = None
    best_score = 0

    with open(MEMORY_FILE, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["issue"] == issue and row["success"] == "1":
                return row["action"]

    return best_action
