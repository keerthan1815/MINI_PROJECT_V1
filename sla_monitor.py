"""
Explainable AI-Based Predictive Failure Detection for
Network Devices Using XGBoost and SHAP

System layer: Monitoring / SLA Layer.

Algorithms / techniques:
    - Availability = (uptime samples / total samples) × 100
    - SQLite aggregation via database.get_sla()

Inputs:
    - network_logs table (UP vs DOWN path samples after each prediction)

Outputs:
    - (sla_percent, uptime_count, downtime_count) for dashboard and PDF reports

Research reference:
    Alghamdi et al. (2025), IJISRT,
    "Artificial Intelligence for Predictive Failures of Network Devices"
"""

from database import get_sla


# Compute path availability after XGBoost has labelled samples UP or DOWN.
def calculate_sla():
    """
    Returns (sla_percent, uptime_count, downtime_count).
    Reads from network_logs table in network_monitor.db.
    """
    return get_sla()
