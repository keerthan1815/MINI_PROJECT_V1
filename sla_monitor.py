"""
sla_monitor.py - SLA calculation using the SQLite database.

SLA = (uptime readings / total readings) x 100
"""

from database import get_sla


def calculate_sla():
    """
    Returns (sla_percent, uptime_count, downtime_count).
    Reads from network_logs table in network_monitor.db.
    """
    return get_sla()
