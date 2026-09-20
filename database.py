"""
Explainable AI-Based Predictive Failure Detection for
Network Devices Using XGBoost and SHAP

System layer: Persistence Layer (SQLite for readings, alerts, SLA, healing).

Algorithms / techniques:
    - SQLite schema for nine network-device metrics (legacy CPU columns dropped)
    - JSON storage of RCA reasons and optional SHAP values
    - SLA numerator/denominator from UP/DOWN path logs

Inputs:
    - Metric dicts, XGBoost prediction labels, confidence, health, RCA fields

Outputs:
    - network_monitor.db tables: device_readings, alerts, network_logs, healing_logs

Research reference:
    Alghamdi et al. (2025), IJISRT,
    "Artificial Intelligence for Predictive Failures of Network Devices"
"""

import json
import sqlite3
from datetime import datetime

DB_FILE = "network_monitor.db"


# Open the failure-monitoring database (shared with the Streamlit dashboard thread).
def _conn():
    c = sqlite3.connect(DB_FILE, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


# Create tables for device symptoms, predicted failures, SLA samples, and healing.
def init_db():
    con = _conn()
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS device_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        device TEXT,
        router_latency_ms REAL,
        router_packet_loss REAL,
        dns_latency_ms REAL,
        dns_packet_loss REAL,
        rssi_dbm REAL,
        tx_rate_mbps REAL,
        jitter_ms REAL,
        nic_errors_per_sec REAL,
        traffic_kbps REAL,
        prediction TEXT,
        confidence REAL,
        health_score REAL,
        severity TEXT,
        failing_device TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        device TEXT,
        severity TEXT,
        lead_time TEXT,
        reasons TEXT,
        shap_values TEXT,
        resolved INTEGER DEFAULT 0
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS network_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time TEXT,
        network_status TEXT,
        prediction TEXT,
        risk REAL,
        failing_device TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS healing_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        device TEXT,
        issue TEXT,
        action TEXT,
        result TEXT
    )""")
    con.commit()
    # Legacy PC-metric schema cannot store device symptoms.
    for table in ("device_readings", "network_logs"):
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()]
        if "cpu" in cols:
            con.execute(f"DROP TABLE {table}")
            con.commit()
            con.close()
            init_db()
            return
    con.close()


# Store one XGBoost inference tick: nine metrics plus predicted failing device.
def insert_reading(device, metrics, prediction, confidence, health_score,
                   severity, failing_device=""):
    con = _conn()
    con.execute("""
    INSERT INTO device_readings
      (timestamp,device,router_latency_ms,router_packet_loss,dns_latency_ms,
       dns_packet_loss,rssi_dbm,tx_rate_mbps,jitter_ms,nic_errors_per_sec,
       traffic_kbps,prediction,confidence,health_score,severity,failing_device)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        device,
        metrics.get("router_latency_ms"),
        metrics.get("router_packet_loss"),
        metrics.get("dns_latency_ms"),
        metrics.get("dns_packet_loss"),
        metrics.get("rssi_dbm"),
        metrics.get("tx_rate_mbps"),
        metrics.get("jitter_ms"),
        metrics.get("nic_errors_per_sec"),
        metrics.get("traffic_kbps"),
        prediction, confidence, health_score, severity, failing_device,
    ))
    con.commit()
    con.close()


# Persist a predicted device-failure alert (severity, lead time, SHAP/RCA text).
def insert_alert(device, severity, lead_time, reasons, shap_values=None):
    con = _conn()
    con.execute("""
    INSERT INTO alerts (timestamp,device,severity,lead_time,reasons,shap_values)
    VALUES (?,?,?,?,?,?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        device, severity, lead_time,
        json.dumps(reasons),
        json.dumps(shap_values) if shap_values else "{}",
    ))
    con.commit()
    con.close()


# Record whether the observed path was UP or DOWN after this prediction (for SLA).
def insert_network_log(network_status, prediction, risk, failing_device=""):
    con = _conn()
    con.execute("""
    INSERT INTO network_logs (time,network_status,prediction,risk,failing_device)
    VALUES (?,?,?,?,?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        network_status, prediction, risk, failing_device,
    ))
    con.commit()
    con.close()


# Log a network-side recovery action taken after a predicted device failure.
def insert_healing_log(device, issue, action, result):
    con = _conn()
    con.execute("""
    INSERT INTO healing_logs (timestamp,device,issue,action,result)
    VALUES (?,?,?,?,?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          device, issue, action, result))
    con.commit()
    con.close()


# Fetch recent metric+prediction rows for analytics charts on a device or all paths.
def get_recent_readings(device=None, limit=100):
    con = _conn()
    if device:
        rows = con.execute(
            "SELECT * FROM device_readings WHERE device=? ORDER BY timestamp DESC LIMIT ?",
            (device, limit)).fetchall()
    else:
        rows = con.execute(
            "SELECT * FROM device_readings ORDER BY timestamp DESC LIMIT ?",
            (limit,)).fetchall()
    con.close()
    return [dict(r) for r in rows]


# Return recent predicted-failure alerts for the dashboard, PDF, and assistant.
def get_alerts(limit=50):
    con = _conn()
    rows = con.execute(
        "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


# Count alerts per router/switch/firewall so operators see which device fails most.
def get_failure_count_by_device():
    con = _conn()
    rows = con.execute(
        "SELECT device, COUNT(*) as cnt FROM alerts GROUP BY device"
    ).fetchall()
    con.close()
    return {r["device"]: r["cnt"] for r in rows}


# Retrieve DNS-flush / port-flag / router-check history after predicted failures.
def get_healing_logs(limit=50):
    con = _conn()
    rows = con.execute(
        "SELECT * FROM healing_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


# Compute path availability from logged UP vs DOWN samples after XGBoost ticks.
def get_sla():
    con = _conn()
    total = con.execute("SELECT COUNT(*) FROM network_logs").fetchone()[0]
    down = con.execute(
        "SELECT COUNT(*) FROM network_logs WHERE network_status='DOWN'"
    ).fetchone()[0]
    con.close()
    if total == 0:
        return 100.0, 0, 0
    uptime = total - down
    return round((uptime / total) * 100, 2), uptime, down


init_db()
