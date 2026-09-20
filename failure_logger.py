"""Persist predicted network-device failures."""

import csv
import os
from datetime import datetime
from database import insert_alert

FAILURE_CSV = "failure_history.csv"


def log_failure(device, metrics, severity, lead_time, reasons):
    file_exists = os.path.exists(FAILURE_CSV)
    with open(FAILURE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists or os.stat(FAILURE_CSV).st_size == 0:
            writer.writerow([
                "Time", "Device",
                "router_latency_ms", "router_packet_loss",
                "dns_latency_ms", "dns_packet_loss",
                "rssi_dbm", "tx_rate_mbps", "jitter_ms",
                "nic_errors_per_sec", "traffic_kbps",
                "Severity", "Lead_Time", "Reasons",
            ])
        writer.writerow([
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
            severity, lead_time,
            " | ".join(reasons) if isinstance(reasons, list) else reasons,
        ])
    insert_alert(
        device=device,
        severity=severity,
        lead_time=lead_time,
        reasons=reasons if isinstance(reasons, list) else [reasons],
    )
