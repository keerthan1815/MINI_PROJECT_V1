"""
conftest.py – Comprehensive shared pytest fixtures for network failure detection tests.

Provides:
  1. Per-device & per-scenario NetworkSimulator instances
  2. Telemetry reading fixtures (healthy, degraded, critical, wifi-impaired, packet-storm)
  3. Edge case and boundary value readings (boundary latency, packet loss, extreme metrics)
  4. Malformed and corrupted reading dictionaries for exception handling tests
  5. Mock database fixture (in-memory SQLite with full schema)
  6. Pipeline feature DataFrame fixture with precomputed rolling and trend windows
  7. Factory fixtures for bulk readings and simulated event injection
"""

import os
import sys
import tempfile
import sqlite3
import pytest
import pandas as pd
import numpy as np

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from network_simulator import NetworkSimulator
from config import (
    ALL_FEATURES,
    RAW_FEATURES,
    ROLLING_FEATURES,
    TREND_FEATURES,
    FAILURE_SCENARIOS,
    ROUTER_LATENCY_DEAD,
)


# ---------------------------------------------------------------------------
# 1. Per-device and per-scenario simulators
# ---------------------------------------------------------------------------

@pytest.fixture()
def router_sim():
    """A fresh Router simulator."""
    return NetworkSimulator("Router")


@pytest.fixture()
def switch_sim():
    """A fresh Switch simulator."""
    return NetworkSimulator("Switch")


@pytest.fixture()
def firewall_sim():
    """A fresh Firewall simulator."""
    return NetworkSimulator("Firewall")


@pytest.fixture()
def all_sims(router_sim, switch_sim, firewall_sim):
    """Dict of {device_name: simulator} for multi-device tests."""
    return {
        "Router": router_sim,
        "Switch": switch_sim,
        "Firewall": firewall_sim,
    }


@pytest.fixture()
def scenario_simulators():
    """Returns a dict of simulators, one for each failure scenario."""
    return {scenario: NetworkSimulator(scenario_name=scenario) for scenario in FAILURE_SCENARIOS}


# ---------------------------------------------------------------------------
# 2. Baseline and failure reading fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def healthy_reading():
    """Nominal healthy baseline telemetry reading."""
    return {
        "timestamp": 1726000000.0,
        "scenario": "Network",
        "device": "Router",
        "router_latency": 10.0,
        "router_packet_loss": 0.0,
        "dns_latency": 35.0,
        "dns_packet_loss": 0.0,
        "rssi": -55.0,
        "tx_rate": 250.0,
        "jitter": 2.0,
        "nic_errors": 0.0,
        "traffic": 45.0,
        # Backward-compatible aliases
        "router_latency_ms": 10.0,
        "dns_latency_ms": 35.0,
        "rssi_dbm": -55.0,
        "tx_rate_mbps": 250.0,
        "jitter_ms": 2.0,
        "nic_errors_per_sec": 0.0,
        "traffic_kbps": 45.0,
        "_wifi_available": True,
        "is_failure": 0,
        "failure_type": "none",
        "minutes_to_failure": 10.0,
    }


@pytest.fixture()
def degraded_reading():
    """Telemetry reading in pre-failure degradation phase."""
    return {
        "timestamp": 1726000005.0,
        "scenario": "router_congestion",
        "device": "Router",
        "router_latency": 75.0,
        "router_packet_loss": 8.0,
        "dns_latency": 45.0,
        "dns_packet_loss": 2.0,
        "rssi": -60.0,
        "tx_rate": 180.0,
        "jitter": 18.0,
        "nic_errors": 1.0,
        "traffic": 120.0,
        "router_latency_ms": 75.0,
        "dns_latency_ms": 45.0,
        "rssi_dbm": -60.0,
        "tx_rate_mbps": 180.0,
        "jitter_ms": 18.0,
        "nic_errors_per_sec": 1.0,
        "traffic_kbps": 120.0,
        "_wifi_available": True,
        "is_failure": 1,
        "failure_type": "none",
        "minutes_to_failure": 4.5,
    }


@pytest.fixture()
def critical_reading():
    """Telemetry reading in hard failure state (router unreachable)."""
    return {
        "timestamp": 1726000010.0,
        "scenario": "router_unreachable",
        "device": "Router",
        "router_latency": ROUTER_LATENCY_DEAD,
        "router_packet_loss": 100.0,
        "dns_latency": ROUTER_LATENCY_DEAD,
        "dns_packet_loss": 100.0,
        "rssi": -90.0,
        "tx_rate": 1.0,
        "jitter": 85.0,
        "nic_errors": 25.0,
        "traffic": 0.0,
        "router_latency_ms": ROUTER_LATENCY_DEAD,
        "dns_latency_ms": ROUTER_LATENCY_DEAD,
        "rssi_dbm": -90.0,
        "tx_rate_mbps": 1.0,
        "jitter_ms": 85.0,
        "nic_errors_per_sec": 25.0,
        "traffic_kbps": 0.0,
        "_wifi_available": True,
        "is_failure": 1,
        "failure_type": "router_unreachable",
        "minutes_to_failure": 0.0,
    }


@pytest.fixture()
def wifi_degraded_reading():
    """Telemetry reading representing RF signal degradation."""
    return {
        "timestamp": 1726000015.0,
        "scenario": "wifi_degradation",
        "device": "Router",
        "router_latency": 25.0,
        "router_packet_loss": 5.0,
        "dns_latency": 50.0,
        "dns_packet_loss": 2.0,
        "rssi": -88.0,
        "tx_rate": 12.0,
        "jitter": 15.0,
        "nic_errors": 0.0,
        "traffic": 30.0,
        "router_latency_ms": 25.0,
        "dns_latency_ms": 50.0,
        "rssi_dbm": -88.0,
        "tx_rate_mbps": 12.0,
        "jitter_ms": 15.0,
        "nic_errors_per_sec": 0.0,
        "traffic_kbps": 30.0,
        "_wifi_available": True,
        "is_failure": 1,
        "failure_type": "wifi_degradation",
        "minutes_to_failure": 2.0,
    }


@pytest.fixture()
def packet_storm_reading():
    """Telemetry reading representing massive packet loss across paths."""
    return {
        "timestamp": 1726000020.0,
        "scenario": "packet_storm",
        "device": "Switch",
        "router_latency": 150.0,
        "router_packet_loss": 45.0,
        "dns_latency": 200.0,
        "dns_packet_loss": 50.0,
        "rssi": -65.0,
        "tx_rate": 80.0,
        "jitter": 40.0,
        "nic_errors": 35.0,
        "traffic": 480.0,
        "router_latency_ms": 150.0,
        "dns_latency_ms": 200.0,
        "rssi_dbm": -65.0,
        "tx_rate_mbps": 80.0,
        "jitter_ms": 40.0,
        "nic_errors_per_sec": 35.0,
        "traffic_kbps": 480.0,
        "_wifi_available": True,
        "is_failure": 1,
        "failure_type": "packet_storm",
        "minutes_to_failure": 0.0,
    }


# ---------------------------------------------------------------------------
# 3. Edge-case and boundary value fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def boundary_readings():
    """Dictionary of boundary condition telemetry readings for threshold testing."""
    return {
        "warning_boundary_low": {
            "router_latency": 59.9,
            "router_packet_loss": 4.9,
            "dns_latency": 79.9,
            "dns_packet_loss": 4.9,
            "rssi": -69.9,
            "tx_rate": 50.1,
            "jitter": 9.9,
            "nic_errors": 4.9,
            "traffic": 499.0,
        },
        "warning_boundary_high": {
            "router_latency": 60.0,
            "router_packet_loss": 5.0,
            "dns_latency": 80.0,
            "dns_packet_loss": 5.0,
            "rssi": -70.0,
            "tx_rate": 50.0,
            "jitter": 10.0,
            "nic_errors": 5.0,
            "traffic": 500.0,
        },
        "critical_boundary_low": {
            "router_latency": 99.9,
            "router_packet_loss": 19.9,
            "dns_latency": 149.9,
            "dns_packet_loss": 19.9,
            "rssi": -84.9,
            "tx_rate": 20.1,
            "jitter": 29.9,
            "nic_errors": 19.9,
            "traffic": 499.0,
        },
        "critical_boundary_high": {
            "router_latency": 100.0,
            "router_packet_loss": 20.0,
            "dns_latency": 150.0,
            "dns_packet_loss": 20.0,
            "rssi": -85.0,
            "tx_rate": 20.0,
            "jitter": 30.0,
            "nic_errors": 20.0,
            "traffic": 500.0,
        },
    }


@pytest.fixture()
def malformed_readings():
    """List of invalid or corrupted readings to test exception handling."""
    return [
        {},  # completely empty
        {"router_latency": None, "dns_latency": np.nan},  # None and NaN values
        {"router_latency": "invalid_string", "rssi": [1, 2, 3]},  # wrong types
        {"router_latency": float("inf"), "jitter": -float("inf")},  # infinite values
        {k: -999.0 for k in RAW_FEATURES},  # negative values outside physical bounds
    ]


# ---------------------------------------------------------------------------
# 4. Database & Feature DataFrame fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_db():
    """Provides a fresh, temporary SQLite database connection for test isolation."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS network_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            scenario TEXT,
            device TEXT,
            router_latency REAL,
            router_packet_loss REAL,
            dns_latency REAL,
            dns_packet_loss REAL,
            rssi REAL,
            tx_rate REAL,
            jitter REAL,
            nic_errors REAL,
            traffic REAL,
            is_failure INTEGER,
            failure_type TEXT,
            minutes_to_failure REAL,
            health_score REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS failure_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            scenario TEXT,
            device TEXT,
            failure_type TEXT,
            severity TEXT,
            health_score REAL,
            predicted_minutes REAL,
            reasons TEXT
        )
    """)
    conn.commit()
    yield {"path": db_path, "conn": conn}
    conn.close()
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass


@pytest.fixture(scope="module")
def feature_sample_df():
    """Constructs a clean 50-row DataFrame with all 15 engineered features populated."""
    sim = NetworkSimulator("router_congestion")
    readings = [sim.next_reading() for _ in range(50)]
    df = pd.DataFrame(readings)

    window = 5
    df["router_latency_rolling5"] = df["router_latency"].rolling(window, min_periods=1).mean().round(2)
    df["dns_latency_rolling5"] = df["dns_latency"].rolling(window, min_periods=1).mean().round(2)
    df["rssi_rolling5"] = df["rssi"].rolling(window, min_periods=1).mean().round(2)

    df["router_latency_trend"] = (df["router_latency"] - df["router_latency"].shift(window)).fillna(0.0).round(2)
    df["dns_latency_trend"] = (df["dns_latency"] - df["dns_latency"].shift(window)).fillna(0.0).round(2)
    df["rssi_trend"] = (df["rssi"] - df["rssi"].shift(window)).fillna(0.0).round(2)

    return df


@pytest.fixture()
def bulk_readings():
    """Factory fixture to collect n readings from a simulator."""
    def _get(sim: NetworkSimulator, n: int = 100):
        return [sim.next_reading() for _ in range(n)]
    return _get
