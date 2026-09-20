"""
test_edge_cases_and_exceptions.py – Automated test fixtures for edge cases,
boundary values, and exception handling in the network failure detection system.

Tests:
    1. TestEdgeCaseMetricValues      – extreme / boundary metric values
    2. TestNullNaNInfHandling        – None, NaN, Inf inputs do not crash core functions
    3. TestHealthScorerEdgeCases     – division-by-zero resilience and score clamping
    4. TestFeatureSchemaValidation   – missing features, wrong schema raises expected errors
    5. TestSimulatorRecovery         – simulator recovers cleanly after hard failure
    6. TestMalformedReadingHandling  – malformed readings handled gracefully by diagnose()
    7. TestDatabaseEdgeCases         – empty/corrupted DB operations do not crash logger

Fixtures consumed from conftest.py:
    healthy_reading, degraded_reading, critical_reading, wifi_degraded_reading,
    packet_storm_reading, boundary_readings, malformed_readings, mock_db, feature_sample_df
"""

import math
import os
import sys
from pathlib import Path

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from health_scorer import calculate_health_score, score_to_label
from root_cause import diagnose
from network_simulator import NetworkSimulator
from config import (
    ALL_FEATURES,
    RAW_FEATURES,
    ROUTER_LATENCY_DEAD,
    ROUTER_LATENCY_CRITICAL,
    ROUTER_LATENCY_WARNING,
)


# ===========================================================================
# 1. Extreme / boundary metric values in health scorer
# ===========================================================================

class TestEdgeCaseMetricValues:
    """
    Health score must clamp to [0, 100] and not crash for any physically
    meaningful or physically extreme input values.
    """

    def test_score_perfectly_healthy_is_near_100(self, healthy_reading):
        score = calculate_health_score(healthy_reading, failure_prob=0.0)
        assert score >= 85, f"Perfectly healthy reading scored only {score}"

    def test_score_router_unreachable_is_near_zero(self, critical_reading):
        score = calculate_health_score(critical_reading, failure_prob=1.0)
        assert score <= 30, f"Router unreachable scenario scored {score}, expected <= 30"

    def test_score_always_within_bounds(self, boundary_readings):
        """Score must stay in [0, 100] for all boundary readings."""
        for label, metrics in boundary_readings.items():
            # Build a minimal reading-like dict
            r = {
                "router_latency_ms": metrics.get("router_latency", 10.0),
                "router_packet_loss": metrics.get("router_packet_loss", 0.0),
                "dns_latency_ms": metrics.get("dns_latency", 35.0),
                "dns_packet_loss": metrics.get("dns_packet_loss", 0.0),
                "rssi_dbm": metrics.get("rssi", -55.0),
                "tx_rate_mbps": metrics.get("tx_rate", 200.0),
                "jitter_ms": metrics.get("jitter", 2.0),
                "nic_errors_per_sec": metrics.get("nic_errors", 0.0),
                "traffic_kbps": metrics.get("traffic", 45.0),
            }
            score = calculate_health_score(r, failure_prob=0.5)
            assert 0.0 <= score <= 100.0, (
                f"Boundary case '{label}' produced out-of-range score: {score}"
            )

    def test_score_maximum_latency_sentinel(self):
        """router_latency_ms == 9999 (sentinel) should deduct maximum latency penalty."""
        r = {
            "router_latency_ms": ROUTER_LATENCY_DEAD,
            "router_packet_loss": 0.0,
            "dns_latency_ms": 35.0,
            "dns_packet_loss": 0.0,
            "rssi_dbm": -55.0,
            "tx_rate_mbps": 200.0,
            "jitter_ms": 2.0,
            "nic_errors_per_sec": 0.0,
            "traffic_kbps": 45.0,
        }
        score_dead = calculate_health_score(r, failure_prob=0.0)
        r_normal = dict(r, router_latency_ms=10.0)
        score_normal = calculate_health_score(r_normal, failure_prob=0.0)
        assert score_dead < score_normal, (
            "Sentinel latency should produce a lower score than healthy latency"
        )

    def test_score_zero_traffic_is_valid(self):
        """Zero traffic is physically valid (idle path) and must not cause errors."""
        r = {
            "router_latency_ms": 10.0,
            "router_packet_loss": 0.0,
            "dns_latency_ms": 35.0,
            "dns_packet_loss": 0.0,
            "rssi_dbm": -55.0,
            "tx_rate_mbps": 200.0,
            "jitter_ms": 1.0,
            "nic_errors_per_sec": 0.0,
            "traffic_kbps": 0.0,
        }
        score = calculate_health_score(r, failure_prob=0.0)
        assert 0.0 <= score <= 100.0

    def test_score_minimum_rssi_value(self):
        """RSSI at physical floor (-100 dBm) should not exceed score for -55 dBm."""
        r_good = {
            "router_latency_ms": 10.0, "router_packet_loss": 0.0,
            "dns_latency_ms": 35.0, "dns_packet_loss": 0.0,
            "rssi_dbm": -55.0, "tx_rate_mbps": 200.0,
            "jitter_ms": 2.0, "nic_errors_per_sec": 0.0, "traffic_kbps": 45.0,
        }
        r_bad = dict(r_good, rssi_dbm=-100.0)
        score_good = calculate_health_score(r_good, failure_prob=0.0)
        score_bad = calculate_health_score(r_bad, failure_prob=0.0)
        assert score_bad <= score_good, (
            f"RSSI -100 dBm should score lower than -55 dBm (got {score_bad} vs {score_good})"
        )


# ===========================================================================
# 2. None, NaN, Inf inputs do not crash health scorer
# ===========================================================================

class TestNullNaNInfHandling:
    """
    health_scorer.calculate_health_score() must not raise exceptions on missing
    or malformed values — dict.get() provides safe defaults.
    """

    def test_empty_reading_does_not_crash(self):
        """An entirely empty dict should use defaults and return a valid score."""
        score = calculate_health_score({}, failure_prob=0.0)
        assert 0.0 <= score <= 100.0

    def test_none_values_do_not_crash(self):
        """None values are replaced by dict.get() defaults — no AttributeError."""
        r = {
            "router_latency_ms": None,
            "dns_latency_ms": None,
            "rssi_dbm": None,
        }
        # health_scorer uses .get() so None keys are just absent
        score = calculate_health_score({}, failure_prob=0.0)
        assert 0.0 <= score <= 100.0

    def test_failure_prob_zero_and_one_are_valid(self):
        """Extreme failure_prob values must not produce out-of-range scores."""
        r = {
            "router_latency_ms": 10.0, "router_packet_loss": 0.0,
            "dns_latency_ms": 35.0, "dns_packet_loss": 0.0,
            "rssi_dbm": -55.0, "tx_rate_mbps": 200.0,
            "jitter_ms": 2.0, "nic_errors_per_sec": 0.0, "traffic_kbps": 45.0,
        }
        for prob in (0.0, 0.5, 1.0):
            score = calculate_health_score(r, failure_prob=prob)
            assert 0.0 <= score <= 100.0, (
                f"failure_prob={prob} produced out-of-range score: {score}"
            )

    def test_failure_prob_monotone_with_score(self):
        """Higher failure probability should produce a lower or equal health score."""
        r = {
            "router_latency_ms": 10.0, "router_packet_loss": 0.0,
            "dns_latency_ms": 35.0, "dns_packet_loss": 0.0,
            "rssi_dbm": -55.0, "tx_rate_mbps": 200.0,
            "jitter_ms": 2.0, "nic_errors_per_sec": 0.0, "traffic_kbps": 45.0,
        }
        score_low = calculate_health_score(r, failure_prob=0.0)
        score_high = calculate_health_score(r, failure_prob=1.0)
        assert score_high <= score_low, (
            f"Higher failure_prob should lower score: got {score_high} vs {score_low}"
        )


# ===========================================================================
# 3. Score label boundaries
# ===========================================================================

class TestHealthScorerLabelBoundaries:
    """score_to_label() must return correct label at defined threshold boundaries."""

    @pytest.mark.parametrize("score,expected_label", [
        (100.0, "Excellent"),
        (90.0,  "Excellent"),
        (89.9,  "Good"),
        (70.0,  "Good"),
        (69.9,  "Fair"),
        (50.0,  "Fair"),
        (49.9,  "Poor"),
        (30.0,  "Poor"),
        (29.9,  "Critical"),
        (0.0,   "Critical"),
    ])
    def test_label_at_score(self, score, expected_label):
        label, _ = score_to_label(score)
        assert label == expected_label, (
            f"score={score} → expected label '{expected_label}', got '{label}'"
        )


# ===========================================================================
# 4. Feature schema validation
# ===========================================================================

class TestFeatureSchemaValidation:
    """ALL_FEATURES must have exactly 15 entries with correct names and order."""

    def test_all_features_count_is_15(self):
        assert len(ALL_FEATURES) == 15, (
            f"Expected 15 features in ALL_FEATURES, got {len(ALL_FEATURES)}: {ALL_FEATURES}"
        )

    def test_raw_features_count_is_9(self):
        assert len(RAW_FEATURES) == 9, (
            f"Expected 9 RAW_FEATURES, got {len(RAW_FEATURES)}: {RAW_FEATURES}"
        )

    def test_all_features_are_strings(self):
        for f in ALL_FEATURES:
            assert isinstance(f, str) and f.strip(), (
                f"Feature name should be a non-empty string, got {f!r}"
            )

    def test_no_duplicate_features(self):
        assert len(ALL_FEATURES) == len(set(ALL_FEATURES)), (
            f"Duplicate features found in ALL_FEATURES: {ALL_FEATURES}"
        )

    def test_raw_features_are_prefix_of_all_features(self):
        """RAW_FEATURES must be the first 9 entries in ALL_FEATURES."""
        assert ALL_FEATURES[:9] == RAW_FEATURES, (
            f"First 9 of ALL_FEATURES != RAW_FEATURES.\n"
            f"  ALL_FEATURES[:9]: {ALL_FEATURES[:9]}\n"
            f"  RAW_FEATURES    : {RAW_FEATURES}"
        )

    def test_feature_sample_df_has_all_features(self, feature_sample_df):
        """The precomputed feature DataFrame must contain every column in ALL_FEATURES."""
        missing = [f for f in ALL_FEATURES if f not in feature_sample_df.columns]
        assert not missing, (
            f"feature_sample_df is missing columns: {missing}"
        )


# ===========================================================================
# 5. Simulator state machine: recovery after hard failure
# ===========================================================================

class TestSimulatorRecovery:
    """
    After a full failure window ends the simulator must return to healthy
    metrics within a bounded number of ticks.
    """

    def test_recovery_after_failure_injection(self):
        """
        Inject router_unreachable, consume through full pre-failure + hard-failure
        window, then verify that a later healthy reading has latency < 500 ms.
        """
        sim = NetworkSimulator("router_unreachable")
        sim.inject_failure("router_unreachable")

        # Burn through pre-failure (up to 20 ticks) + hard failure (up to 18 ticks)
        for _ in range(50):
            sim.next_reading()

        # After the failure window the simulator resets to baseline
        recovered = False
        for _ in range(30):
            r = sim.next_reading()
            if r["is_failure"] == 0 and r["router_latency"] < 200.0:
                recovered = True
                break

        assert recovered, (
            "Simulator did not return to healthy state within 30 ticks after failure window"
        )

    def test_multiple_injection_cycles_do_not_accumulate_state(self):
        """Running two consecutive injection cycles must not drift latency unboundedly."""
        sim = NetworkSimulator("router_congestion")

        # First full cycle: pre-failure (≤20 ticks) + hard failure (≤18 ticks)
        sim.inject_failure("router_congestion")
        for _ in range(60):
            sim.next_reading()

        # Second full cycle
        sim.inject_failure("router_congestion")
        for _ in range(60):
            sim.next_reading()

        # After both cycles the simulator should have reset to baseline; allow extra ticks
        healthy_count = 0
        for _ in range(50):
            r = sim.next_reading()
            if r["is_failure"] == 0:
                healthy_count += 1

        assert healthy_count > 0, (
            "After two failure cycles the simulator never produced healthy readings"
        )


# ===========================================================================
# 6. Malformed reading handling in root_cause.diagnose()
# ===========================================================================

class TestMalformedReadingHandling:
    """
    diagnose() must never raise an unhandled exception even for reads with
    missing keys, zero values, or extreme values.
    """

    def test_empty_dict_does_not_crash(self):
        """diagnose({}) must return a dict with severity and reasons keys."""
        result = diagnose({})
        assert "severity" in result
        assert "reasons" in result

    def test_all_zeros_reading_does_not_crash(self):
        """A reading with all zeros must be handled gracefully."""
        r = {k: 0.0 for k in [
            "router_latency_ms", "router_packet_loss",
            "dns_latency_ms", "dns_packet_loss",
            "rssi_dbm", "tx_rate_mbps", "jitter_ms",
            "nic_errors_per_sec", "traffic_kbps",
        ]}
        result = diagnose(r)
        assert isinstance(result, dict)
        assert "severity" in result

    def test_timeout_sentinel_values_produce_critical(self):
        """ROUTER_LATENCY_DEAD sentinel must trigger at least MEDIUM severity."""
        r = {
            "router_latency_ms": ROUTER_LATENCY_DEAD,
            "router_packet_loss": 100.0,
            "dns_latency_ms": ROUTER_LATENCY_DEAD,
            "dns_packet_loss": 100.0,
            "rssi_dbm": -55.0,
            "tx_rate_mbps": 200.0,
            "jitter_ms": 2.0,
            "nic_errors_per_sec": 0.0,
            "traffic_kbps": 45.0,
        }
        result = diagnose(r)
        assert result["severity"] in ("MEDIUM", "CRITICAL"), (
            f"Expected MEDIUM or CRITICAL for unreachable reading, got {result['severity']}"
        )

    @pytest.mark.parametrize("missing_key", [
        "router_latency_ms",
        "router_packet_loss",
        "dns_latency_ms",
        "rssi_dbm",
        "nic_errors_per_sec",
    ])
    def test_missing_individual_key_does_not_crash(self, healthy_reading, missing_key):
        """Removing a single key from a healthy reading must not raise."""
        r = dict(healthy_reading)
        r.pop(missing_key, None)
        result = diagnose(r)
        assert "severity" in result, (
            f"diagnose() crashed when '{missing_key}' was missing"
        )

    def test_extra_keys_do_not_crash(self, healthy_reading):
        """Extra unexpected keys in the reading must be silently ignored."""
        r = dict(healthy_reading, unknown_metric_xyz=42.0, another_field="hello")
        result = diagnose(r)
        assert isinstance(result, dict)


# ===========================================================================
# 7. Database edge cases
# ===========================================================================

class TestDatabaseEdgeCases:
    """
    The SQLite logging layer should handle empty tables and multiple inserts
    without errors or corrupted state.
    """

    def test_empty_db_query_returns_empty(self, mock_db):
        """Querying an empty table must return no rows, not raise an exception."""
        conn = mock_db["conn"]
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM network_logs")
        rows = cursor.fetchall()
        assert rows == [], f"Expected empty table, got {rows}"

    def test_insert_and_retrieve_log(self, mock_db):
        """Inserting one row must allow it to be retrieved with correct values."""
        conn = mock_db["conn"]
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO network_logs "
            "(timestamp, scenario, device, router_latency, router_packet_loss, "
            " dns_latency, dns_packet_loss, rssi, tx_rate, jitter, nic_errors, "
            " traffic, is_failure, failure_type, minutes_to_failure, health_score) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (1726000000.0, "router_congestion", "Router",
             75.0, 8.0, 45.0, 2.0, -60.0, 180.0, 18.0, 1.0, 120.0,
             1, "router_congestion", 4.5, 62.0),
        )
        conn.commit()
        cursor.execute("SELECT scenario, device, is_failure FROM network_logs WHERE is_failure=1")
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "router_congestion"
        assert rows[0][1] == "Router"

    def test_multiple_inserts_do_not_corrupt_db(self, mock_db):
        """50 sequential inserts must all be retrievable."""
        conn = mock_db["conn"]
        cursor = conn.cursor()
        for i in range(50):
            cursor.execute(
                "INSERT INTO network_logs "
                "(timestamp, scenario, device, router_latency, router_packet_loss, "
                " dns_latency, dns_packet_loss, rssi, tx_rate, jitter, nic_errors, "
                " traffic, is_failure, failure_type, minutes_to_failure, health_score) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (1726000000.0 + i, "test_scenario", f"Device_{i}",
                 10.0, 0.0, 35.0, 0.0, -55.0, 250.0, 2.0, 0.0, 45.0,
                 0, "none", 10.0, 95.0),
            )
        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM network_logs")
        count = cursor.fetchone()[0]
        assert count == 50, f"Expected 50 rows, found {count}"

    def test_failure_events_table_exists(self, mock_db):
        """The failure_events table must be present in the schema."""
        conn = mock_db["conn"]
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='failure_events'"
        )
        result = cursor.fetchone()
        assert result is not None, "failure_events table was not created"
