"""
test_health_scorer.py – pytest unit tests for health_scorer.calculate_health_score().

Tests:
    1. test_perfect_network_scores_near_100   — ideal metrics → score >= 85
    2. test_failed_network_scores_below_30    — worst-case metrics → score <= 30
    3. test_score_always_between_0_and_100    — 100 random combos never escape [0, 100]
"""

import random
import pytest
from health_scorer import calculate_health_score
from features import TIMEOUT_MS


# ---------------------------------------------------------------------------
# Helper: build a reading dict from keyword args
# ---------------------------------------------------------------------------

def _reading(
    router_latency=12.0,
    dns_latency=35.0,
    rssi=-55.0,
    tx_rate=240.0,
    jitter=2.0,
    nic_errors=0.0,
    traffic=45.0,
    router_packet_loss=0.0,
    dns_packet_loss=0.0,
):
    return {
        "router_latency_ms": router_latency,
        "router_packet_loss": router_packet_loss,
        "dns_latency_ms": dns_latency,
        "dns_packet_loss": dns_packet_loss,
        "rssi_dbm": rssi,
        "tx_rate_mbps": tx_rate,
        "jitter_ms": jitter,
        "nic_errors_per_sec": nic_errors,
        "traffic_kbps": traffic,
    }


# ---------------------------------------------------------------------------
# 1. Perfect network scores near 100
# ---------------------------------------------------------------------------

class TestPerfectNetworkScoresNear100:
    """
    Metrics that are all well inside healthy bands should produce a high score.
    The spec threshold is >= 85.
    """

    def test_score_at_least_85(self):
        reading = _reading(
            router_latency=8.0,
            dns_latency=30.0,
            rssi=-50.0,
            tx_rate=300.0,
            jitter=1.0,
            nic_errors=0.0,
            traffic=20.0,
        )
        score = calculate_health_score(reading, failure_prob=0.02)
        assert score >= 85, (
            f"Expected health score >= 85 for a perfect network reading, got {score}"
        )

    def test_score_does_not_exceed_100(self):
        reading = _reading(
            router_latency=8.0,
            dns_latency=30.0,
            rssi=-50.0,
            tx_rate=300.0,
            jitter=1.0,
            nic_errors=0.0,
            traffic=20.0,
        )
        score = calculate_health_score(reading, failure_prob=0.02)
        assert score <= 100.0, (
            f"Health score must not exceed 100, got {score}"
        )


# ---------------------------------------------------------------------------
# 2. Failed network scores below 30
# ---------------------------------------------------------------------------

class TestFailedNetworkScoresBelow30:
    """
    Worst-case metric values (gateway timeout, DNS timeout, extreme RSSI,
    near-zero TX, high jitter, high NIC errors, traffic overload, and a
    near-certain failure probability) must push the score to <= 30.
    """

    def test_score_at_most_30(self):
        reading = _reading(
            router_latency=TIMEOUT_MS,   # 9999 — gateway unreachable
            dns_latency=TIMEOUT_MS,      # 9999 — DNS/WAN unreachable
            rssi=-95.0,                  # extremely weak AP signal
            tx_rate=5.0,                 # link auto-downgraded to bare minimum
            jitter=100.0,                # very unstable RTT
            nic_errors=20.0,             # high switch-port error rate
            traffic=500.0,               # heavy congestion
            router_packet_loss=100.0,    # all pings to gateway lost
            dns_packet_loss=100.0,       # all DNS queries lost
        )
        score = calculate_health_score(reading, failure_prob=0.97)
        assert score <= 30, (
            f"Expected health score <= 30 for a fully failed network, got {score}"
        )

    def test_score_is_not_negative(self):
        """calculate_health_score must clamp to 0.0, never go below."""
        reading = _reading(
            router_latency=TIMEOUT_MS,
            dns_latency=TIMEOUT_MS,
            rssi=-95.0,
            tx_rate=5.0,
            jitter=100.0,
            nic_errors=20.0,
            traffic=500.0,
            router_packet_loss=100.0,
            dns_packet_loss=100.0,
        )
        score = calculate_health_score(reading, failure_prob=0.97)
        assert score >= 0.0, (
            f"Health score must not go below 0, got {score}"
        )


# ---------------------------------------------------------------------------
# 3. Score always between 0 and 100 across random inputs
# ---------------------------------------------------------------------------

class TestScoreAlwaysBetween0And100:
    """
    Stress-test calculate_health_score() with 100 random metric combinations.
    The function documents a [0, 100] clamp — verify it holds for every input.
    """

    # Fixed seed for reproducibility while still covering wide input space
    _RNG = random.Random(42)

    def _random_reading(self):
        rng = self._RNG
        return _reading(
            router_latency=rng.uniform(0.0, TIMEOUT_MS + 500),
            dns_latency=rng.uniform(0.0, TIMEOUT_MS + 500),
            rssi=rng.uniform(-100.0, -20.0),
            tx_rate=rng.uniform(0.0, 600.0),
            jitter=rng.uniform(0.0, 500.0),
            nic_errors=rng.uniform(0.0, 100.0),
            traffic=rng.uniform(0.0, 2000.0),
            router_packet_loss=rng.uniform(0.0, 100.0),
            dns_packet_loss=rng.uniform(0.0, 100.0),
        )

    def test_100_random_combinations_stay_in_range(self):
        violations = []
        for i in range(100):
            reading = self._random_reading()
            failure_prob = self._RNG.uniform(0.0, 1.0)
            score = calculate_health_score(reading, failure_prob)
            if not (0.0 <= score <= 100.0):
                violations.append((i, score, reading, failure_prob))

        assert not violations, (
            f"{len(violations)} out-of-range scores found (showing first):\n"
            f"  iteration={violations[0][0]}, score={violations[0][1]}, "
            f"failure_prob={violations[0][3]:.3f}"
        )
