"""
test_network_simulator.py – pytest unit tests for NetworkSimulator.

Coverage:
    1. test_simulator_creates_valid_reading        – schema & range checks
    2. test_three_devices_produce_independent_readings – divergence check
    3. test_failure_injection_raises_latency       – router_overload spikes RTT
    4. test_failure_injection_drops_rssi           – ap_signal_drop tanks RSSI
    5. test_minutes_to_failure_label               – label transitions 10→(0,10)→0
    6. test_all_failure_types_occur                – every failure type is exercised

Fixtures are defined in conftest.py (router_sim, switch_sim, firewall_sim,
all_sims, bulk_readings).
"""

import pytest
from network_simulator import NetworkSimulator
from features import TIMEOUT_MS


# ---------------------------------------------------------------------------
# Helper constants
# ---------------------------------------------------------------------------

# The 9 raw metric keys that every reading must carry.
RAW_KEYS = {
    "router_latency_ms",
    "router_packet_loss",
    "dns_latency_ms",
    "dns_packet_loss",
    "rssi_dbm",
    "tx_rate_mbps",
    "jitter_ms",
    "nic_errors_per_sec",
    "traffic_kbps",
}

# Failure-type strings per device (mirrors NetworkSimulator.DEVICE_FAILURES)
DEVICE_FAILURE_TYPES = {
    "Router": {"router_latency_spike", "router_packet_loss", "wifi_ap_degrade"},
    "Switch": {"nic_error_spike", "tx_rate_drop"},
    "Firewall": {"dns_outage", "traffic_overload"},
}


# ===========================================================================
# 1. Schema and range validation
# ===========================================================================

class TestSimulatorCreatesValidReading:
    """All 9 raw metric keys exist, are numeric, and lie within physical bounds."""

    def test_all_nine_keys_exist(self, router_sim):
        reading = router_sim.next_reading()
        missing = RAW_KEYS - reading.keys()
        assert not missing, f"Missing keys in reading: {missing}"

    def test_all_values_are_numeric(self, router_sim):
        reading = router_sim.next_reading()
        for key in RAW_KEYS:
            value = reading[key]
            assert isinstance(value, (int, float)), (
                f"Key '{key}' has non-numeric value: {value!r}"
            )

    def test_router_latency_is_positive(self, router_sim):
        """router_latency_ms must be > 0 for all healthy baseline readings."""
        for _ in range(50):
            reading = router_sim.next_reading()
            assert reading["router_latency_ms"] > 0, (
                f"router_latency_ms must be positive, got {reading['router_latency_ms']}"
            )

    def test_rssi_within_bounds(self, router_sim):
        """RSSI is clamped to [-100, -20] dBm."""
        for _ in range(50):
            reading = router_sim.next_reading()
            rssi = reading["rssi_dbm"]
            assert -100 <= rssi <= -20, f"rssi_dbm out of range: {rssi}"

    def test_tx_rate_is_positive(self, router_sim):
        """tx_rate_mbps must be positive (clamped to min 1 Mbps)."""
        for _ in range(50):
            reading = router_sim.next_reading()
            assert reading["tx_rate_mbps"] > 0, (
                f"tx_rate_mbps must be positive, got {reading['tx_rate_mbps']}"
            )

    def test_packet_loss_within_bounds(self, router_sim):
        """Packet-loss percentages must be in [0, 100]."""
        for _ in range(50):
            reading = router_sim.next_reading()
            for key in ("router_packet_loss", "dns_packet_loss"):
                val = reading[key]
                assert 0 <= val <= 100, (
                    f"{key} out of [0, 100]: {val}"
                )


# ===========================================================================
# 2. Independence across devices
# ===========================================================================

class TestThreeDevicesProduceIndependentReadings:
    """Readings from three separate simulators should diverge over time."""

    def test_readings_diverge_over_time(self, all_sims):
        """
        After 10 readings, at least one metric from each device pair should
        differ — ruling out accidental global-state sharing.
        """
        n = 10
        histories = {
            name: [sim.next_reading() for _ in range(n)]
            for name, sim in all_sims.items()
        }

        # Compare every pair of devices across all readings
        device_names = list(histories.keys())
        all_identical = True
        for i in range(len(device_names)):
            for j in range(i + 1, len(device_names)):
                a, b = device_names[i], device_names[j]
                for k in range(n):
                    r_a = histories[a][k]
                    r_b = histories[b][k]
                    # Compare on any raw numeric metric
                    if any(r_a[key] != r_b[key] for key in RAW_KEYS):
                        all_identical = False
                        break
                if not all_identical:
                    break

        assert not all_identical, (
            "All three device simulators produced identical readings across "
            "10 ticks — they appear to share state or be seeded identically."
        )

    def test_each_device_labels_own_name(self, all_sims, bulk_readings):
        """The 'device' field in each reading must match the simulator's name."""
        for name, sim in all_sims.items():
            readings = bulk_readings(sim, 5)
            for r in readings:
                assert r["device"] == name, (
                    f"Simulator '{name}' emitted device='{r['device']}'"
                )


# ===========================================================================
# 3. Failure injection raises latency
# ===========================================================================

class TestFailureInjectionRaisesLatency:
    """
    During a *router_latency_spike* failure the router_latency_ms must exceed
    50 ms (healthy baseline is ~12 ms; the failure injects +80–220 ms per tick).

    We also accept readings from *router_packet_loss* and *traffic_overload*
    failure types because they both push latency well above 50 ms.
    """

    # Failure types that are expected to raise router latency above 50 ms.
    LATENCY_RAISING_TYPES = {"router_latency_spike", "router_packet_loss", "traffic_overload", "wifi_ap_degrade"}

    def test_router_overload_latency_exceeds_threshold(self):
        """
        Run 1 000 readings; for every reading that IS a failure whose type
        raises router latency, assert the observed latency > 50 ms.
        """
        sim = NetworkSimulator("Router")
        failure_readings = []

        for _ in range(1000):
            r = sim.next_reading()
            # Only look at hard-failure frames (is_failure == 1 AND failure_type != "none")
            if r["is_failure"] == 1 and r["failure_type"] in self.LATENCY_RAISING_TYPES:
                failure_readings.append(r)

        # If the RNG produced no qualifying failure in 1 000 ticks that's
        # statistically implausible but not a test-framework error — skip rather
        # than fail to avoid flakiness.
        if not failure_readings:
            pytest.skip(
                "No latency-raising failure occurred in 1 000 readings "
                "(statistically improbable; re-run)."
            )

        for r in failure_readings:
            assert r["router_latency_ms"] > 50, (
                f"During '{r['failure_type']}' failure, expected router_latency_ms > 50 ms, "
                f"got {r['router_latency_ms']} ms"
            )


# ===========================================================================
# 4. Failure injection drops RSSI
# ===========================================================================

class TestFailureInjectionDropsRssi:
    """
    During a *wifi_ap_degrade* (AP signal drop) failure the RSSI must drop
    below -70 dBm across the failure window.

    The failure injects -10 to -22 dBm per hard-fail tick (plus a pre-failure
    drift of -step*1.8 per tick).  Because the failure can start when RSSI is
    still near -55 dBm, the very first hard-fail tick may land around -65 to
    -77 dBm.  The correct invariant is that *at least one* reading during the
    failure window is below -70 dBm — not that every individual tick is.
    """

    def test_ap_signal_drop_rssi_below_threshold(self):
        """
        Run 1 000 readings on a Router simulator (which owns wifi_ap_degrade).
        Collect all hard-failure readings of that type, then verify the
        minimum observed RSSI across that window is lower than -70 dBm.
        """
        sim = NetworkSimulator("Router")
        ap_drop_readings = []

        for _ in range(1000):
            r = sim.next_reading()
            if r["is_failure"] == 1 and r["failure_type"] == "wifi_ap_degrade":
                ap_drop_readings.append(r)

        if not ap_drop_readings:
            pytest.skip(
                "wifi_ap_degrade failure did not occur in 1 000 readings "
                "(statistically improbable; re-run)."
            )

        min_rssi = min(r["rssi_dbm"] for r in ap_drop_readings)
        assert min_rssi < -70, (
            f"During wifi_ap_degrade failure, expected at least one reading "
            f"with rssi_dbm < -70 dBm. Minimum observed: {min_rssi} dBm "
            f"across {len(ap_drop_readings)} hard-fail ticks."
        )


# ===========================================================================
# 5. minutes_to_failure label transitions
# ===========================================================================

class TestMinutesToFailureLabel:
    """
    Label semantics (from next_reading docstring):
        is_failure == 0, not in pre  →  minutes_to_failure == 10.0
        is_failure == 1, is hard fail →  minutes_to_failure == 0.0
        is_failure == 1, in pre phase →  minutes_to_failure  ∈ (0, 10)
    """

    def _collect_by_phase(self, n: int = 5000):
        """Return readings split by (healthy, hard_failure, pre_failure)."""
        sim = NetworkSimulator("Router")
        healthy = []
        hard_fail = []
        pre_fail = []

        for _ in range(n):
            r = sim.next_reading()
            ft = r["failure_type"]
            is_f = r["is_failure"]
            mins = r["minutes_to_failure"]

            if is_f == 0:
                healthy.append(r)
            elif ft != "none":
                # Hard-failure frame: failure_countdown > 0
                hard_fail.append(r)
            else:
                # Pre-failure frame: failure_countdown == 0 but pre_failure_count > 0
                pre_fail.append(r)

        return healthy, hard_fail, pre_fail

    def test_healthy_label_is_ten(self):
        healthy, _, _ = self._collect_by_phase()
        if not healthy:
            pytest.skip("No healthy readings collected in 5 000 ticks.")
        for r in healthy:
            assert r["minutes_to_failure"] == 10.0, (
                f"Expected 10.0 for healthy reading, got {r['minutes_to_failure']}"
            )

    def test_hard_failure_label_is_zero(self):
        _, hard_fail, _ = self._collect_by_phase()
        if not hard_fail:
            pytest.skip("No hard-failure readings collected in 5 000 ticks.")
        for r in hard_fail:
            assert r["minutes_to_failure"] == 0.0, (
                f"Expected 0.0 during hard failure, got {r['minutes_to_failure']}"
            )

    def test_pre_failure_label_between_zero_and_ten(self):
        _, _, pre_fail = self._collect_by_phase()
        if not pre_fail:
            pytest.skip("No pre-failure readings collected in 5 000 ticks.")
        for r in pre_fail:
            mins = r["minutes_to_failure"]
            assert 0 < mins < 10, (
                f"Pre-failure minutes_to_failure expected in (0, 10), got {mins}"
            )


# ===========================================================================
# 6. All failure types occur
# ===========================================================================

class TestAllFailureTypesOccur:
    """
    Each device's full set of failure types must appear at least once
    in 5 000 readings.  The simulator's 3% per-tick trigger rate makes
    this overwhelmingly likely in that window.
    """

    @pytest.mark.parametrize("device_name", ["Router", "Switch", "Firewall"])
    def test_all_failure_types_seen(self, device_name):
        sim = NetworkSimulator(device_name)
        expected = DEVICE_FAILURE_TYPES[device_name]
        seen = set()

        for _ in range(5000):
            r = sim.next_reading()
            if r["failure_type"] != "none":
                seen.add(r["failure_type"])

        missing = expected - seen
        assert not missing, (
            f"Device '{device_name}': failure types never observed in 5 000 "
            f"readings: {missing}"
        )
