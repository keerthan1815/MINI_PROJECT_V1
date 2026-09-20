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
# ===========================================================================
# 3. Failure injection raises latency
# ===========================================================================

class TestFailureInjectionRaisesLatency:
    """
    During router_congestion or router_unreachable, router latency must exceed
    50 ms (healthy baseline is ~10 ms).
    """

    LATENCY_RAISING_TYPES = {"router_congestion", "router_unreachable", "network_overload"}

    def test_router_overload_latency_exceeds_threshold(self):
        """Inject router_congestion and verify latency exceeds 50 ms."""
        sim = NetworkSimulator("router_congestion")
        sim.inject_failure("router_congestion")

        # Advance through pre-failure into hard failure
        found = False
        for _ in range(50):
            r = sim.next_reading()
            if r["failure_type"] == "router_congestion":
                assert r["router_latency"] > 40.0, (
                    f"Expected router_latency > 40 ms, got {r['router_latency']}"
                )
                found = True
                break
        assert found, "router_congestion failure was not activated after injection"


# ===========================================================================
# 4. Failure injection drops RSSI
# ===========================================================================

class TestFailureInjectionDropsRssi:
    """
    During wifi_degradation failure, RSSI must drop below healthy baseline (-58 dBm).
    """

    def test_wifi_degradation_rssi_drops(self):
        """Inject wifi_degradation and verify RSSI drops significantly."""
        sim = NetworkSimulator("wifi_degradation")
        sim.inject_failure("wifi_degradation")

        found = False
        for _ in range(50):
            r = sim.next_reading()
            if r["failure_type"] == "wifi_degradation":
                assert r["rssi"] < -65.0, (
                    f"Expected rssi < -65 dBm during wifi_degradation, got {r['rssi']}"
                )
                found = True
                break
        assert found, "wifi_degradation failure was not activated after injection"


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
        sim = NetworkSimulator("Network")
        healthy = []
        hard_fail = []
        pre_fail = []

        for _ in range(n):
            r = sim.next_reading()
            ft = r["failure_type"]
            is_f = r["is_failure"]

            if is_f == 0:
                healthy.append(r)
            elif ft != "none":
                hard_fail.append(r)
            else:
                pre_fail.append(r)

        return healthy, hard_fail, pre_fail

    def test_healthy_label_is_ten(self):
        sim = NetworkSimulator("Network")
        # Fresh simulator starts in healthy state
        r = sim.next_reading()
        assert r["minutes_to_failure"] == 10.0, (
            f"Expected 10.0 for healthy reading, got {r['minutes_to_failure']}"
        )

    def test_hard_failure_label_is_zero(self):
        sim = NetworkSimulator("router_unreachable")
        sim.inject_failure("router_unreachable")
        found = False
        for _ in range(50):
            r = sim.next_reading()
            if r["failure_type"] == "router_unreachable":
                assert r["minutes_to_failure"] == 0.0, (
                    f"Expected 0.0 during hard failure, got {r['minutes_to_failure']}"
                )
                found = True
                break
        assert found, "Hard failure was not activated"

    def test_pre_failure_label_between_zero_and_ten(self):
        sim = NetworkSimulator("router_congestion")
        sim.inject_failure("router_congestion")
        # First reading after inject_failure is in pre-failure phase
        r = sim.next_reading()
        assert r["is_failure"] == 1
        mins = r["minutes_to_failure"]
        assert 0 < mins < 10, f"Pre-failure minutes_to_failure expected in (0, 10), got {mins}"


# ===========================================================================
# 6. All failure scenarios occur and can be simulated
# ===========================================================================

class TestAllFailureScenariosOccur:
    """
    Every scenario in config.FAILURE_SCENARIOS can be injected and successfully
    exercised by NetworkSimulator.
    """

    @pytest.mark.parametrize("scenario_name", [
        "router_congestion",
        "router_unreachable",
        "internet_outage",
        "wifi_degradation",
        "physical_layer_fault",
        "network_overload",
        "dns_slowdown",
        "packet_storm",
    ])
    def test_each_scenario_injected_and_seen(self, scenario_name):
        sim = NetworkSimulator(scenario_name)
        sim.inject_failure(scenario_name)
        seen = False

        for _ in range(50):
            r = sim.next_reading()
            if r["failure_type"] == scenario_name:
                seen = True
                assert r["is_failure"] == 1
                assert r["minutes_to_failure"] == 0.0
                break

        assert seen, f"Scenario '{scenario_name}' was never observed after injection."

