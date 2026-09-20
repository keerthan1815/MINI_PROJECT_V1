"""
network_simulator.py — Simulated network health indicators as seen from a monitoring agent.

Simulates what an observer on a network connection detects as failures approach:
  - 9 raw metrics (from config.RAW_FEATURES)
  - 8 failure scenarios (from config.FAILURE_SCENARIOS)
  - Gradual pre-failure degradation state machine
  - Hard failure countdown and recovery
  - Gaussian/uniform observational noise and strict physical bounds
"""

import random
import time

from config import (
    RAW_FEATURES,
    FAILURE_SCENARIOS,
    NUM_READINGS_PER_DEVICE,
    SIMULATION_SPEED,
)


class NetworkSimulator:
    """Simulates network connection health indicators observed from a client/agent."""

    # Baseline healthy metric values
    BASELINE_ROUTER_LATENCY = 10.0
    BASELINE_ROUTER_PACKET_LOSS = 0.0
    BASELINE_DNS_LATENCY = 45.0
    BASELINE_DNS_PACKET_LOSS = 0.0
    BASELINE_RSSI = -58.0
    BASELINE_TX_RATE = 200.0
    BASELINE_JITTER = 2.0
    BASELINE_NIC_ERRORS = 0.0
    BASELINE_TRAFFIC = 50.0

    def __init__(self, scenario_name="Network", **kwargs):
        """
        Initialize the network health simulator.

        Parameters
        ----------
        scenario_name : str, optional
            Identifier for this simulation stream or target failure scenario,
            by default "Network".
        """
        # Accept either scenario_name or legacy device_name keyword
        if "device_name" in kwargs:
            scenario_name = kwargs["device_name"]
        self.scenario_name = str(scenario_name)
        self.device_name = self.scenario_name  # Backward-compatibility alias

        # Metric state variables
        self.router_latency = self.BASELINE_ROUTER_LATENCY
        self.router_packet_loss = self.BASELINE_ROUTER_PACKET_LOSS
        self.dns_latency = self.BASELINE_DNS_LATENCY
        self.dns_packet_loss = self.BASELINE_DNS_PACKET_LOSS
        self.rssi = self.BASELINE_RSSI
        self.tx_rate = self.BASELINE_TX_RATE
        self.jitter = self.BASELINE_JITTER
        self.nic_errors = self.BASELINE_NIC_ERRORS
        self.traffic = self.BASELINE_TRAFFIC

        # Failure state machine
        self.pre_failure_count = 0
        self.total_pre_failure = 0
        self.failure_countdown = 0
        self.active_failure_type = "none"

    def _reset_baseline(self):
        """Reset internal metrics back to clean baseline state upon recovery."""
        self.router_latency = self.BASELINE_ROUTER_LATENCY
        self.router_packet_loss = self.BASELINE_ROUTER_PACKET_LOSS
        self.dns_latency = self.BASELINE_DNS_LATENCY
        self.dns_packet_loss = self.BASELINE_DNS_PACKET_LOSS
        self.rssi = self.BASELINE_RSSI
        self.tx_rate = self.BASELINE_TX_RATE
        self.jitter = self.BASELINE_JITTER
        self.nic_errors = self.BASELINE_NIC_ERRORS
        self.traffic = self.BASELINE_TRAFFIC

    def _start_failure(self):
        """Initiate a failure sequence with pre-failure warning phase."""
        if self.scenario_name in FAILURE_SCENARIOS:
            self.active_failure_type = self.scenario_name
        else:
            self.active_failure_type = random.choice(list(FAILURE_SCENARIOS.keys()))

        # 10 to 20 readings of gradual degradation before full failure
        self.total_pre_failure = random.randint(10, 20)
        self.pre_failure_count = self.total_pre_failure
        # 8 to 18 readings of full hard failure
        self.failure_countdown = random.randint(8, 18)

    def inject_failure(self, scenario_name=None):
        """Manually trigger a failure scenario."""
        if scenario_name and scenario_name in FAILURE_SCENARIOS:
            self.active_failure_type = scenario_name
        else:
            self.active_failure_type = random.choice(list(FAILURE_SCENARIOS.keys()))
        self.total_pre_failure = random.randint(10, 20)
        self.pre_failure_count = self.total_pre_failure
        self.failure_countdown = random.randint(8, 18)

    def _apply_scenario_effect(self, scenario: str, pf_factor: float = None):
        """
        Apply scenario symptom deltas to the network metrics.

        If pf_factor is provided (0.0 < pf_factor <= 1.0), symptoms are scaled
        to represent pre-failure degradation.
        """
        is_pre = pf_factor is not None
        factor = pf_factor if is_pre else 1.0

        if scenario == "router_congestion":
            self.router_latency += random.uniform(40.0, 120.0) * factor
            self.jitter += random.uniform(15.0, 50.0) * factor
            self.router_packet_loss += random.uniform(2.0, 15.0) * factor

        elif scenario == "router_unreachable":
            if is_pre:
                self.router_latency += random.uniform(50.0, 150.0) * factor
                self.router_packet_loss += random.uniform(10.0, 30.0) * factor
                self.dns_latency += random.uniform(50.0, 150.0) * factor
                self.dns_packet_loss += random.uniform(10.0, 30.0) * factor
            else:
                self.router_latency = 9999.0
                self.router_packet_loss = 100.0
                self.dns_latency = 9999.0
                self.dns_packet_loss = 100.0

        elif scenario == "internet_outage":
            if is_pre:
                self.dns_latency += random.uniform(100.0, 300.0) * factor
                self.dns_packet_loss += random.uniform(10.0, 30.0) * factor
                # router_latency stays normal
            else:
                self.dns_latency = 9999.0
                self.dns_packet_loss = 100.0
                # router_latency stays normal

        elif scenario == "wifi_degradation":
            self.rssi -= random.uniform(8.0, 25.0) * factor
            self.tx_rate -= random.uniform(30.0, 100.0) * factor
            self.jitter += random.uniform(10.0, 30.0) * factor
            self.router_packet_loss += random.uniform(3.0, 20.0) * factor

        elif scenario == "physical_layer_fault":
            self.nic_errors += random.uniform(8.0, 25.0) * factor
            self.router_packet_loss += random.uniform(15.0, 40.0) * factor
            self.router_latency += random.uniform(20.0, 80.0) * factor

        elif scenario == "network_overload":
            self.traffic += random.uniform(200.0, 500.0) * factor
            self.router_latency += random.uniform(30.0, 80.0) * factor
            self.dns_latency += random.uniform(50.0, 150.0) * factor
            self.jitter += random.uniform(20.0, 60.0) * factor

        elif scenario == "dns_slowdown":
            self.dns_latency += random.uniform(150.0, 500.0) * factor
            self.dns_packet_loss += random.uniform(10.0, 40.0) * factor
            # router_latency stays normal

        elif scenario == "packet_storm":
            self.router_packet_loss += random.uniform(30.0, 70.0) * factor
            self.dns_packet_loss += random.uniform(30.0, 70.0) * factor
            self.nic_errors += random.uniform(5.0, 20.0) * factor

    def next_reading(self) -> dict:
        """
        Produce the next simulated network telemetry sample.

        Returns
        -------
        dict
            Telemetry reading containing:
              - timestamp (epoch float)
              - scenario (scenario_name string)
              - 9 raw metric values from RAW_FEATURES
              - is_failure (0 or 1)
              - failure_type (scenario string or "none")
              - minutes_to_failure (float: 0.0 when failing, 10.0 when normal,
                proportional during pre_failure)
        """
        # Start new failure with 2.5% probability when no failure is active
        if self.pre_failure_count == 0 and self.failure_countdown == 0:
            if random.random() < 0.025:
                self._start_failure()

        in_pre = self.pre_failure_count > 0
        is_fail = (not in_pre) and (self.failure_countdown > 0)

        # 1. State machine progression and failure effects
        if in_pre:
            # pf_factor decreases from 1.0 to 0 as pre_failure progresses
            pf_factor = self.pre_failure_count / float(self.total_pre_failure)
            self._apply_scenario_effect(self.active_failure_type, pf_factor=pf_factor)

            # Proportional time to failure (strictly in (0, 10))
            minutes = round(
                max(0.2, min(9.8, 10.0 * (self.pre_failure_count / (self.total_pre_failure + 1)))),
                2,
            )
            is_failure_label = 1
            reported_failure_type = "none"

            self.pre_failure_count -= 1

        elif is_fail:
            self._apply_scenario_effect(self.active_failure_type, pf_factor=None)

            minutes = 0.0
            is_failure_label = 1
            reported_failure_type = self.active_failure_type

            self.failure_countdown -= 1
            if self.failure_countdown == 0:
                self.active_failure_type = "none"
                self._reset_baseline()

        else:
            # Normal healthy operation: gentle mean reversion keeps metrics centered around baseline
            self.router_latency += 0.05 * (self.BASELINE_ROUTER_LATENCY - self.router_latency)
            self.dns_latency += 0.05 * (self.BASELINE_DNS_LATENCY - self.dns_latency)
            self.rssi += 0.05 * (self.BASELINE_RSSI - self.rssi)
            self.tx_rate += 0.05 * (self.BASELINE_TX_RATE - self.tx_rate)
            self.jitter += 0.05 * (self.BASELINE_JITTER - self.jitter)
            self.traffic += 0.05 * (self.BASELINE_TRAFFIC - self.traffic)

            minutes = 10.0
            is_failure_label = 0
            reported_failure_type = "none"

        # 2. Add small normal noise to all metrics per reading
        self.router_latency += random.uniform(-1.5, 1.5)
        self.dns_latency += random.uniform(-2.0, 2.0)
        self.rssi += random.uniform(-0.5, 0.5)
        self.tx_rate += random.uniform(-5.0, 5.0)
        self.jitter += random.uniform(-0.5, 0.5)
        self.traffic += random.uniform(-5.0, 5.0)
        self.nic_errors = max(0.0, self.nic_errors - 0.5)
        self.router_packet_loss = max(0.0, self.router_packet_loss - 1.0)
        self.dns_packet_loss = max(0.0, self.dns_packet_loss - 1.0)

        # 3. Enforce realistic physical bounds
        self.router_latency = min(9999.0, max(1.0, self.router_latency))
        self.dns_latency = min(9999.0, max(1.0, self.dns_latency))
        self.router_packet_loss = min(100.0, max(0.0, self.router_packet_loss))
        self.dns_packet_loss = min(100.0, max(0.0, self.dns_packet_loss))
        self.rssi = min(-20.0, max(-100.0, self.rssi))
        self.tx_rate = min(600.0, max(1.0, self.tx_rate))
        self.jitter = min(200.0, max(0.5, self.jitter))
        self.nic_errors = max(0.0, self.nic_errors)
        self.traffic = max(0.0, self.traffic)

        # 4. Assemble reading dictionary
        reading = {
            "timestamp": time.time(),
            "scenario": self.scenario_name,
            # Backward-compatibility alias
            "device": self.device_name,
            # Exactly the 9 RAW_FEATURES from config.py
            "router_latency": round(self.router_latency, 2),
            "router_packet_loss": round(self.router_packet_loss, 2),
            "dns_latency": round(self.dns_latency, 2),
            "dns_packet_loss": round(self.dns_packet_loss, 2),
            "rssi": round(self.rssi, 2),
            "tx_rate": round(self.tx_rate, 2),
            "jitter": round(self.jitter, 2),
            "nic_errors": round(self.nic_errors, 2),
            "traffic": round(self.traffic, 2),
            # Legacy metric aliases for existing components
            "router_latency_ms": round(self.router_latency, 2),
            "dns_latency_ms": round(self.dns_latency, 2),
            "rssi_dbm": round(self.rssi, 2),
            "tx_rate_mbps": round(self.tx_rate, 2),
            "jitter_ms": round(self.jitter, 2),
            "nic_errors_per_sec": round(self.nic_errors, 2),
            "traffic_kbps": round(self.traffic, 2),
            # Target labels and metadata
            "is_failure": is_failure_label,
            "failure_type": reported_failure_type,
            "minutes_to_failure": minutes,
        }

        return reading
