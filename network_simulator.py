"""
Simulate NETWORK SYMPTOMS of device failure — not PC CPU/RAM.

Router tab   → gateway latency, router packet loss, jitter, WiFi AP fade
Switch tab   → NIC/switch-port errors, TX-rate collapse
Firewall tab → DNS/WAN latency, DNS loss, traffic overload
"""

import random
import time
from collections import deque

from features import TIMEOUT_MS


class NetworkSimulator:
    """One simulated network device, observed from a client host."""

    DEVICE_FAILURES = {
        "Router": [
            "router_latency_spike",
            "router_packet_loss",
            "wifi_ap_degrade",
        ],
        "Switch": [
            "nic_error_spike",
            "tx_rate_drop",
        ],
        "Firewall": [
            "dns_outage",
            "traffic_overload",
        ],
    }

    def __init__(self, device_name="Router"):
        self.device_name = device_name
        self.router_latency_ms = 12.0
        self.dns_latency_ms = 40.0
        self.rssi_dbm = -55.0
        self.tx_rate_mbps = 240.0
        self.nic_errors_per_sec = 0.0
        self.traffic_kbps = 45.0
        self._router_ok = deque([True] * 8, maxlen=8)
        self._dns_ok = deque([True] * 8, maxlen=8)
        self._rtt_samples = deque([12.0] * 5, maxlen=5)
        self.failure_countdown = 0
        self.pre_failure_count = 0
        self.failure_type = None

    def _start_failure(self):
        options = self.DEVICE_FAILURES.get(
            self.device_name, ["router_latency_spike"])
        self.failure_type = random.choice(options)
        self.failure_countdown = random.randint(10, 18)
        self.pre_failure_count = random.randint(12, 22)

    def _loss_pct(self, window):
        if not window:
            return 0.0
        lost = sum(1 for ok in window if not ok)
        return round(100.0 * lost / len(window), 1)

    def _jitter(self):
        vals = [v for v in self._rtt_samples if v < 9000]
        if len(vals) < 2:
            return 0.0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return round(var ** 0.5, 2)

    def next_reading(self):
        if self.failure_countdown == 0 and self.pre_failure_count == 0:
            if random.random() < 0.03:
                self._start_failure()

        in_pre = self.pre_failure_count > 0
        is_fail = self.failure_countdown > 0

        self.router_latency_ms += random.uniform(-0.8, 0.8)
        self.dns_latency_ms += random.uniform(-1.5, 1.5)
        self.rssi_dbm += random.uniform(-0.4, 0.4)
        self.tx_rate_mbps += random.uniform(-4, 4)
        self.traffic_kbps += random.uniform(-6, 6)
        self.nic_errors_per_sec = max(
            0.0, self.nic_errors_per_sec * 0.6 + random.uniform(0, 0.4))

        router_timeout = False
        dns_timeout = False

        if in_pre:
            ft = self.failure_type
            step = self.pre_failure_count / 22.0
            if ft == "router_latency_spike":
                self.router_latency_ms += step * 8
            elif ft == "router_packet_loss":
                self.router_latency_ms += step * 4
                if random.random() < 0.15:
                    router_timeout = True
            elif ft == "wifi_ap_degrade":
                self.rssi_dbm -= step * 1.8
                self.tx_rate_mbps -= step * 8
            elif ft == "nic_error_spike":
                self.nic_errors_per_sec += step * 2
            elif ft == "tx_rate_drop":
                self.tx_rate_mbps -= step * 10
            elif ft == "dns_outage":
                self.dns_latency_ms += step * 18
            elif ft == "traffic_overload":
                self.traffic_kbps += step * 20
            self.pre_failure_count -= 1

        if is_fail:
            ft = self.failure_type
            if ft == "router_latency_spike":
                self.router_latency_ms += random.uniform(80, 220)
                if random.random() < 0.25:
                    router_timeout = True
            elif ft == "router_packet_loss":
                self.router_latency_ms += random.uniform(40, 120)
                router_timeout = random.random() < 0.55
            elif ft == "wifi_ap_degrade":
                self.rssi_dbm -= random.uniform(10, 22)
                self.tx_rate_mbps -= random.uniform(40, 120)
                self.router_latency_ms += random.uniform(15, 50)
                if random.random() < 0.2:
                    router_timeout = True
            elif ft == "nic_error_spike":
                self.nic_errors_per_sec += random.uniform(8, 22)
                self.tx_rate_mbps -= random.uniform(20, 80)
                if random.random() < 0.2:
                    router_timeout = True
            elif ft == "tx_rate_drop":
                self.tx_rate_mbps -= random.uniform(80, 180)
                self.nic_errors_per_sec += random.uniform(2, 8)
            elif ft == "dns_outage":
                if random.random() < 0.45:
                    dns_timeout = True
                else:
                    self.dns_latency_ms = random.uniform(400, 2500)
            elif ft == "traffic_overload":
                self.traffic_kbps += random.uniform(250, 550)
                self.dns_latency_ms += random.uniform(40, 180)
                self.router_latency_ms += random.uniform(10, 40)
            self.failure_countdown -= 1
            if self.failure_countdown == 0:
                self.failure_type = None

        self.router_latency_ms = min(max(self.router_latency_ms, 1.0), TIMEOUT_MS)
        self.dns_latency_ms = min(max(self.dns_latency_ms, 1.0), TIMEOUT_MS)
        self.rssi_dbm = min(max(self.rssi_dbm, -100.0), -20.0)
        self.tx_rate_mbps = min(max(self.tx_rate_mbps, 1.0), 600.0)
        self.traffic_kbps = max(self.traffic_kbps, 0.0)
        self.nic_errors_per_sec = min(max(self.nic_errors_per_sec, 0.0), 80.0)

        shown_router = TIMEOUT_MS if router_timeout else self.router_latency_ms
        shown_dns = TIMEOUT_MS if dns_timeout else self.dns_latency_ms

        self._router_ok.append(not router_timeout)
        self._dns_ok.append(not dns_timeout)
        if shown_router < 9000:
            self._rtt_samples.append(shown_router)

        if is_fail:
            minutes = 0.0
        elif in_pre:
            minutes = round(max(self.pre_failure_count / 6.0, 0.2), 2)
        else:
            minutes = 10.0

        return {
            "timestamp": time.time(),
            "device": self.device_name,
            "router_latency_ms": round(shown_router, 2),
            "router_packet_loss": self._loss_pct(self._router_ok),
            "dns_latency_ms": round(shown_dns, 2),
            "dns_packet_loss": self._loss_pct(self._dns_ok),
            "rssi_dbm": round(self.rssi_dbm, 1),
            "tx_rate_mbps": round(self.tx_rate_mbps, 1),
            "jitter_ms": self._jitter(),
            "nic_errors_per_sec": round(self.nic_errors_per_sec, 2),
            "traffic_kbps": round(self.traffic_kbps, 2),
            "is_failure": int(is_fail or in_pre),
            "failure_type": self.failure_type if is_fail else "none",
            "minutes_to_failure": minutes,
            "_conn_type": "WiFi",
            "_router_ip": "sim-gateway",
            "_dns_ip": "sim-dns",
        }
