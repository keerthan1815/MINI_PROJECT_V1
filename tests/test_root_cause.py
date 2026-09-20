"""
test_root_cause.py – pytest unit tests for root_cause.diagnose().

Tests:
    1. test_router_overload_detected         — RTT=150 → Router in reasons, MEDIUM/CRITICAL
    2. test_router_crash_is_critical         — RTT=9999, loss=100 → CRITICAL + UNREACHABLE
    3. test_dns_outage_identified            — DNS timeout + local-OK → DNS/ISP, CRITICAL
    4. test_ap_signal_drop_identified        — RSSI=-90, TX=15 → WiFi/AP, MEDIUM/CRITICAL
    5. test_normal_reading_has_no_critical   — all metrics healthy → severity LOW
    6. test_lead_time_matches_severity       — lead_time strings per severity level
"""

import pytest
from root_cause import diagnose
from features import TIMEOUT_MS


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normal_reading(**overrides):
    """
    Returns a reading dict with all metrics inside the healthy baseline range.
    Any keyword arg overrides a specific key.
    """
    base = {
        "router_latency_ms": 12.0,
        "router_packet_loss": 0.0,
        "dns_latency_ms": 35.0,
        "dns_packet_loss": 0.0,
        "rssi_dbm": -55.0,
        "tx_rate_mbps": 240.0,
        "jitter_ms": 2.0,
        "nic_errors_per_sec": 0.0,
        "traffic_kbps": 45.0,
        "_wifi_available": True,
        "device": "Router",
    }
    base.update(overrides)
    return base


def _reasons_text(result):
    """Return all reason strings joined into one lower-case blob for easy searching."""
    return " ".join(result["reasons"]).lower()


# ---------------------------------------------------------------------------
# 1. Router overload detected
# ---------------------------------------------------------------------------

class TestRouterOverloadDetected:
    """
    A gateway RTT of 150 ms is well above the 100 ms 'failing' threshold.
    The diagnose() function must mention 'Router' and rate this MEDIUM/CRITICAL.
    """

    def test_router_appears_in_reasons(self):
        reading = _normal_reading(router_latency_ms=150.0)
        result = diagnose(reading)
        reasons_text = _reasons_text(result)
        assert "router" in reasons_text, (
            f"'Router' not found in reasons for RTT=150 ms. "
            f"Got reasons: {result['reasons']}"
        )

    def test_severity_is_medium_or_critical(self):
        reading = _normal_reading(router_latency_ms=150.0)
        result = diagnose(reading)
        assert result["severity"] in ("MEDIUM", "CRITICAL"), (
            f"Expected MEDIUM or CRITICAL for RTT=150 ms, got {result['severity']!r}"
        )


# ---------------------------------------------------------------------------
# 2. Router crash is CRITICAL
# ---------------------------------------------------------------------------

class TestRouterCrashIsCritical:
    """
    router_latency=9999 equals TIMEOUT_MS (gateway unreachable);
    router_packet_loss=100 adds high loss.  Both push severity to CRITICAL
    and the 'unreachable' keyword must appear in at least one reason.
    """

    @pytest.fixture()
    def crash_result(self):
        reading = _normal_reading(
            router_latency_ms=TIMEOUT_MS,
            router_packet_loss=100.0,
        )
        return diagnose(reading)

    def test_severity_is_critical(self, crash_result):
        assert crash_result["severity"] == "CRITICAL", (
            f"Expected CRITICAL for gateway timeout+100% loss, "
            f"got {crash_result['severity']!r}"
        )

    def test_unreachable_in_reasons(self, crash_result):
        reasons_text = _reasons_text(crash_result)
        assert "unreachable" in reasons_text, (
            f"Expected 'UNREACHABLE' (case-insensitive) in reasons. "
            f"Got reasons: {crash_result['reasons']}"
        )


# ---------------------------------------------------------------------------
# 3. DNS outage identified
# ---------------------------------------------------------------------------

class TestDnsOutageIdentified:
    """
    When the local router is fine but DNS/WAN is timed out and has 100% loss,
    the root cause should name DNS/ISP and be rated CRITICAL.
    """

    @pytest.fixture()
    def dns_result(self):
        reading = _normal_reading(
            router_latency_ms=10.0,      # local router is healthy
            router_packet_loss=0.0,
            dns_latency_ms=TIMEOUT_MS,   # DNS/WAN unreachable
            dns_packet_loss=100.0,
        )
        return diagnose(reading)

    def test_dns_or_isp_in_reasons(self, dns_result):
        reasons_text = _reasons_text(dns_result)
        assert "dns" in reasons_text or "isp" in reasons_text, (
            f"Expected 'DNS' or 'ISP' in reasons for DNS outage. "
            f"Got reasons: {dns_result['reasons']}"
        )

    def test_severity_is_critical(self, dns_result):
        assert dns_result["severity"] == "CRITICAL", (
            f"Expected CRITICAL for DNS timeout+100% loss, "
            f"got {dns_result['severity']!r}"
        )


# ---------------------------------------------------------------------------
# 4. AP signal drop identified
# ---------------------------------------------------------------------------

class TestApSignalDropIdentified:
    """
    RSSI = -90 dBm is below the -80 threshold → 'WiFi AP failing'.
    TX rate = 15 Mbps is below the 50 Mbps threshold → AP downgraded the link.
    Severity must be MEDIUM or CRITICAL, and reasons must mention WiFi/AP.
    """

    @pytest.fixture()
    def ap_result(self):
        reading = _normal_reading(
            rssi_dbm=-90.0,
            tx_rate_mbps=15.0,
            _wifi_available=True,
        )
        return diagnose(reading)

    def test_wifi_or_ap_in_reasons(self, ap_result):
        reasons_text = _reasons_text(ap_result)
        assert "wifi" in reasons_text or "ap" in reasons_text, (
            f"Expected 'WiFi' or 'AP' in reasons for RSSI=-90 dBm. "
            f"Got reasons: {ap_result['reasons']}"
        )

    def test_severity_is_medium_or_critical(self, ap_result):
        assert ap_result["severity"] in ("MEDIUM", "CRITICAL"), (
            f"Expected MEDIUM or CRITICAL for RSSI=-90 dBm, "
            f"got {ap_result['severity']!r}"
        )


# ---------------------------------------------------------------------------
# 5. Normal reading has no critical flags
# ---------------------------------------------------------------------------

class TestNormalReadingHasNoCriticalFlags:
    """
    A reading with all metrics squarely in the healthy range should produce
    severity == 'LOW' — no thresholds crossed.
    """

    def test_severity_is_low(self):
        reading = _normal_reading()
        result = diagnose(reading)
        assert result["severity"] == "LOW", (
            f"Expected LOW severity for fully healthy reading, "
            f"got {result['severity']!r}. Reasons: {result['reasons']}"
        )


# ---------------------------------------------------------------------------
# 6. Lead-time string matches severity
# ---------------------------------------------------------------------------

class TestLeadTimeMatchesSeverity:
    """
    Each severity level maps to a specific lead_time prefix.
    We test with readings crafted to produce each severity tier.
    """

    def test_critical_maps_to_imminent(self):
        reading = _normal_reading(router_latency_ms=TIMEOUT_MS)
        result = diagnose(reading)
        assert result["severity"] == "CRITICAL"
        assert result["lead_time"].lower().startswith("imminent"), (
            f"Expected lead_time starting 'Imminent' for CRITICAL, "
            f"got {result['lead_time']!r}"
        )

    def test_medium_maps_to_near_term(self):
        # RTT=150 ms → MEDIUM (>100 ms but < TIMEOUT_MS, no other CRITICAL trigger)
        reading = _normal_reading(router_latency_ms=150.0)
        result = diagnose(reading)
        assert result["severity"] == "MEDIUM"
        assert result["lead_time"].lower().startswith("near-term"), (
            f"Expected lead_time starting 'Near-term' for MEDIUM, "
            f"got {result['lead_time']!r}"
        )

    def test_low_maps_to_early_warning(self):
        reading = _normal_reading()
        result = diagnose(reading)
        assert result["severity"] == "LOW"
        assert result["lead_time"].lower().startswith("early warning"), (
            f"Expected lead_time starting 'Early warning' for LOW, "
            f"got {result['lead_time']!r}"
        )
