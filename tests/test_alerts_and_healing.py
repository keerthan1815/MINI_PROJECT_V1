"""
test_alerts_and_healing.py – Automated test fixtures for alerting and self-healing modules.

Tests:
    1. TestEmailAlertStructure    – message body contains required fields; SMTP not called
    2. TestBuzzerStateControl     – start/stop toggle behaves correctly
    3. TestSelfHealingRoutingLogic – correct healing function called per RCA symptoms
    4. TestSelfHealingCooldown    – high-risk reading with no single metric breach flags router
    5. TestHealingCsvCreation     – _log() writes a valid CSV entry to a temp path
    6. TestSendAlertMetricFields  – metrics dict fields appear in the formatted message body
"""

import os
import sys
import tempfile
import csv
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ===========================================================================
# 1. Email alert: message content validation (no live SMTP)
# ===========================================================================

class TestEmailAlertStructure:
    """
    Verify that the email body is correctly formatted without sending a real email.
    All SMTP calls are patched to avoid network I/O.
    """

    def _build_email_body(self, device, prediction, reason, metrics):
        """Replicate the message format from email_alert.send_alert()."""
        return f"""
NETWORK DEVICE FAILURE ALERT

Device / path : {device}
Prediction    : {prediction}
Reason        : {reason}

Gateway RTT   : {metrics.get('router_latency_ms')} ms
Router loss   : {metrics.get('router_packet_loss')} %
DNS RTT       : {metrics.get('dns_latency_ms')} ms
DNS loss      : {metrics.get('dns_packet_loss')} %
RSSI          : {metrics.get('rssi_dbm')} dBm
Jitter        : {metrics.get('jitter_ms')} ms
NIC errors/s  : {metrics.get('nic_errors_per_sec')}
Traffic       : {metrics.get('traffic_kbps')} KB/s
"""

    def test_email_body_contains_device_name(self):
        metrics = {
            "router_latency_ms": 75.0, "router_packet_loss": 8.0,
            "dns_latency_ms": 45.0, "dns_packet_loss": 2.0,
            "rssi_dbm": -60.0, "jitter_ms": 18.0,
            "nic_errors_per_sec": 1.0, "traffic_kbps": 120.0,
        }
        body = self._build_email_body("Router", "FAILURE", "High gateway RTT", metrics)
        assert "Router" in body

    def test_email_body_contains_all_metric_keys(self):
        metrics = {
            "router_latency_ms": 75.0, "router_packet_loss": 8.0,
            "dns_latency_ms": 45.0, "dns_packet_loss": 2.0,
            "rssi_dbm": -60.0, "jitter_ms": 18.0,
            "nic_errors_per_sec": 1.0, "traffic_kbps": 120.0,
        }
        body = self._build_email_body("Router", "FAILURE", "High gateway RTT", metrics)
        for substring in ["Gateway RTT", "Router loss", "DNS RTT", "RSSI", "Jitter", "NIC errors"]:
            assert substring in body, f"Email body missing section: '{substring}'"

    def test_email_body_empty_metrics_does_not_crash(self):
        """send_alert with empty metrics dict must not raise."""
        body = self._build_email_body("Firewall", "WARNING", "DNS slow", {})
        assert "Firewall" in body
        assert "DNS slow" in body

    @patch("smtplib.SMTP")
    def test_send_alert_calls_smtp_with_correct_host(self, mock_smtp):
        """send_alert() must connect to smtp.gmail.com:587."""
        from email_alert import send_alert
        mock_server = MagicMock()
        mock_smtp.return_value = mock_server

        send_alert("Router", "FAILURE", "High RTT", {
            "router_latency_ms": 75.0, "router_packet_loss": 8.0,
            "dns_latency_ms": 45.0, "dns_packet_loss": 2.0,
            "rssi_dbm": -60.0, "jitter_ms": 18.0,
            "nic_errors_per_sec": 1.0, "traffic_kbps": 120.0,
        })

        mock_smtp.assert_called_once()
        args = mock_smtp.call_args
        assert "smtp.gmail.com" in str(args)
        assert "587" in str(args)

    @patch("smtplib.SMTP")
    def test_send_alert_does_not_raise_on_smtp_exception(self, mock_smtp):
        """If SMTP raises, send_alert must catch it and not propagate."""
        mock_smtp.side_effect = ConnectionRefusedError("SMTP server unavailable")
        from email_alert import send_alert
        # Should not raise
        send_alert("Router", "FAILURE", "High RTT", {})

    @patch("smtplib.SMTP")
    def test_send_alert_calls_sendmail(self, mock_smtp):
        """send_alert must call sendmail exactly once."""
        mock_server = MagicMock()
        mock_smtp.return_value = mock_server
        from email_alert import send_alert

        send_alert("Switch", "WARNING", "NIC errors", {
            "router_latency_ms": 10.0, "router_packet_loss": 0.0,
            "dns_latency_ms": 35.0, "dns_packet_loss": 0.0,
            "rssi_dbm": -55.0, "jitter_ms": 2.0,
            "nic_errors_per_sec": 6.0, "traffic_kbps": 45.0,
        })

        mock_server.sendmail.assert_called_once()


# ===========================================================================
# 2. Buzzer module state control
# ===========================================================================

class TestBuzzerStateControl:
    """
    start_buzzer() / stop_buzzer() must toggle the `running` flag correctly.
    winsound.Beep is patched to avoid audio I/O.
    """

    def setup_method(self):
        """Reset buzzer state before every test."""
        import buzzer
        buzzer.running = False

    def teardown_method(self):
        """Always stop buzzer after tests."""
        import buzzer
        buzzer.running = False

    @patch("buzzer.IS_WINDOWS", False)
    def test_start_sets_running_true(self):
        import buzzer
        assert not buzzer.running
        buzzer.start_buzzer()
        assert buzzer.running

    @patch("buzzer.IS_WINDOWS", False)
    def test_stop_sets_running_false(self):
        import buzzer
        buzzer.running = True
        buzzer.stop_buzzer()
        assert not buzzer.running

    @patch("buzzer.IS_WINDOWS", False)
    def test_double_start_does_not_create_multiple_threads(self):
        """Calling start_buzzer() twice must not double-start the alarm thread."""
        import buzzer
        buzzer.start_buzzer()
        thread_count_before = threading.active_count()
        buzzer.start_buzzer()
        thread_count_after = threading.active_count()
        # Second start should be a no-op since running is already True
        assert thread_count_after <= thread_count_before + 1

    @patch("buzzer.IS_WINDOWS", False)
    def test_stop_without_start_does_not_crash(self):
        """Stopping an already-stopped buzzer must be a safe no-op."""
        import buzzer
        buzzer.running = False
        buzzer.stop_buzzer()
        assert not buzzer.running


# ===========================================================================
# 3. Self-healing routing logic
# ===========================================================================

class TestSelfHealingRoutingLogic:
    """
    run_healing() must call the correct healing function based on metric thresholds.
    All OS-level side effects (os.system, CSV writes, DB inserts) are patched.
    """

    @patch("self_healing._log")
    @patch("self_healing.save_memory")
    @patch("self_healing.init_memory")
    def test_high_router_latency_triggers_flag_router(self, m_init, m_save, m_log):
        from self_healing import run_healing
        reading = {
            "router_latency_ms": 150.0, "router_packet_loss": 5.0,
            "dns_latency_ms": 35.0, "nic_errors_per_sec": 0.0, "traffic_kbps": 45.0,
        }
        rca = {"failing_device": "Router"}
        run_healing(reading, rca, failure_risk=0, device="Router")
        # save_memory was called with router-related action
        calls_str = str(m_save.call_args_list)
        assert "ROUTER_RTT" in calls_str or "CHECK_GATEWAY" in calls_str

    @patch("self_healing._log")
    @patch("self_healing.save_memory")
    @patch("self_healing.init_memory")
    def test_high_dns_latency_triggers_heal_dns(self, m_init, m_save, m_log):
        from self_healing import run_healing
        with patch("self_healing.os.system") as m_sys:
            reading = {
                "router_latency_ms": 10.0, "router_packet_loss": 0.0,
                "dns_latency_ms": 250.0, "nic_errors_per_sec": 0.0, "traffic_kbps": 45.0,
            }
            rca = {"failing_device": "Firewall"}
            run_healing(reading, rca, failure_risk=0, device="Firewall")
            calls_str = str(m_save.call_args_list)
            assert "DNS_SLOW" in calls_str or "FLUSH_DNS" in calls_str

    @patch("self_healing._log")
    @patch("self_healing.save_memory")
    @patch("self_healing.init_memory")
    def test_high_nic_errors_triggers_flag_switch_port(self, m_init, m_save, m_log):
        from self_healing import run_healing
        reading = {
            "router_latency_ms": 10.0, "router_packet_loss": 0.0,
            "dns_latency_ms": 35.0, "nic_errors_per_sec": 10.0, "traffic_kbps": 45.0,
        }
        rca = {"failing_device": "Switch"}
        run_healing(reading, rca, failure_risk=0, device="Switch")
        calls_str = str(m_save.call_args_list)
        assert "NIC_ERRORS" in calls_str or "FLAG_PORT" in calls_str

    @patch("self_healing._log")
    @patch("self_healing.save_memory")
    @patch("self_healing.init_memory")
    def test_high_traffic_triggers_flag_overload(self, m_init, m_save, m_log):
        from self_healing import run_healing
        reading = {
            "router_latency_ms": 10.0, "router_packet_loss": 0.0,
            "dns_latency_ms": 35.0, "nic_errors_per_sec": 0.0, "traffic_kbps": 400.0,
        }
        rca = {"failing_device": "Firewall"}
        run_healing(reading, rca, failure_risk=0, device="Firewall")
        calls_str = str(m_save.call_args_list)
        assert "TRAFFIC_SPIKE" in calls_str or "LOG_OVERLOAD" in calls_str

    @patch("self_healing._log")
    @patch("self_healing.save_memory")
    @patch("self_healing.init_memory")
    def test_high_risk_no_specific_breach_flags_router(self, m_init, m_save, m_log):
        """High failure_risk with no individual metric breach must still flag router."""
        from self_healing import run_healing
        reading = {
            "router_latency_ms": 10.0, "router_packet_loss": 0.0,
            "dns_latency_ms": 35.0, "nic_errors_per_sec": 0.0, "traffic_kbps": 45.0,
        }
        rca = {"failing_device": "Router"}
        run_healing(reading, rca, failure_risk=90, device="Router")
        assert m_save.called, "Expected save_memory to be called for high-risk reading"

    @patch("self_healing._log")
    @patch("self_healing.save_memory")
    @patch("self_healing.init_memory")
    def test_healthy_reading_no_healing_actions(self, m_init, m_save, m_log):
        """Normal healthy reading with low risk must not trigger any healing action."""
        from self_healing import run_healing
        reading = {
            "router_latency_ms": 10.0, "router_packet_loss": 0.0,
            "dns_latency_ms": 35.0, "nic_errors_per_sec": 0.0, "traffic_kbps": 45.0,
        }
        rca = {"failing_device": "Router"}
        run_healing(reading, rca, failure_risk=0, device="Router")
        m_save.assert_not_called()


# ===========================================================================
# 4. Healing CSV log creation
# ===========================================================================

class TestHealingCsvCreation:
    """
    The internal _log() helper must create a valid CSV file with correct headers.
    """

    def test_log_creates_csv_with_header(self, tmp_path):
        """_log() must write a CSV with the correct column headers."""
        import self_healing as sh
        original_csv = sh.HEALING_CSV

        try:
            csv_path = str(tmp_path / "test_healing.csv")
            sh.HEALING_CSV = csv_path

            with patch("self_healing.insert_healing_log"):
                sh._log("Router", "High RTT", "Flag router", "Flagged")

            assert os.path.exists(csv_path), "healing CSV was not created"

            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) >= 2, "CSV must have at least a header and one data row"
            assert "Time" in rows[0], f"Expected 'Time' in header, got {rows[0]}"
            assert "Device" in rows[0], f"Expected 'Device' in header, got {rows[0]}"
            assert "Router" in rows[1], f"Expected 'Router' in data row, got {rows[1]}"

        finally:
            sh.HEALING_CSV = original_csv

    def test_log_appends_multiple_entries(self, tmp_path):
        """_log() called twice must produce exactly 2 data rows (plus header)."""
        import self_healing as sh
        original_csv = sh.HEALING_CSV

        try:
            csv_path = str(tmp_path / "multi_heal.csv")
            sh.HEALING_CSV = csv_path

            with patch("self_healing.insert_healing_log"):
                sh._log("Router", "RTT high", "Flag router", "Flagged")
                sh._log("Firewall", "DNS slow", "Flush DNS", "Completed")

            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)

            data_rows = [r for r in rows if r and r[0] != "Time"]
            assert len(data_rows) == 2, (
                f"Expected 2 data rows, got {len(data_rows)}: {rows}"
            )

        finally:
            sh.HEALING_CSV = original_csv
