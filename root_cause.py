"""
root_cause.py — Rule-based root cause analysis and explainability engine.

Translates real or simulated network metrics into human-readable diagnostic reasons,
identifies the most likely failed component, and assigns severity and lead time.
"""

from config import (
    ROUTER_LATENCY_WARNING,
    ROUTER_LATENCY_CRITICAL,
    ROUTER_LATENCY_DEAD,
    ROUTER_LOSS_WARNING,
    ROUTER_LOSS_CRITICAL,
    DNS_LATENCY_WARNING,
    DNS_LATENCY_CRITICAL,
    DNS_LATENCY_DEAD,
    RSSI_WARNING,
    RSSI_CRITICAL,
    TX_RATE_WARNING,
    TX_RATE_CRITICAL,
    JITTER_WARNING,
    JITTER_CRITICAL,
    NIC_ERRORS_WARNING,
    NIC_ERRORS_CRITICAL,
    TRAFFIC_WARNING,
    TRAFFIC_CRITICAL,
)


def _fmt(val) -> str:
    """Format numeric values cleanly without trailing decimal zeros when whole."""
    if val is None:
        return "0"
    try:
        f = float(val)
        return str(int(f)) if f.is_integer() else f"{f:.1f}"
    except Exception:
        return str(val)


def diagnose(reading: dict) -> dict:
    """
    Diagnose network telemetry reading against operational thresholds.

    Parameters
    ----------
    reading : dict
        Network metrics dictionary containing RAW_FEATURES.

    Returns
    -------
    dict
        Contains:
          - severity: "LOW", "MEDIUM", or "CRITICAL"
          - reasons: list of human-readable diagnostic strings
          - lead_time: estimated failure lead time description
    """
    # 1. Extract metric values with safe defaults and legacy fallbacks
    router_latency = reading.get("router_latency")
    if router_latency is None:
        router_latency = reading.get("router_latency_ms", 10.0)
    router_latency = float(router_latency) if router_latency is not None else 10.0

    router_packet_loss = reading.get("router_packet_loss")
    if router_packet_loss is None:
        router_packet_loss = 0.0
    router_packet_loss = float(router_packet_loss)

    dns_latency = reading.get("dns_latency")
    if dns_latency is None:
        dns_latency = reading.get("dns_latency_ms", 45.0)
    dns_latency = float(dns_latency) if dns_latency is not None else 45.0

    dns_packet_loss = reading.get("dns_packet_loss")
    if dns_packet_loss is None:
        dns_packet_loss = 0.0
    dns_packet_loss = float(dns_packet_loss)

    rssi = reading.get("rssi")
    if rssi is None:
        rssi = reading.get("rssi_dbm")
    if rssi is not None:
        rssi = float(rssi)

    tx_rate = reading.get("tx_rate")
    if tx_rate is None:
        tx_rate = reading.get("tx_rate_mbps")
    if tx_rate is not None:
        tx_rate = float(tx_rate)

    jitter = reading.get("jitter")
    if jitter is None:
        jitter = reading.get("jitter_ms", 0.0)
    jitter = float(jitter) if jitter is not None else 0.0

    nic_errors = reading.get("nic_errors")
    if nic_errors is None:
        nic_errors = reading.get("nic_errors_per_sec", 0.0)
    nic_errors = float(nic_errors) if nic_errors is not None else 0.0

    traffic = reading.get("traffic")
    if traffic is None:
        traffic = reading.get("traffic_kbps", 0.0)
    traffic = float(traffic) if traffic is not None else 0.0

    # Determine if host is explicitly on Ethernet (skip RF-specific alerts)
    conn_type = reading.get("_conn_type", "")
    is_ethernet = (conn_type == "Ethernet") or (rssi == 0.0 and tx_rate == 0.0)

    reasons = []

    # 2. Threshold-based diagnostics
    router_unreachable = False
    router_congested = False
    if router_latency >= ROUTER_LATENCY_DEAD or router_packet_loss >= 80:
        reasons.append("ROUTER UNREACHABLE — router has crashed or lost power. Entire network is down.")
        router_unreachable = True
    elif router_latency > ROUTER_LATENCY_CRITICAL:
        reasons.append(
            f"Router severely congested — latency {_fmt(router_latency)}ms "
            f"(critical threshold: {ROUTER_LATENCY_CRITICAL}ms). "
            f"Router is overloaded and dropping connections."
        )
        router_congested = True
    elif router_latency > ROUTER_LATENCY_WARNING:
        reasons.append(
            f"Router under load — latency {_fmt(router_latency)}ms "
            f"(warning threshold: {ROUTER_LATENCY_WARNING}ms)."
        )
        router_congested = True

    if router_packet_loss > ROUTER_LOSS_CRITICAL and not router_unreachable:
        reasons.append(
            f"Router dropping {_fmt(router_packet_loss)}% of packets — "
            f"data loss occurring on local network path."
        )

    dns_issue = False
    if dns_latency >= DNS_LATENCY_DEAD or dns_packet_loss >= 90:
        dns_issue = True
        if router_latency < ROUTER_LATENCY_WARNING:
            reasons.append(
                "INTERNET DOWN — local router is healthy but DNS server is unreachable. "
                "ISP outage or upstream failure."
            )
        else:
            reasons.append("Complete network failure — both router and internet unreachable.")
    elif dns_latency > DNS_LATENCY_WARNING:
        dns_issue = True
        reasons.append(
            f"DNS/ISP slowdown — {_fmt(dns_latency)}ms response (normal <80ms). "
            f"Internet degraded but reachable."
        )

    wifi_issue = False
    if not is_ethernet and rssi is not None:
        if rssi < RSSI_CRITICAL:
            wifi_issue = True
            reasons.append(
                f"WiFi signal critically weak — {_fmt(rssi)}dBm "
                f"(critical: {RSSI_CRITICAL}dBm). Wireless link near failure."
            )
        elif rssi < RSSI_WARNING:
            wifi_issue = True
            reasons.append(
                f"WiFi signal degrading — {_fmt(rssi)}dBm. "
                f"Access point may be overloaded or failing."
            )

    if not is_ethernet and tx_rate is not None and tx_rate < TX_RATE_CRITICAL:
        wifi_issue = True
        reasons.append(
            f"WiFi link speed critically low — {_fmt(tx_rate)}Mbps "
            f"(critical: {TX_RATE_CRITICAL}Mbps). AP has severely reduced link quality."
        )

    nic_high = False
    if nic_errors > NIC_ERRORS_CRITICAL:
        nic_high = True
        reasons.append(
            f"Critical physical layer errors — {_fmt(nic_errors)}/sec. "
            f"Cable damage or switch port failure. Check cable and port connections."
        )
    elif nic_errors > NIC_ERRORS_WARNING:
        nic_high = True
        reasons.append(
            f"NIC errors elevated — {_fmt(nic_errors)}/sec. "
            f"Possible cable or switch port degradation."
        )

    jitter_elevated = False
    if jitter > JITTER_CRITICAL:
        jitter_elevated = True
        reasons.append(
            f"Severe network jitter — {_fmt(jitter)}ms std-dev. "
            f"Network path is unstable. Video calls and real-time apps will fail."
        )
    elif jitter > JITTER_WARNING:
        jitter_elevated = True
        reasons.append(f"Network jitter elevated — {_fmt(jitter)}ms. Connection is unstable.")

    traffic_elevated = False
    if traffic > TRAFFIC_CRITICAL:
        traffic_elevated = True
        reasons.append(
            f"Traffic overload — {_fmt(traffic)}KB/s. "
            f"Network bandwidth saturated, stressing router and switch hardware."
        )

    # 3. Determine the MOST LIKELY FAILED COMPONENT
    most_likely = None
    if router_unreachable:
        most_likely = "Most likely cause: Router failure or power loss"
    elif dns_issue and router_latency < ROUTER_LATENCY_CRITICAL:
        most_likely = "Most likely cause: ISP outage or DNS server failure"
    elif wifi_issue:
        most_likely = "Most likely cause: WiFi access point degradation"
    elif nic_high:
        most_likely = "Most likely cause: Physical layer fault (cable/switch port)"
    elif router_congested or jitter_elevated or traffic_elevated or router_packet_loss > ROUTER_LOSS_WARNING:
        most_likely = "Most likely cause: Network congestion and overload"

    if most_likely:
        reasons.append(most_likely)

    # Fallback if no specific thresholds were breached
    if not reasons:
        reasons.append(
            "Anomaly detected by AI model — metrics are within normal thresholds individually "
            "but the pattern combination indicates risk"
        )

    # 4. Severity evaluation
    is_critical = (
        router_latency >= ROUTER_LATENCY_DEAD
        or router_packet_loss >= 80.0
        or dns_latency >= DNS_LATENCY_DEAD
        or dns_packet_loss >= 90.0
        or (not is_ethernet and rssi is not None and rssi < RSSI_CRITICAL)
        or (not is_ethernet and tx_rate is not None and tx_rate < TX_RATE_CRITICAL)
        or nic_errors > NIC_ERRORS_CRITICAL
        or jitter > JITTER_CRITICAL
    )

    is_medium = (
        router_latency > ROUTER_LATENCY_WARNING
        or router_packet_loss > ROUTER_LOSS_WARNING
        or dns_latency > DNS_LATENCY_WARNING
        or dns_packet_loss > 10.0
        or (not is_ethernet and rssi is not None and rssi < RSSI_WARNING)
        or (not is_ethernet and tx_rate is not None and tx_rate < TX_RATE_WARNING)
        or nic_errors > NIC_ERRORS_WARNING
        or jitter > JITTER_WARNING
        or traffic > TRAFFIC_WARNING
    )

    if is_critical:
        severity = "CRITICAL"
        lead_time = "Imminent — failure occurring now or in seconds"
    elif is_medium:
        severity = "MEDIUM"
        lead_time = "Near-term — failure within 30 seconds to 2 minutes"
    else:
        severity = "LOW"
        lead_time = "Early warning — degradation detected, monitor closely"

    return {
        "severity": severity,
        "reasons": reasons,
        "lead_time": lead_time,
    }
