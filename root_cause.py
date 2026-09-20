"""
Map observed network symptoms to a failing device class.

A router is not failing because the laptop CPU is high.
It is failing when the path to the gateway degrades.
"""

from features import TIMEOUT_MS


def diagnose(reading):
    r_lat = reading.get("router_latency_ms", 0) or 0
    r_loss = reading.get("router_packet_loss", 0) or 0
    d_lat = reading.get("dns_latency_ms", 0) or 0
    d_loss = reading.get("dns_packet_loss", 0) or 0
    rssi = reading.get("rssi_dbm", -55) or -55
    tx = reading.get("tx_rate_mbps", 200) or 200
    jitter = reading.get("jitter_ms", 0) or 0
    errs = reading.get("nic_errors_per_sec", 0) or 0
    traffic = reading.get("traffic_kbps", 0) or 0
    wifi = reading.get("_wifi_available", True)

    reasons = []
    suspected = []

    if r_lat >= TIMEOUT_MS:
        reasons.append("Default gateway unreachable - router is down or not forwarding ICMP")
        suspected.append("Router")
    elif r_lat > 100:
        reasons.append(f"Gateway RTT {r_lat:.0f} ms (failing >100 ms) - router overloaded or congested")
        suspected.append("Router")
    elif r_lat > 40:
        reasons.append(f"Elevated gateway RTT {r_lat:.0f} ms — early router stress")
        suspected.append("Router")

    if r_loss > 30:
        reasons.append(f"High loss to router ({r_loss:.0f}%) — device dropping packets")
        suspected.append("Router")
    elif r_loss > 5:
        reasons.append(f"Router packet loss {r_loss:.0f}% — unstable gateway path")
        suspected.append("Router")

    if jitter > 50:
        reasons.append(f"Jitter {jitter:.0f} ms - unstable forwarding (router/queue)")
        suspected.append("Router")
    elif jitter > 20:
        reasons.append(f"Jitter {jitter:.0f} ms - congestion on the local hop")
        suspected.append("Router")

    if wifi and rssi < -80:
        reasons.append(f"RSSI {rssi:.0f} dBm - WiFi AP failing or radio too weak")
        suspected.append("WiFi AP")
    elif wifi and rssi < -70:
        reasons.append(f"RSSI {rssi:.0f} dBm - AP coverage degrading")
        suspected.append("WiFi AP")

    if wifi and tx < 50:
        reasons.append(f"TX rate {tx:.0f} Mbps - AP/switch auto-downgraded the link")
        suspected.append("WiFi AP / Switch")

    if errs > 15:
        reasons.append(f"NIC errors {errs:.1f}/s - failing switch port or cable/PHY")
        suspected.append("Switch")
    elif errs > 5:
        reasons.append(f"NIC errors {errs:.1f}/s - switch-port / physical-layer issue")
        suspected.append("Switch")

    if d_lat >= TIMEOUT_MS:
        reasons.append("DNS/WAN unreachable - firewall, ISP, or upstream outage")
        suspected.append("Firewall / ISP")
    elif d_lat > 200:
        reasons.append(f"DNS RTT {d_lat:.0f} ms - firewall/ISP path degrading")
        suspected.append("Firewall / ISP")

    if d_loss > 10:
        reasons.append(f"DNS packet loss {d_loss:.0f}% - WAN/firewall dropping queries")
        suspected.append("Firewall / ISP")

    if traffic > 500:
        reasons.append(f"Traffic {traffic:.0f} KB/s - overload stressing the edge device")
        suspected.append("Firewall")
    elif traffic > 200:
        reasons.append(f"Heavy load {traffic:.0f} KB/s - device queues may grow")

    local_ok = r_lat < 50 and r_loss < 5
    wan_bad = d_lat > 200 or d_loss > 10
    if local_ok and wan_bad:
        reasons.append("Local router looks healthy - problem is WAN / firewall / ISP")
    elif (r_lat > 100 or r_loss > 30) and d_lat < 150:
        reasons.append("Internet path looks OK - the local router is the failing device")

    if not reasons:
        reasons.append("Metric pattern unusual, but no single device threshold crossed")

    n = len([r for r in reasons if "looks" not in r.lower()])
    if (r_lat >= TIMEOUT_MS or r_loss > 30 or d_lat >= TIMEOUT_MS or errs > 15):
        severity = "CRITICAL"
        lead_time = "Imminent - device may already be failing"
    elif n >= 2 or r_lat > 100 or rssi < -80:
        severity = "MEDIUM"
        lead_time = "Near-term - minutes if trend continues"
    else:
        severity = "LOW"
        lead_time = "Early warning - watch the trend"

    # Unique suspected devices, most-mentioned first
    seen = []
    for d in suspected:
        if d not in seen:
            seen.append(d)
    failing_device = seen[0] if seen else reading.get("device", "Unknown")

    return {
        "severity": severity,
        "reasons": reasons,
        "lead_time": lead_time,
        "failing_device": failing_device,
        "suspected_devices": seen,
    }
