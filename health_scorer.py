"""
Health score from NETWORK DEVICE symptoms only.
"""

from features import TIMEOUT_MS


def _router_icmp_likely_blocked(r_lat, r_loss, d_lat, d_loss, rssi, tx, errs):
    """
    Detect the pattern where a router blocks ICMP pings but the network is
    actually healthy.  Signature: router timeout/100% loss, but DNS responds
    normally, WiFi signal is fine, and NIC has no errors.
    """
    router_dead = r_lat >= TIMEOUT_MS or r_loss >= 90
    dns_ok = d_lat < 200 and d_loss < 10
    link_ok = rssi > -75 and tx > 30
    nic_ok = errs < 3
    return router_dead and dns_ok and link_ok and nic_ok


def calculate_health_score(reading, failure_prob):
    score = 100.0
    r_lat = reading.get("router_latency_ms", 0)
    r_loss = reading.get("router_packet_loss", 0)
    d_lat = reading.get("dns_latency_ms", 0)
    d_loss = reading.get("dns_packet_loss", 0)
    rssi = reading.get("rssi_dbm", -55)
    tx = reading.get("tx_rate_mbps", 200)
    jitter = reading.get("jitter_ms", 0)
    errs = reading.get("nic_errors_per_sec", 0)
    traffic = reading.get("traffic_kbps", 0)

    icmp_blocked = _router_icmp_likely_blocked(
        r_lat, r_loss, d_lat, d_loss, rssi, tx, errs
    )

    # --- Router latency penalty ---
    if r_lat >= TIMEOUT_MS:
        # If ICMP is just blocked, apply a small penalty instead of -28
        score -= 5 if icmp_blocked else 28
    elif r_lat > 20:
        score -= min((r_lat - 20) * 0.25, 20)

    # --- Router packet loss penalty ---
    if r_loss > 5:
        # If ICMP is just blocked, apply a small penalty instead of full
        score -= 3 if icmp_blocked else min(r_loss * 0.45, 25)

    if d_lat >= TIMEOUT_MS:
        score -= 22
    elif d_lat > 80:
        score -= min((d_lat - 80) * 0.08, 18)

    if d_loss > 10:
        score -= min(d_loss * 0.3, 18)

    if rssi < -70:
        score -= min((-rssi - 70) * 1.4, 20)

    if tx < 50:
        score -= min((50 - tx) * 0.2, 12)

    if jitter > 20:
        score -= min((jitter - 20) * 0.25, 12)

    score -= min(errs * 2.5, 18)

    if traffic > 200:
        score -= min((traffic - 200) * 0.03, 10)

    # Also reduce the ML failure-prob penalty when ICMP is blocked
    score -= failure_prob * (10 if icmp_blocked else 30)
    return max(0.0, min(100.0, round(score, 1)))


def score_to_label(score):
    if score >= 90:
        return "Excellent", "🟢"
    if score >= 70:
        return "Good", "🟢"
    if score >= 50:
        return "Fair", "🟡"
    if score >= 30:
        return "Poor", "🟠"
    return "Critical", "🔴"


def rssi_quality(rssi_dbm):
    if rssi_dbm >= -50:
        return "Excellent signal"
    if rssi_dbm >= -60:
        return "Good signal"
    if rssi_dbm >= -70:
        return "Fair signal"
    if rssi_dbm >= -80:
        return "Weak — AP struggling or far"
    return "Very weak — AP / RF failure likely"
