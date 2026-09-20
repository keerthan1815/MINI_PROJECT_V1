"""
real_network_collector.py — Real-time telemetry collector for the connected network.

Measures all 9 raw network metrics from the actual host interface and connection:
  - Zero hardcoded IP addresses or default telemetry values
  - Auto-detection of local IP, default gateway, DNS servers, connection type
  - Ping burst measurements for latency, packet loss, and jitter
  - Wi-Fi radio stats (SSID, RSSI dBm, Transmit rate Mbps, Channel, Radio type)
  - psutil NIC I/O counters for traffic (KB/s) and packet error/drop rates
  - Rolling window statistics and short-term trends for model inference
"""

from collections import deque
import platform
import re
import socket
import subprocess
import time

import numpy as np
import psutil

from config import (
    ALL_FEATURES,
    PING_COUNT,
    RAW_FEATURES,
    ROLLING_FEATURES,
    ROLLING_WINDOW,
    TREND_FEATURES,
)

# ---------------------------------------------------------------------------
# Global State & Rolling Histories
# ---------------------------------------------------------------------------
_config_cache = None

_prev_nic_errors = None
_prev_nic_time = None

_prev_traffic_bytes = None
_prev_traffic_time = None

_router_lat_history = deque(maxlen=ROLLING_WINDOW)
_dns_lat_history = deque(maxlen=ROLLING_WINDOW)
_rssi_history = deque(maxlen=ROLLING_WINDOW)


# ===========================================================================
# SECTION 3 — WIFI STATS (Windows netsh) - defined early for auto-detection
# ===========================================================================

def read_wifi_stats() -> dict:
    """
    Parse active Wi-Fi radio stats on Windows via `netsh wlan show interfaces`.

    Returns
    -------
    dict or None
      Keys: ssid, rssi (dBm), tx_rate (Mbps), channel, radio
      Returns None if not connected to Wi-Fi or command unavailable.
      Never raises exceptions.
    """
    if platform.system().lower() != "windows":
        return None

    try:
        proc = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True,
            text=True,
            timeout=4.0,
        )
        out = proc.stdout
        if "There is no wireless interface" in out or "SSID" not in out:
            return None

        result = {}

        # 1. SSID
        m_ssid = re.search(r"^\s*SSID\s*:\s*(.+)$", out, re.MULTILINE)
        result["ssid"] = m_ssid.group(1).strip() if m_ssid else "Unknown"

        # 2. RSSI (native dBm or converted from signal percentage)
        m_rssi = re.search(r"^\s*Rssi\s*:\s*(-\d+)", out, re.MULTILINE)
        if not m_rssi:
            m_rssi = re.search(r"Signal\s*:\s*(-\d+)\s*dBm", out, re.IGNORECASE)

        if m_rssi:
            result["rssi"] = float(m_rssi.group(1))
        else:
            m_pct = re.search(r"Signal\s*:\s*(\d+)\s*%", out, re.IGNORECASE)
            if m_pct:
                pct = float(m_pct.group(1))
                result["rssi"] = round(-100.0 + (pct * 0.6), 1)
            else:
                result["rssi"] = None

        # 3. Transmit rate (fallback to receive rate)
        m_tx = re.search(r"Transmit rate(?:\s*\(Mbps\))?\s*:\s*([\d.]+)", out, re.IGNORECASE)
        if not m_tx:
            m_tx = re.search(r"Receive rate(?:\s*\(Mbps\))?\s*:\s*([\d.]+)", out, re.IGNORECASE)
        result["tx_rate"] = float(m_tx.group(1)) if m_tx else None

        # 4. Channel
        m_ch = re.search(r"Channel\s*:\s*(\d+)", out, re.IGNORECASE)
        result["channel"] = int(m_ch.group(1)) if m_ch else None

        # 5. Radio type
        m_radio = re.search(r"^\s*Radio type\s*:\s*(.+)$", out, re.MULTILINE)
        result["radio"] = m_radio.group(1).strip() if m_radio else None

        return result
    except Exception:
        return None


# Backward-compatibility alias
_measure_wifi = read_wifi_stats


# ===========================================================================
# SECTION 1 — AUTO-DETECTION (runs once, cached)
# ===========================================================================

def detect_network_config(force_refresh: bool = False) -> dict:
    """
    Auto-detect local host network configuration.

    Discovers:
      - local_ip: Active local IP using socket connection probe
      - gateway_ip: Default gateway from OS routing table
      - dns_servers: Configured DNS servers list
      - primary_dns: First valid DNS server
      - connection_type: "WiFi", "Ethernet", or "Unknown"

    Results are cached unless force_refresh=True.
    """
    global _config_cache
    if _config_cache is not None and not force_refresh:
        return _config_cache

    local_ip = None
    gateway_ip = None
    dns_servers = []
    connection_type = "Unknown"
    sys_type = platform.system().lower()

    # 1. Local IP via UDP socket trick
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2.0)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    # 2. Gateway IP detection
    try:
        if sys_type == "windows":
            out = subprocess.run(
                ["ipconfig"], capture_output=True, text=True, timeout=6
            ).stdout
            for line in out.splitlines():
                if "Default Gateway" in line:
                    matches = re.findall(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", line)
                    for ip in matches:
                        if not ip.startswith("169.") and ip != "0.0.0.0" and ip != "127.0.0.1":
                            gateway_ip = ip
                            break
                    if gateway_ip:
                        break
        elif sys_type == "linux":
            out = subprocess.run(
                ["ip", "route", "show", "default"], capture_output=True, text=True, timeout=4
            ).stdout
            m = re.search(r"default via (\d+\.\d+\.\d+\.\d+)", out)
            if m:
                gateway_ip = m.group(1)
        elif sys_type == "darwin":
            out = subprocess.run(
                ["netstat", "-rn"], capture_output=True, text=True, timeout=4
            ).stdout
            for line in out.splitlines():
                if line.startswith("default"):
                    parts = line.split()
                    if len(parts) >= 2 and re.match(r"^\d+\.\d+\.\d+\.\d+$", parts[1]):
                        gateway_ip = parts[1]
                        break
    except Exception:
        pass

    # Fallback to standard subnet gateway if detection failed
    if not gateway_ip and local_ip and local_ip != "127.0.0.1":
        octets = local_ip.split(".")
        if len(octets) == 4:
            gateway_ip = f"{octets[0]}.{octets[1]}.{octets[2]}.1"

    # 3. DNS Servers detection
    try:
        if sys_type == "windows":
            out = subprocess.run(
                ["ipconfig", "/all"], capture_output=True, text=True, timeout=6
            ).stdout
            in_dns_block = False
            for line in out.splitlines():
                if "DNS Servers" in line:
                    in_dns_block = True
                    m = re.search(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", line)
                    if m:
                        ip = m.group(1)
                        if not ip.startswith("169.") and ip not in dns_servers:
                            dns_servers.append(ip)
                elif in_dns_block:
                    stripped = line.strip()
                    m = re.match(r"^(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})$", stripped)
                    if m:
                        ip = m.group(1)
                        if not ip.startswith("169.") and ip not in dns_servers:
                            dns_servers.append(ip)
                    elif stripped:
                        in_dns_block = False
        elif sys_type == "linux":
            with open("/etc/resolv.conf", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("nameserver"):
                        parts = line.split()
                        if len(parts) >= 2 and re.match(r"^\d+\.\d+\.\d+\.\d+$", parts[1]):
                            if parts[1] not in dns_servers:
                                dns_servers.append(parts[1])
        elif sys_type == "darwin":
            out = subprocess.run(
                ["scutil", "--dns"], capture_output=True, text=True, timeout=4
            ).stdout
            for m in re.finditer(r"nameserver\[\d+\]\s*:\s*(\d+\.\d+\.\d+\.\d+)", out):
                ip = m.group(1)
                if ip not in dns_servers:
                    dns_servers.append(ip)
    except Exception:
        pass

    primary_dns = dns_servers[0] if dns_servers else None

    # 4. Connection Type detection
    try:
        stats = psutil.net_if_stats()
        io = psutil.net_io_counters(pernic=True)
        active_interfaces = []
        for name, stat in stats.items():
            if stat.isup and name in io:
                nic = io[name]
                if nic.bytes_sent + nic.bytes_recv > 0:
                    active_interfaces.append(name.lower())

        # Exclude virtual / container / hyper-v interfaces from determining physical type
        physical_candidates = [
            iface for iface in active_interfaces
            if not any(v in iface for v in ["vethernet", "wsl", "virtual", "hyper-v", "loopback", "vmware", "docker"])
        ]
        eval_list = physical_candidates if physical_candidates else active_interfaces

        # Check for Wi-Fi keywords first
        for iface in eval_list:
            if any(k in iface for k in ["wi-fi", "wifi", "wlan", "wireless", "802.11"]):
                connection_type = "WiFi"
                break

        # If not Wi-Fi, check for Ethernet keywords
        if connection_type == "Unknown":
            for iface in eval_list:
                if any(k in iface for k in ["ethernet", "eth", "lan", "local area"]):
                    connection_type = "Ethernet"
                    break

        # Verification fallback: if Windows netsh detects an active connected Wi-Fi interface
        if connection_type != "WiFi" and sys_type == "windows":
            wifi_check = read_wifi_stats()
            if wifi_check and wifi_check.get("ssid") and wifi_check.get("ssid") != "Unknown":
                connection_type = "WiFi"
    except Exception:
        pass

    _config_cache = {
        "local_ip": local_ip,
        "gateway_ip": gateway_ip,
        "dns_servers": dns_servers,
        "primary_dns": primary_dns,
        "connection_type": connection_type,
        "detected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return _config_cache


# Backward-compatibility alias
detect_all = detect_network_config


# ===========================================================================
# SECTION 2 — PING BURST (for latency, packet loss, jitter)
# ===========================================================================

def ping_burst(host: str, count: int = PING_COUNT) -> tuple:
    """
    Send `count` individual pings to host.
    Collect successful latency values into a list.

    Returns
    -------
    tuple: (avg_latency, packet_loss, jitter)
      avg_latency : float (mean of successful latencies, or 9999.0 if all failed)
      packet_loss : float ((count - received) / count * 100)
      jitter      : float (numpy std of latencies, or 0.0 if <= 1 received)

    Never raises exceptions — returns (9999.0, 100.0, 0.0) on error.
    Uses timeout of 3 seconds per ping.
    """
    if not host:
        return 9999.0, 100.0, 0.0

    latencies = []
    sys_type = platform.system().lower()

    try:
        for _ in range(count):
            if sys_type == "windows":
                cmd = ["ping", "-n", "1", "-w", "3000", str(host)]
            else:
                cmd = ["ping", "-c", "1", "-W", "3", str(host)]

            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=4.0
                )
                out = proc.stdout

                m = re.search(r"[Tt]ime[=<](\d+(?:\.\d+)?)\s*ms", out)
                if not m:
                    m = re.search(r"Average\s*=\s*(\d+(?:\.\d+)?)\s*ms", out)
                if not m and sys_type != "windows":
                    m = re.search(r"time=(\d+(?:\.\d+)?)\s*ms", out)

                if m:
                    latencies.append(float(m.group(1)))
            except (subprocess.TimeoutExpired, Exception):
                pass

        if latencies:
            avg_latency = round(float(np.mean(latencies)), 2)
            jitter = round(float(np.std(latencies)), 2) if len(latencies) > 1 else 0.0
            packet_loss = round(((count - len(latencies)) / float(count)) * 100.0, 1)
            return avg_latency, packet_loss, jitter
        else:
            return 9999.0, 100.0, 0.0
    except Exception:
        return 9999.0, 100.0, 0.0


# ===========================================================================
# SECTION 4 — NIC ERRORS PER SECOND
# ===========================================================================

def read_nic_errors() -> float:
    """
    Calculate NIC error and packet drop rate per second using psutil.net_io_counters().

    Returns
    -------
    float
      Errors per second (0.0 on first call or on error).
    """
    global _prev_nic_errors, _prev_nic_time
    now = time.time()
    try:
        io = psutil.net_io_counters()
        total_errs = int(io.errin + io.errout + io.dropin + io.dropout)
        if _prev_nic_errors is None or _prev_nic_time is None:
            _prev_nic_errors = total_errs
            _prev_nic_time = now
            return 0.0

        elapsed = max(now - _prev_nic_time, 0.001)
        err_rate = max(0.0, (total_errs - _prev_nic_errors) / elapsed)
        _prev_nic_errors = total_errs
        _prev_nic_time = now
        return round(float(err_rate), 2)
    except Exception:
        return 0.0


# ===========================================================================
# SECTION 5 — TRAFFIC KB/S
# ===========================================================================

def read_traffic_kbps() -> float:
    """
    Calculate aggregate network traffic throughput in KB/s using psutil.net_io_counters().

    Returns
    -------
    float
      Traffic in KB/s (0.0 on first call or on error).
    """
    global _prev_traffic_bytes, _prev_traffic_time
    now = time.time()
    try:
        io = psutil.net_io_counters()
        total_bytes = int(io.bytes_sent + io.bytes_recv)
        if _prev_traffic_bytes is None or _prev_traffic_time is None:
            _prev_traffic_bytes = total_bytes
            _prev_traffic_time = now
            return 0.0

        elapsed = max(now - _prev_traffic_time, 0.001)
        kbps = max(0.0, (total_bytes - _prev_traffic_bytes) / elapsed / 1024.0)
        _prev_traffic_bytes = total_bytes
        _prev_traffic_time = now
        return round(float(kbps), 2)
    except Exception:
        return 0.0


# ===========================================================================
# SECTION 6 — ASSEMBLE READING
# ===========================================================================

def get_real_reading(device_name: str = "My Network") -> dict:
    """
    Measure live network metrics and return a feature dictionary matching
    the complete 15-feature schema (ALL_FEATURES) plus diagnostic metadata.

    Parameters
    ----------
    device_name : str, optional
        Label for the observed network, by default "My Network".

    Returns
    -------
    dict
        Full feature set with rolling averages, trends, and network metadata.
    """
    global _router_lat_history, _dns_lat_history, _rssi_history

    cfg = detect_network_config()
    gateway_ip = cfg.get("gateway_ip")
    primary_dns = cfg.get("primary_dns")
    conn_type = cfg.get("connection_type", "Unknown")

    # 1. Measure Router metrics
    r_lat, r_loss, r_jitter = ping_burst(gateway_ip, count=PING_COUNT)

    # 2. Measure DNS metrics
    d_lat, d_loss, d_jitter = ping_burst(primary_dns, count=PING_COUNT)

    # 3. Overall Jitter is the maximum of router and DNS jitter
    jitter = round(float(max(r_jitter, d_jitter)), 2)

    # 4. Wi-Fi Stats (RSSI and Tx Rate)
    wifi = read_wifi_stats()
    if conn_type == "Ethernet":
        rssi = 0.0
        tx_rate = 0.0
    else:
        if wifi:
            rssi = float(wifi["rssi"]) if wifi.get("rssi") is not None else 0.0
            tx_rate = float(wifi["tx_rate"]) if wifi.get("tx_rate") is not None else 0.0
        else:
            rssi = 0.0
            tx_rate = 0.0

    # 5. NIC Errors & Traffic
    nic_errors = read_nic_errors()
    traffic = read_traffic_kbps()

    # 6. Update rolling histories (length = ROLLING_WINDOW)
    _router_lat_history.append(r_lat)
    _dns_lat_history.append(d_lat)
    _rssi_history.append(rssi)

    # Compute rolling averages (rolling5)
    r_lat_rolling = round(float(np.mean(_router_lat_history)), 2)
    d_lat_rolling = round(float(np.mean(_dns_lat_history)), 2)
    rssi_rolling = round(float(np.mean(_rssi_history)), 2)

    # Compute short-term trends (last - first in rolling window)
    r_lat_trend = round(float(_router_lat_history[-1] - _router_lat_history[0]), 2) if len(_router_lat_history) > 1 else 0.0
    d_lat_trend = round(float(_dns_lat_history[-1] - _dns_lat_history[0]), 2) if len(_dns_lat_history) > 1 else 0.0
    rssi_trend = round(float(_rssi_history[-1] - _rssi_history[0]), 2) if len(_rssi_history) > 1 else 0.0

    # Construct the complete reading
    reading = {
        # Metadata
        "timestamp": time.time(),
        "device": device_name,
        "scenario": device_name,
        # 9 RAW_FEATURES
        "router_latency": r_lat,
        "router_packet_loss": r_loss,
        "dns_latency": d_lat,
        "dns_packet_loss": d_loss,
        "rssi": rssi,
        "tx_rate": tx_rate,
        "jitter": jitter,
        "nic_errors": nic_errors,
        "traffic": traffic,
        # 3 ROLLING_FEATURES
        "router_latency_rolling5": r_lat_rolling,
        "dns_latency_rolling5": d_lat_rolling,
        "rssi_rolling5": rssi_rolling,
        # 3 TREND_FEATURES
        "router_latency_trend": r_lat_trend,
        "dns_latency_trend": d_lat_trend,
        "rssi_trend": rssi_trend,
        # Model supervision labels (healthy baseline for live network)
        "is_failure": 0,
        "failure_type": "none",
        "minutes_to_failure": 10.0,
        # Diagnostic & Raw Display Metadata
        "_local_ip": cfg.get("local_ip"),
        "_gateway_ip": gateway_ip,
        "_dns_ip": primary_dns,
        "_dns_servers": cfg.get("dns_servers", []),
        "_conn_type": conn_type,
        "_ssid": wifi.get("ssid") if wifi else "N/A",
        "_channel": wifi.get("channel") if wifi else None,
        "_radio": wifi.get("radio") if wifi else None,
        "_detected_at": cfg.get("detected_at"),
        # Legacy aliases for compatibility with older components
        "router_latency_ms": r_lat,
        "dns_latency_ms": d_lat,
        "rssi_dbm": rssi,
        "tx_rate_mbps": tx_rate,
        "jitter_ms": jitter,
        "nic_errors_per_sec": nic_errors,
        "traffic_kbps": traffic,
        "_router_ip": gateway_ip,
    }

    # Ensure all ALL_FEATURES have numeric non-None values (replace None with 0.0)
    for feat in ALL_FEATURES:
        if reading.get(feat) is None:
            reading[feat] = 0.0

    return reading


# ===========================================================================
# SECTION 7 — STANDALONE TEST
# ===========================================================================

if __name__ == "__main__":
    cfg = detect_network_config()
    print("=" * 65)
    print("REAL NETWORK COLLECTOR — AUTO-DETECTED CONFIGURATION")
    print("=" * 65)
    print(f"Local IP         : {cfg.get('local_ip')}")
    print(f"Gateway IP       : {cfg.get('gateway_ip')}")
    print(f"Primary DNS      : {cfg.get('primary_dns')}")
    print(f"All DNS Servers  : {cfg.get('dns_servers')}")
    print(f"Connection Type  : {cfg.get('connection_type')}")
    print(f"Detected At      : {cfg.get('detected_at')}")
    print("=" * 65)
    print("Polling real network metrics every 5 seconds. Press Ctrl+C to stop.\n")

    try:
        sample_num = 1
        while True:
            r = get_real_reading()
            is_eth = (r["_conn_type"] == "Ethernet")
            rssi_str = "N/A (Ethernet)" if is_eth else f"{r['rssi']} dBm"
            tx_str = "N/A (Ethernet)" if is_eth else f"{r['tx_rate']} Mbps"

            print(f"--- Sample #{sample_num} [{time.strftime('%H:%M:%S')}] ---")
            print(f"  Router Latency      : {r['router_latency']:>7.2f} ms   (pinged: {r['_gateway_ip']})")
            print(f"  Router Packet Loss  : {r['router_packet_loss']:>7.1f} %")
            print(f"  DNS Latency         : {r['dns_latency']:>7.2f} ms   (pinged: {r['_dns_ip']})")
            print(f"  DNS Packet Loss     : {r['dns_packet_loss']:>7.1f} %")
            print(f"  RSSI                : {rssi_str}")
            print(f"  Transmit Rate       : {tx_str}")
            print(f"  Jitter              : {r['jitter']:>7.2f} ms")
            print(f"  NIC Errors          : {r['nic_errors']:>7.2f} /s")
            print(f"  Traffic             : {r['traffic']:>7.2f} KB/s")
            print(f"  Rolling5 RTT (Router/DNS) : {r['router_latency_rolling5']} ms / {r['dns_latency_rolling5']} ms")
            print(f"  RTT Trend (Router/DNS)    : {r['router_latency_trend']:+.2f} ms / {r['dns_latency_trend']:+.2f} ms")
            print()

            sample_num += 1
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nMonitoring terminated by user.")
