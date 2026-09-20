"""
Measure live NETWORK DEVICE health from this host.

PC CPU / RAM are never collected.
Values come from gateway ping, DNS ping, WiFi radio, and NIC counters.
"""

import math
import platform
import re
import socket
import subprocess
import time
from collections import deque

import psutil

from features import TIMEOUT_MS

HISTORY = 8

_prev_bytes = None
_prev_errs = None
_prev_t = None
_router_ok = deque(maxlen=HISTORY)
_dns_ok = deque(maxlen=HISTORY)
_rtt_hist = deque(maxlen=5)
_router_lat_hist = deque(maxlen=5)
_dns_lat_hist = deque(maxlen=5)
_rssi_hist = deque(maxlen=5)
_cache = {}


def _detect_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def _detect_gateway():
    sys = platform.system().lower()
    try:
        if sys == "windows":
            out = subprocess.run(
                ["ipconfig"], capture_output=True, text=True, timeout=6
            ).stdout
            for line in out.split("\n"):
                if "Default Gateway" in line:
                    m = re.search(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", line)
                    if m:
                        ip = m.group(1)
                        if not ip.startswith("169.") and ip != "127.0.0.1":
                            return ip
        elif sys == "linux":
            out = subprocess.run(
                ["ip", "route", "show", "default"],
                capture_output=True, text=True, timeout=4,
            ).stdout
            m = re.search(r"default via (\d+\.\d+\.\d+\.\d+)", out)
            if m:
                return m.group(1)
        elif sys == "darwin":
            out = subprocess.run(
                ["netstat", "-rn"], capture_output=True, text=True, timeout=4
            ).stdout
            for line in out.split("\n"):
                if line.startswith("default"):
                    parts = line.split()
                    if len(parts) >= 2 and re.match(r"\d+\.\d+\.\d+\.\d+", parts[1]):
                        return parts[1]
    except Exception:
        pass
    local = _detect_local_ip()
    if local:
        a, b, c, _ = local.split(".")
        return f"{a}.{b}.{c}.1"
    return None


def _detect_dns_servers():
    servers = []
    sys = platform.system().lower()
    try:
        if sys == "windows":
            out = subprocess.run(
                ["ipconfig", "/all"], capture_output=True, text=True, timeout=6
            ).stdout
            in_dns = False
            for line in out.split("\n"):
                if "DNS Servers" in line:
                    in_dns = True
                    m = re.search(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", line)
                    if m and not m.group(1).startswith("169."):
                        servers.append(m.group(1))
                elif in_dns:
                    stripped = line.strip()
                    m = re.match(r"^(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})$", stripped)
                    if m and m.group(1) not in servers:
                        servers.append(m.group(1))
                    elif stripped:
                        in_dns = False
        elif sys == "linux":
            with open("/etc/resolv.conf", "r") as f:
                for line in f:
                    if line.startswith("nameserver"):
                        parts = line.split()
                        if len(parts) >= 2 and parts[1] not in servers:
                            servers.append(parts[1])
        elif sys == "darwin":
            out = subprocess.run(
                ["scutil", "--dns"], capture_output=True, text=True, timeout=4
            ).stdout
            for m in re.finditer(
                r"nameserver\[\d+\]\s*:\s*(\d+\.\d+\.\d+\.\d+)", out
            ):
                if m.group(1) not in servers:
                    servers.append(m.group(1))
    except Exception:
        pass
    return servers


def _detect_connection_type():
    try:
        stats = psutil.net_if_stats()
        io = psutil.net_io_counters(pernic=True)
        active = []
        for name, stat in stats.items():
            if stat.isup and name in io:
                nic = io[name]
                if nic.bytes_sent + nic.bytes_recv > 0:
                    active.append(name.lower())
        for name in active:
            if any(k in name for k in ["wi-fi", "wifi", "wlan", "wireless", "wl"]):
                return "WiFi"
            if any(k in name for k in ["ethernet", "eth", "lan", "local area"]):
                return "Ethernet"
    except Exception:
        pass
    return "Unknown"


def detect_all():
    global _cache
    if _cache:
        return _cache
    dns_list = _detect_dns_servers()
    _cache = {
        "local_ip": _detect_local_ip(),
        "gateway_ip": _detect_gateway(),
        "dns_servers": dns_list,
        "primary_dns": dns_list[0] if dns_list else None,
        "connection_type": _detect_connection_type(),
        "detected_at": time.strftime("%H:%M:%S"),
    }
    return _cache


def refresh_detection():
    global _cache
    _cache = {}
    return detect_all()


def _ping(host):
    if not host:
        return None, None
    try:
        sys = platform.system().lower()
        cmd = (["ping", "-n", "1", "-w", "1000", host]
               if sys == "windows"
               else ["ping", "-c", "1", "-W", "2", host])
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5
        ).stdout
        if sys == "windows":
            m = re.search(r"[Aa]verage\s*=\s*(\d+)\s*ms", out)
            if not m:
                m = re.search(r"[Tt]ime[=<](\d+)\s*ms", out)
        else:
            m = re.search(r"= [\d.]+/([\d.]+)/[\d.]+", out)
        latency = float(m.group(1)) if m else TIMEOUT_MS
        pl_m = re.search(r"(\d+)%\s*(packet\s*)?loss", out.lower())
        packet_loss = (float(pl_m.group(1)) if pl_m
                       else (0.0 if latency < 9000 else 100.0))
        return latency, packet_loss
    except subprocess.TimeoutExpired:
        return TIMEOUT_MS, 100.0
    except Exception:
        return TIMEOUT_MS, 100.0


def _measure_wifi():
    if platform.system().lower() != "windows":
        return None
    try:
        out = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True, text=True, timeout=4,
        ).stdout
        if "There is no wireless interface" in out or "SSID" not in out:
            return None
        result = {}
        m = re.search(r"^\s+SSID\s*:\s*(.+)$", out, re.MULTILINE)
        result["ssid"] = m.group(1).strip() if m else "Unknown"
        m = re.search(r"[Ss]ignal\s*:\s*(-\d+)\s*[Dd][Bb][Mm]", out)
        if m:
            result["rssi"] = float(m.group(1))
        else:
            m = re.search(r"[Ss]ignal\s*:\s*(\d+)\s*%", out)
            if m:
                pct = float(m.group(1))
                result["rssi"] = round(-100 + pct * 0.6, 1)
            else:
                result["rssi"] = None
        m = re.search(r"[Tt]ransmit rate\s*:\s*([\d.]+)", out)
        if m:
            result["tx_rate"] = float(m.group(1))
        else:
            m = re.search(r"[Rr]eceive rate\s*:\s*([\d.]+)", out)
            result["tx_rate"] = float(m.group(1)) if m else None
        m = re.search(r"[Cc]hannel\s*:\s*(\d+)", out)
        result["channel"] = int(m.group(1)) if m else None
        m = re.search(r"[Rr]adio type\s*:\s*(.+)", out)
        result["radio"] = m.group(1).strip() if m else None
        return result
    except Exception:
        return None


def _nic_deltas():
    """Return (traffic_kbps, nic_errors_per_sec) from counter deltas."""
    global _prev_bytes, _prev_errs, _prev_t
    now = time.time()
    net = psutil.net_io_counters()
    bytes_now = net.bytes_sent + net.bytes_recv
    errs_now = int(net.errin + net.errout + net.dropin + net.dropout)
    traffic = 0.0
    err_rate = 0.0
    if _prev_bytes is not None and _prev_t is not None:
        elapsed = max(now - _prev_t, 0.1)
        traffic = max((bytes_now - _prev_bytes) / elapsed / 1024.0, 0.0)
        err_rate = max((errs_now - _prev_errs) / elapsed, 0.0)
    _prev_bytes = bytes_now
    _prev_errs = errs_now
    _prev_t = now
    return round(traffic, 2), round(err_rate, 2)


def _mean(dq, fallback):
    return round(sum(dq) / len(dq), 2) if dq else fallback


def _trend(dq):
    lst = list(dq)
    return round(lst[-1] - lst[0], 2) if len(lst) >= 2 else 0.0


def _loss_window(dq):
    if not dq:
        return 0.0
    return round(100.0 * sum(1 for ok in dq if not ok) / len(dq), 1)


def get_real_reading(device_name="My Network"):
    cfg = detect_all()
    gateway = cfg["gateway_ip"]
    dns = cfg["primary_dns"]
    wifi = _measure_wifi()
    traffic, err_rate = _nic_deltas()

    r_lat, _ = _ping(gateway)
    d_lat, _ = _ping(dns)

    if r_lat is None:
        r_lat = TIMEOUT_MS
    if d_lat is None:
        d_lat = TIMEOUT_MS

    router_ok = r_lat < 9000
    dns_ok = d_lat < 9000
    _router_ok.append(router_ok)
    _dns_ok.append(dns_ok)
    if router_ok:
        _rtt_hist.append(r_lat)
        _router_lat_hist.append(r_lat)
    if dns_ok:
        _dns_lat_hist.append(d_lat)

    rtts = list(_rtt_hist)
    if len(rtts) >= 2:
        mean = sum(rtts) / len(rtts)
        jitter = math.sqrt(sum((x - mean) ** 2 for x in rtts) / len(rtts))
    else:
        jitter = 0.0

    rssi = wifi["rssi"] if wifi else None
    tx_rate = wifi["tx_rate"] if wifi else None
    ssid = wifi["ssid"] if wifi else None
    if rssi is not None:
        _rssi_hist.append(rssi)

    # Ethernet has no RF metrics. Do not invent a failing AP.
    model_rssi = rssi if rssi is not None else -45.0
    model_tx = tx_rate if tx_rate is not None else 1000.0

    reading = {
        "timestamp": time.time(),
        "device": device_name,
        "router_latency_ms": round(r_lat, 2),
        "router_packet_loss": _loss_window(_router_ok),
        "dns_latency_ms": round(d_lat, 2),
        "dns_packet_loss": _loss_window(_dns_ok),
        "rssi_dbm": round(model_rssi, 1),
        "tx_rate_mbps": round(model_tx, 1),
        "jitter_ms": round(jitter, 2),
        "nic_errors_per_sec": err_rate,
        "traffic_kbps": traffic,
        "router_latency_rolling5": _mean(_router_lat_hist, r_lat if router_ok else 0),
        "dns_latency_rolling5": _mean(_dns_lat_hist, d_lat if dns_ok else 0),
        "rssi_rolling5": _mean(_rssi_hist, model_rssi),
        "router_trend": _trend(_router_lat_hist),
        "dns_trend": _trend(_dns_lat_hist),
        "rssi_trend": _trend(_rssi_hist),
        "is_failure": 0,
        "failure_type": "none",
        "minutes_to_failure": 10.0,
        "_local_ip": cfg["local_ip"],
        "_router_ip": gateway,
        "_dns_ip": dns,
        "_dns_servers": cfg["dns_servers"],
        "_conn_type": cfg["connection_type"],
        "_ssid": ssid,
        "_wifi_channel": wifi["channel"] if wifi else None,
        "_radio_type": wifi["radio"] if wifi else None,
        "_wifi_available": wifi is not None,
        "_detected_at": cfg["detected_at"],
    }
    return reading


if __name__ == "__main__":
    cfg = detect_all()
    print("Local IP :", cfg["local_ip"])
    print("Router   :", cfg["gateway_ip"])
    print("DNS      :", cfg["primary_dns"])
    print("Link     :", cfg["connection_type"])
    r = get_real_reading()
    for k in [
        "router_latency_ms", "router_packet_loss", "dns_latency_ms",
        "dns_packet_loss", "rssi_dbm", "tx_rate_mbps", "jitter_ms",
        "nic_errors_per_sec", "traffic_kbps",
    ]:
        print(f"{k:22} {r[k]}")
