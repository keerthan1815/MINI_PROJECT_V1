"""SNMP Device Monitoring Collector.

Queries real network devices (Routers, Switches, Access Points, Servers) via SNMP v1/v2c.
Extracts system details, uptime, interface bandwidth/traffic, interface errors, CPU load.
Inserts readings into database.py.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

try:
    from pysnmp.hlapi.v3arch.asyncio import (
        get_cmd, SnmpEngine, CommunityData, UdpTransportTarget,
        ContextData, ObjectIdentity, ObjectType
    )
    HAS_PYSNMP = True
except ImportError:
    HAS_PYSNMP = False

from database import insert_snmp_reading

# Standard MIB-II OIDs
OIDS = {
    "sysDescr": "1.3.6.1.2.1.1.1.0",
    "sysUpTime": "1.3.6.1.2.1.1.3.0",
    "sysName": "1.3.6.1.2.1.1.5.0",
    "ifInOctets": "1.3.6.1.2.1.2.2.1.10.1",
    "ifOutOctets": "1.3.6.1.2.1.2.2.1.16.1",
    "ifInErrors": "1.3.6.1.2.1.2.2.1.14.1",
    "ifOutErrors": "1.3.6.1.2.1.2.2.1.20.1",
    "ifSpeed": "1.3.6.1.2.1.2.2.1.5.1",
    "cpuLoad": "1.3.6.1.2.1.25.3.3.1.2.1",
}


async def _async_query_snmp(ip: str = "127.0.0.1", community: str = "public",
                           port: int = 161, timeout: float = 1.5, retries: int = 1) -> Dict[str, Any]:
    """Perform real SNMP GET queries for standard network MIB items."""
    if not HAS_PYSNMP:
        return {
            "status": "ERROR",
            "ip": ip,
            "error": "pysnmp library not installed"
        }

    try:
        target = await UdpTransportTarget.create((ip, port), timeout=timeout, retries=retries)
        engine = SnmpEngine()
        auth = CommunityData(community, mpModel=0)  # SNMP v1/v2c
        ctx = ContextData()

        # Build list of ObjectTypes
        object_types = [ObjectType(ObjectIdentity(oid)) for oid in OIDS.values()]

        errorIndication, errorStatus, errorIndex, varBinds = await get_cmd(
            engine, auth, target, ctx, *object_types
        )

        if errorIndication:
            return {
                "status": "UNREACHABLE",
                "ip": ip,
                "error": str(errorIndication)
            }
        elif errorStatus:
            return {
                "status": "ERROR",
                "ip": ip,
                "error": f"{errorStatus.prettyPrint()} at {errorIndex}"
            }

        # Parse results
        results_map = {}
        oid_keys = list(OIDS.keys())
        for idx, varBind in enumerate(varBinds):
            key = oid_keys[idx] if idx < len(oid_keys) else f"oid_{idx}"
            results_map[key] = varBind[1]

        # Process fields
        sys_name = str(results_map.get("sysName", "Unknown"))
        sys_descr = str(results_map.get("sysDescr", "N/A"))
        
        # sysUpTime is in hundredths of a second (TimeTicks)
        uptime_ticks = int(results_map.get("sysUpTime", 0))
        uptime_sec = uptime_ticks / 100.0

        if_in_octets = int(results_map.get("ifInOctets", 0))
        if_out_octets = int(results_map.get("ifOutOctets", 0))
        if_in_errors = int(results_map.get("ifInErrors", 0))
        if_out_errors = int(results_map.get("ifOutErrors", 0))
        
        speed_bps = int(results_map.get("ifSpeed", 0))
        if_speed_mbps = speed_bps / 1_000_000.0 if speed_bps > 0 else 100.0

        cpu_val = int(results_map.get("cpuLoad", 0))

        return {
            "status": "OK",
            "ip": ip,
            "sys_name": sys_name,
            "sys_descr": sys_descr,
            "uptime_sec": uptime_sec,
            "if_in_octets": if_in_octets,
            "if_out_octets": if_out_octets,
            "if_in_errors": if_in_errors,
            "if_out_errors": if_out_errors,
            "if_speed_mbps": if_speed_mbps,
            "cpu_usage": float(cpu_val),
            "mem_usage": 0.0,
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "ip": ip,
            "error": str(e)
        }


def poll_snmp_device(ip: str = "127.0.0.1", community: str = "public",
                     port: int = 161, timeout: float = 1.5) -> Dict[str, Any]:
    """Synchronous entry point for querying SNMP device and saving to database."""
    res = asyncio.run(_async_query_snmp(ip=ip, community=community, port=port, timeout=timeout))

    if res.get("status") == "OK":
        insert_snmp_reading(
            ip=res["ip"],
            sys_name=res["sys_name"],
            sys_descr=res["sys_descr"],
            uptime_sec=res["uptime_sec"],
            if_in_octets=res["if_in_octets"],
            if_out_octets=res["if_out_octets"],
            if_in_errors=res["if_in_errors"],
            if_out_errors=res["if_out_errors"],
            if_speed_mbps=res["if_speed_mbps"],
            cpu_usage=res["cpu_usage"],
            mem_usage=res["mem_usage"],
            status="OK"
        )
    else:
        # Save attempt log so UI shows attempt timestamp and status
        insert_snmp_reading(
            ip=ip,
            sys_name="N/A",
            sys_descr=res.get("error", "Unreachable"),
            uptime_sec=0,
            if_in_octets=0,
            if_out_octets=0,
            if_in_errors=0,
            if_out_errors=0,
            if_speed_mbps=0,
            cpu_usage=0,
            mem_usage=0,
            status=res.get("status", "UNREACHABLE")
        )

    return res


if __name__ == "__main__":
    print("Testing SNMP collection on 127.0.0.1...")
    result = poll_snmp_device("127.0.0.1")
    print("Result:", result)
