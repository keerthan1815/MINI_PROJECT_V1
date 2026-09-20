"""
Build labelled training data from device-specific network simulators.
"""

import time

import pandas as pd

from features import ALL_FEATURES, RAW_FEATURES, SIM_DEVICES
from network_simulator import NetworkSimulator

OUTPUT_FILE = "training_data.csv"
NUM_READINGS = 2500
WINDOW = 5


def add_rolling_features(df):
    parts = []
    for device in df["device"].unique():
        sub = df[df["device"] == device].copy().reset_index(drop=True)
        sub["router_latency_rolling5"] = (
            sub["router_latency_ms"].rolling(WINDOW, min_periods=1).mean().round(2))
        sub["dns_latency_rolling5"] = (
            sub["dns_latency_ms"].rolling(WINDOW, min_periods=1).mean().round(2))
        sub["rssi_rolling5"] = (
            sub["rssi_dbm"].rolling(WINDOW, min_periods=1).mean().round(2))
        sub["router_trend"] = (
            sub["router_latency_ms"] - sub["router_latency_ms"].shift(WINDOW)
        ).fillna(0).round(2)
        sub["dns_trend"] = (
            sub["dns_latency_ms"] - sub["dns_latency_ms"].shift(WINDOW)
        ).fillna(0).round(2)
        sub["rssi_trend"] = (
            sub["rssi_dbm"] - sub["rssi_dbm"].shift(WINDOW)
        ).fillna(0).round(2)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def main():
    sims = {d: NetworkSimulator(d) for d in SIM_DEVICES}
    rows = []
    print(f"Collecting {NUM_READINGS} rounds x {len(SIM_DEVICES)} devices...")
    for i in range(NUM_READINGS):
        for d in SIM_DEVICES:
            rows.append(sims[d].next_reading())
        if i % 500 == 0:
            print(f"  {i}/{NUM_READINGS}")
        time.sleep(0.002)

    df = add_rolling_features(pd.DataFrame(rows))
    cols = (
        ["timestamp", "device"]
        + RAW_FEATURES
        + [c for c in ALL_FEATURES if c not in RAW_FEATURES]
        + ["is_failure", "failure_type", "minutes_to_failure"]
    )
    df[cols].to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} rows -> {OUTPUT_FILE}")
    print("Normal :", int((df["is_failure"] == 0).sum()))
    print("Failure:", int((df["is_failure"] == 1).sum()))
    print(df["failure_type"].value_counts().to_dict())
    print(df["device"].value_counts().to_dict())


if __name__ == "__main__":
    main()
