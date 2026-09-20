"""
generate_training_data.py — Generates labelled training dataset for network failure prediction.

Collects telemetry from NetworkSimulator across all 8 network failure scenarios:
  - 8 failure scenarios from config.FAILURE_SCENARIOS
  - NUM_READINGS_PER_DEVICE samples per scenario
  - Rolling mean and lag trend features computed per scenario group
  - Supervision labels: is_failure (binary), failure_type (multiclass), minutes_to_failure (regression)
  - Saves to TRAINING_DATA (training_data.csv)
"""

import time
import pandas as pd

from config import (
    ALL_FEATURES,
    FAILURE_SCENARIOS,
    NUM_READINGS_PER_DEVICE,
    RAW_FEATURES,
    ROLLING_FEATURES,
    ROLLING_WINDOW,
    SIMULATION_SPEED,
    TRAINING_DATA,
    TREND_FEATURES,
)
from network_simulator import NetworkSimulator


def compute_scenario_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling-window averages and short-term trends separately
    for each scenario group so readings across scenarios are never mixed.

    Parameters
    ----------
    df : pd.DataFrame
        Raw telemetry dataframe containing 'scenario' and RAW_FEATURES.

    Returns
    -------
    pd.DataFrame
        Dataframe with ROLLING_FEATURES and TREND_FEATURES populated.
    """
    scenario_dfs = []

    for scenario_name in df["scenario"].unique():
        sub = df[df["scenario"] == scenario_name].copy().reset_index(drop=True)

        # 1. Rolling window averages (window = ROLLING_WINDOW, min_periods=1)
        sub["router_latency_rolling5"] = (
            sub["router_latency"].rolling(ROLLING_WINDOW, min_periods=1).mean().round(2)
        )
        sub["dns_latency_rolling5"] = (
            sub["dns_latency"].rolling(ROLLING_WINDOW, min_periods=1).mean().round(2)
        )
        sub["rssi_rolling5"] = (
            sub["rssi"].rolling(ROLLING_WINDOW, min_periods=1).mean().round(2)
        )

        # 2. Short-term trends (current - value ROLLING_WINDOW ticks ago)
        sub["router_latency_trend"] = (
            sub["router_latency"] - sub["router_latency"].shift(ROLLING_WINDOW)
        ).fillna(0.0).round(2)
        sub["dns_latency_trend"] = (
            sub["dns_latency"] - sub["dns_latency"].shift(ROLLING_WINDOW)
        ).fillna(0.0).round(2)
        sub["rssi_trend"] = (
            sub["rssi"] - sub["rssi"].shift(ROLLING_WINDOW)
        ).fillna(0.0).round(2)

        scenario_dfs.append(sub)

    return pd.concat(scenario_dfs, ignore_index=True)


def generate_dataset() -> pd.DataFrame:
    """
    Simulate network telemetry for all 8 scenarios and assemble the training dataset.

    Returns
    -------
    pd.DataFrame
        Complete engineered training dataset.
    """
    scenarios = list(FAILURE_SCENARIOS.keys())
    total_expected = len(scenarios) * NUM_READINGS_PER_DEVICE
    print(f"Generating training data across {len(scenarios)} scenarios...")
    print(f"Readings per scenario: {NUM_READINGS_PER_DEVICE}")
    print(f"Total expected rows  : {total_expected}")

    # One NetworkSimulator instance per scenario
    simulators = {s: NetworkSimulator(scenario_name=s) for s in scenarios}
    rows = []

    for i in range(NUM_READINGS_PER_DEVICE):
        for s in scenarios:
            reading = simulators[s].next_reading()
            rows.append(reading)

        if (i + 1) % 500 == 0 or (i + 1) == NUM_READINGS_PER_DEVICE:
            collected_so_far = (i + 1) * len(scenarios)
            print(f"  Progress: tick {i + 1}/{NUM_READINGS_PER_DEVICE} ({collected_so_far}/{total_expected} rows)")

    raw_df = pd.DataFrame(rows)

    # Compute rolling and trend features per scenario group
    featured_df = compute_scenario_features(raw_df)

    # Ensure columns match target schema
    output_columns = (
        ["timestamp", "scenario"]
        + RAW_FEATURES
        + ROLLING_FEATURES
        + TREND_FEATURES
        + ["is_failure", "failure_type", "minutes_to_failure"]
    )

    final_df = featured_df[output_columns]

    # Save to TRAINING_DATA
    final_df.to_csv(TRAINING_DATA, index=False)

    # Print summary statistics
    total_rows = len(final_df)
    normal_count = int((final_df["is_failure"] == 0).sum())
    failure_count = int((final_df["is_failure"] == 1).sum())

    print("\n" + "=" * 60)
    print("TRAINING DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total rows collected : {total_rows}")
    print(f"Normal readings count: {normal_count} ({normal_count / total_rows * 100:.1f}%)")
    print(f"Failure readings count: {failure_count} ({failure_count / total_rows * 100:.1f}%)")
    print("\nCount per failure_type:")
    for ft, count in final_df["failure_type"].value_counts().items():
        print(f"  {ft:<22}: {count:>5}")
    print(f"\nFile saved to path   : {TRAINING_DATA}")
    print("=" * 60)

    return final_df


if __name__ == "__main__":
    generate_dataset()
