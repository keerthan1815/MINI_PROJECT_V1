"""
test_integration.py – End-to-end integration tests for the full pipeline.

Tests:
    1. test_full_pipeline_simulated           — 200 readings → feature build → model.predict()
    2. test_root_cause_consistent_with_simulator — diagnose() never crashes on live readings
    3. test_feature_names_match_model         — model's feature_names_in_ == ALL_FEATURES
    4. test_reading_has_all_required_keys     — schema + numeric-only check on raw reading

Notes:
    • Tests 1 and 3 require failure_model.pkl and are skipped when the file is absent.
    • Rolling/trend features (router_latency_rolling5 etc.) are NOT produced by
      next_reading() directly — they are computed by generate_training_data.py's
      add_rolling_features().  The pipeline test builds them inline with pandas so
      the model can be exercised without running the full data-generation script.
"""

import math
import os
import sys
from collections import deque
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup – allow running pytest from any working directory
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "failure_model.pkl"
CSV_PATH   = ROOT / "training_data.csv"

_model_missing = not MODEL_PATH.exists()
_csv_missing   = not CSV_PATH.exists()

# Decorator applied to tests that need the trained model on disk
requires_model = pytest.mark.skipif(
    _model_missing,
    reason="failure_model.pkl not found — run train_model.py first",
)

from network_simulator import NetworkSimulator
from root_cause import diagnose
from features import (
    ALL_FEATURES, RAW_FEATURES, ROLLING_FEATURES, TREND_FEATURES
)

WINDOW = 5   # must match generate_training_data.py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_feature_df(readings):
    """
    Replicate the pandas rolling/trend feature engineering from
    generate_training_data.py so the integration test does not depend on
    the CSV being regenerated.
    """
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not installed — cannot build feature DataFrame")

    df = pd.DataFrame(readings)
    df["router_latency_rolling5"] = (
        df["router_latency"].rolling(WINDOW, min_periods=1).mean().round(2)
    )
    df["dns_latency_rolling5"] = (
        df["dns_latency"].rolling(WINDOW, min_periods=1).mean().round(2)
    )
    df["rssi_rolling5"] = (
        df["rssi"].rolling(WINDOW, min_periods=1).mean().round(2)
    )
    df["router_latency_trend"] = (
        df["router_latency"] - df["router_latency"].shift(WINDOW)
    ).fillna(0.0).round(2)
    df["dns_latency_trend"] = (
        df["dns_latency"] - df["dns_latency"].shift(WINDOW)
    ).fillna(0.0).round(2)
    df["rssi_trend"] = (
        df["rssi"] - df["rssi"].shift(WINDOW)
    ).fillna(0.0).round(2)
    # Backward compatibility aliases
    df["router_trend"] = df["router_latency_trend"]
    df["dns_trend"] = df["dns_latency_trend"]
    return df


# ===========================================================================
# 1. Full pipeline: simulator → feature engineering → model.predict()
# ===========================================================================

@requires_model
class TestFullPipelineSimulated:
    """
    Drive 200 readings through the complete pipeline:
        NetworkSimulator → raw readings → rolling/trend features → XGBoost.predict()

    Verifies that the feature matrix has the correct 15-column schema and that
    all predictions are valid binary labels (0 or 1).
    """

    @pytest.fixture(scope="class")
    @classmethod
    def pipeline_artefacts(cls):
        """Build 200 readings, feature DataFrame, and model predictions once."""
        try:
            import joblib
            import pandas as pd
        except ImportError:
            pytest.skip("joblib or pandas not installed")

        sim = NetworkSimulator("Router")
        readings = [sim.next_reading() for _ in range(200)]
        df = _build_feature_df(readings)
        model = joblib.load(str(MODEL_PATH))

        # Use the last 10 rows (after warm-up windows are stable)
        X_sample = df[ALL_FEATURES].fillna(0).tail(10)
        preds = model.predict(X_sample)

        return {"df": df, "model": model, "X_sample": X_sample, "preds": preds}

    def test_feature_dataframe_has_15_columns(self, pipeline_artefacts):
        df = pipeline_artefacts["df"]
        missing = [c for c in ALL_FEATURES if c not in df.columns]
        assert not missing, (
            f"Feature DataFrame is missing columns: {missing}"
        )
        assert len(ALL_FEATURES) == 15, (
            f"Expected 15 features in ALL_FEATURES, got {len(ALL_FEATURES)}"
        )

    def test_predictions_are_binary(self, pipeline_artefacts):
        preds = pipeline_artefacts["preds"]
        assert len(preds) == 10, f"Expected 10 predictions, got {len(preds)}"
        for i, p in enumerate(preds):
            assert p in (0, 1), (
                f"Prediction at index {i} is not 0 or 1: {p!r}"
            )

    def test_no_exception_during_predict(self, pipeline_artefacts):
        """
        If this fixture completes without raising, no exception was raised
        during the pipeline.  This is an explicit pass/fail marker.
        """
        assert pipeline_artefacts["preds"] is not None


# ===========================================================================
# 2. Root-cause diagnose() is consistent with every failure reading
# ===========================================================================

class TestRootCauseConsistentWithSimulator:
    """
    Run 500 readings from a Router simulator.  For every reading labelled
    is_failure == 1, pass the reading to diagnose() and verify:
        • reasons list is never empty
        • severity is never None
    """

    def test_diagnose_never_returns_empty_reasons(self):
        sim = NetworkSimulator("Router")
        failure_readings = [
            r for _ in range(500)
            if (r := sim.next_reading())["is_failure"] == 1
        ]

        if not failure_readings:
            pytest.skip("No failure readings produced in 500 ticks — re-run.")

        for r in failure_readings:
            result = diagnose(r)
            assert result["reasons"], (
                f"diagnose() returned empty reasons for reading: {r}"
            )

    def test_diagnose_severity_is_never_none(self):
        sim = NetworkSimulator("Router")
        failure_readings = [
            r for _ in range(500)
            if (r := sim.next_reading())["is_failure"] == 1
        ]

        if not failure_readings:
            pytest.skip("No failure readings produced in 500 ticks — re-run.")

        valid_severities = {"LOW", "MEDIUM", "CRITICAL"}
        for r in failure_readings:
            result = diagnose(r)
            assert result["severity"] in valid_severities, (
                f"diagnose() returned unexpected severity "
                f"{result['severity']!r} for reading: {r}"
            )


# ===========================================================================
# 3. Model feature names exactly match ALL_FEATURES
# ===========================================================================

@requires_model
class TestFeatureNamesMatchModel:
    """
    Load failure_model.pkl and verify that its feature_names_in_ attribute
    matches ALL_FEATURES from features.py exactly — both order and content.

    This is a regression guard for the feature-mismatch error that occurred
    during development when the training schema diverged from the live schema.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def model(cls):
        try:
            import joblib
        except ImportError:
            pytest.skip("joblib not installed")
        return joblib.load(str(MODEL_PATH))

    def test_model_has_feature_names_in(self, model):
        assert hasattr(model, "feature_names_in_"), (
            "Model does not have feature_names_in_ attribute. "
            "Re-train with a scikit-learn-compatible XGBoost version."
        )

    def test_feature_count_matches(self, model):
        if not hasattr(model, "feature_names_in_"):
            pytest.skip("Model has no feature_names_in_")
        model_features = list(model.feature_names_in_)
        assert len(model_features) == len(ALL_FEATURES), (
            f"Model has {len(model_features)} features, "
            f"ALL_FEATURES has {len(ALL_FEATURES)}"
        )

    def test_feature_names_exact_match(self, model):
        if not hasattr(model, "feature_names_in_"):
            pytest.skip("Model has no feature_names_in_")
        model_features = list(model.feature_names_in_)
        assert model_features == ALL_FEATURES, (
            "Feature mismatch between model and ALL_FEATURES!\n"
            f"  Model    : {model_features}\n"
            f"  ALL_FEATURES: {ALL_FEATURES}"
        )

    def test_feature_order_preserved(self, model):
        """Column order matters for XGBoost — a transposition is a silent bug."""
        if not hasattr(model, "feature_names_in_"):
            pytest.skip("Model has no feature_names_in_")
        model_features = list(model.feature_names_in_)
        mismatches = [
            (i, mf, af)
            for i, (mf, af) in enumerate(zip(model_features, ALL_FEATURES))
            if mf != af
        ]
        assert not mismatches, (
            f"Feature order mismatch at positions: "
            + ", ".join(f"[{i}] model={mf!r} ≠ schema={af!r}"
                        for i, mf, af in mismatches)
        )


# ===========================================================================
# 4. Raw reading schema and numeric-only value check
# ===========================================================================

class TestReadingHasAllRequiredKeys:
    """
    Verify the dict returned by next_reading() has the right keys and that
    every value is a numeric type with no None or NaN hidden inside.

    Note: rolling/trend feature keys (e.g. router_latency_rolling5) are NOT
    emitted by next_reading() — they are computed by generate_training_data.py.
    This test validates the raw-reading contract, not the engineered features.
    """

    @pytest.fixture()
    def reading(self):
        return NetworkSimulator("Router").next_reading()

    def test_all_raw_feature_keys_present(self, reading):
        missing = [k for k in RAW_FEATURES if k not in reading]
        assert not missing, (
            f"Raw reading is missing feature keys: {missing}"
        )

    def test_label_keys_present(self, reading):
        for key in ("is_failure", "minutes_to_failure"):
            assert key in reading, (
                f"Label key '{key}' missing from reading"
            )

    def test_rolling_trend_keys_absent_from_raw_reading(self, reading):
        """
        Document the known contract: rolling/trend features are NOT in a raw
        reading.  If this ever starts failing, it means the simulator now
        computes them itself — update generate_training_data.py accordingly.
        """
        engineered = ROLLING_FEATURES + TREND_FEATURES
        present = [k for k in engineered if k in reading]
        assert not present, (
            f"Unexpected: engineered keys found in raw reading: {present}. "
            "Update generate_training_data.py if this is intentional."
        )

    def test_all_raw_values_are_numeric(self, reading):
        for key in RAW_FEATURES:
            val = reading[key]
            assert isinstance(val, (int, float)), (
                f"Key '{key}' has non-numeric value: {val!r} ({type(val).__name__})"
            )
            assert not (isinstance(val, float) and math.isnan(val)), (
                f"Key '{key}' contains NaN"
            )

    def test_label_values_are_numeric(self, reading):
        for key in ("is_failure", "minutes_to_failure"):
            val = reading[key]
            assert isinstance(val, (int, float)), (
                f"Label '{key}' is not numeric: {val!r}"
            )
            assert not (isinstance(val, float) and math.isnan(val)), (
                f"Label '{key}' contains NaN"
            )

    def test_is_failure_is_binary(self, reading):
        assert reading["is_failure"] in (0, 1), (
            f"is_failure must be 0 or 1, got {reading['is_failure']!r}"
        )
