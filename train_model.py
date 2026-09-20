"""
train_model.py — End-to-end training pipeline for XGBoost failure prediction & SHAP explainability.

Pipeline stages:
  1. Load training dataset from TRAINING_DATA & validate schema
  2. Balance classification classes with SMOTE
  3. Stratified 80/20 train/test split for classifier, separate 80/20 split for regressor
  4. Train XGBoost Classifier (is_failure) & evaluate (Accuracy, Precision, Recall, F1, ROC-AUC)
  5. Generate Confusion Matrix chart (CHART_CONFUSION_MATRIX)
  6. Generate ROC Curve chart with AUC score (CHART_ROC_CURVE)
  7. Train XGBoost Regressor (minutes_to_failure) & evaluate (MAE, R2, Critical Acc)
  8. Generate Feature Importance bar chart with color-coded feature groups (CHART_SHAP_BAR)
  9. Generate SHAP Beeswarm summary plot (CHART_SHAP_BEESWARM)
  10. Print comprehensive metrics and output artifact paths
"""

import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from config import (
    ALL_FEATURES,
    CHART_CONFUSION_MATRIX,
    CHART_REGRESSION_DIST,
    CHART_ROC_CURVE,
    CHART_SHAP_BAR,
    CHART_SHAP_BEESWARM,
    CLASS_LABEL,
    MODEL_CLF_PATH,
    MODEL_REG_PATH,
    RAW_FEATURES,
    REGRESSION_LABEL,
    ROLLING_FEATURES,
    SMOTE_RANDOM_STATE,
    TRAINING_DATA,
    TREND_FEATURES,
    XGBOOST_CLF_PARAMS,
    XGBOOST_REG_PARAMS,
)

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def print_step(step_num: int, title: str):
    """Print formatted section header."""
    print(f"\n{'=' * 65}")
    print(f"STEP {step_num} — {title}")
    print(f"{'=' * 65}")


def main():
    # -----------------------------------------------------------------------
    # STEP 1 — Load data from TRAINING_DATA
    # -----------------------------------------------------------------------
    print_step(1, "Load data from TRAINING_DATA")

    if not os.path.exists(TRAINING_DATA):
        print(f"ERROR: Training data file '{TRAINING_DATA}' was not found.")
        print("Please run generate_training_data.py first to create the dataset:")
        print("    python generate_training_data.py")
        sys.exit(1)

    df = pd.read_csv(TRAINING_DATA)
    print(f"Loaded dataset: '{TRAINING_DATA}' with {len(df)} total rows and {len(df.columns)} columns.")

    missing_features = [col for col in ALL_FEATURES if col not in df.columns]
    if missing_features:
        print(f"ERROR: Missing expected feature columns in '{TRAINING_DATA}':")
        for col in missing_features:
            print(f"  - {col}")
        print("Please run generate_training_data.py first to regenerate with all features:")
        print("    python generate_training_data.py")
        sys.exit(1)

    normal_count = int((df[CLASS_LABEL] == 0).sum())
    failure_count = int((df[CLASS_LABEL] == 1).sum())
    print(f"Class distribution in raw data:")
    print(f"  Normal  ({CLASS_LABEL}=0) : {normal_count:>6} rows ({normal_count / len(df) * 100:.1f}%)")
    print(f"  Failure ({CLASS_LABEL}=1) : {failure_count:>6} rows ({failure_count / len(df) * 100:.1f}%)")

    X = df[ALL_FEATURES].fillna(0.0)
    y_cls = df[CLASS_LABEL]
    y_reg = df[REGRESSION_LABEL]

    # -----------------------------------------------------------------------
    # STEP 2 — SMOTE balancing for classifier
    # -----------------------------------------------------------------------
    print_step(2, "SMOTE balancing for classifier")
    print(f"Before SMOTE:")
    print(f"  Normal: {(y_cls == 0).sum()} | Failure: {(y_cls == 1).sum()}")

    smote = SMOTE(random_state=SMOTE_RANDOM_STATE)
    X_bal, y_bal = smote.fit_resample(X, y_cls)

    print(f"After SMOTE (balanced):")
    print(f"  Normal: {(y_bal == 0).sum()} | Failure: {(y_bal == 1).sum()}")
    print(f"Total balanced samples: {len(y_bal)}")

    # -----------------------------------------------------------------------
    # STEP 3 — 80/20 train/test split
    # -----------------------------------------------------------------------
    print_step(3, "80/20 train/test split")

    # Classifier: stratified split on SMOTE-balanced data
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_bal,
        y_bal,
        test_size=0.2,
        random_state=SMOTE_RANDOM_STATE,
        stratify=y_bal,
    )
    print(f"Classifier split (stratified):")
    print(f"  Train samples: {len(X_clf_train)} | Test samples: {len(X_clf_test)}")

    # Regressor: separate split on original unbalanced data
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X,
        y_reg,
        test_size=0.2,
        random_state=SMOTE_RANDOM_STATE,
    )
    print(f"Regressor split (natural distribution):")
    print(f"  Train samples: {len(X_reg_train)} | Test samples: {len(X_reg_test)}")

    # -----------------------------------------------------------------------
    # STEP 4 — Train XGBoost Classifier
    # -----------------------------------------------------------------------
    print_step(4, "Train XGBoost Classifier")
    print(f"Hyperparameters: {XGBOOST_CLF_PARAMS}")

    clf = XGBClassifier(**XGBOOST_CLF_PARAMS)
    clf.fit(X_clf_train, y_clf_train)

    joblib.dump(clf, MODEL_CLF_PATH)
    print(f"Saved classifier model -> '{MODEL_CLF_PATH}'")

    y_clf_pred = clf.predict(X_clf_test)
    y_clf_prob = clf.predict_proba(X_clf_test)[:, 1]

    acc = accuracy_score(y_clf_test, y_clf_pred)
    prec = precision_score(y_clf_test, y_clf_pred, zero_division=0)
    rec = recall_score(y_clf_test, y_clf_pred, zero_division=0)
    f1 = f1_score(y_clf_test, y_clf_pred, zero_division=0)
    roc_auc = roc_auc_score(y_clf_test, y_clf_prob)

    print("\nClassifier Performance Metrics:")
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  Recall    : {rec * 100:.2f}%")
    print(f"  F1-Score  : {f1 * 100:.2f}%")
    print(f"  ROC-AUC   : {roc_auc:.4f}")

    print("\nFull Classification Report:")
    print(classification_report(y_clf_test, y_clf_pred, target_names=["Normal", "Failure"]))

    # -----------------------------------------------------------------------
    # STEP 5 — Confusion matrix chart
    # -----------------------------------------------------------------------
    print_step(5, "Generate Confusion Matrix chart")

    cm = confusion_matrix(y_clf_test, y_clf_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Failure"])
    disp.plot(cmap="Blues", ax=ax, values_format="d")

    title_text = f"Confusion Matrix — Network Failure Detection\nAccuracy: {acc * 100:.2f}% | F1 Score: {f1 * 100:.2f}%"
    ax.set_title(title_text, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(CHART_CONFUSION_MATRIX, dpi=150)
    plt.close()
    print(f"Saved confusion matrix chart -> '{CHART_CONFUSION_MATRIX}'")

    # -----------------------------------------------------------------------
    # STEP 6 — ROC curve chart
    # -----------------------------------------------------------------------
    print_step(6, "Generate ROC Curve chart")

    fpr, tpr, _ = roc_curve(y_clf_test, y_clf_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#ff7f0e", lw=2.5, label=f"XGBoost Classifier (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#1d3557", lw=1.5, linestyle="--", label="Random Guess (AUC = 0.5000)")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title(f"ROC Curve — Network Failure Prediction (AUC = {roc_auc:.4f})", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHART_ROC_CURVE, dpi=150)
    plt.close()
    print(f"Saved ROC curve chart -> '{CHART_ROC_CURVE}'")

    # -----------------------------------------------------------------------
    # STEP 7 — XGBoost Regressor for minutes to failure
    # -----------------------------------------------------------------------
    print_step(7, "XGBoost Regressor for minutes to failure")
    print(f"Hyperparameters: {XGBOOST_REG_PARAMS}")

    reg = XGBRegressor(**XGBOOST_REG_PARAMS)
    reg.fit(X_reg_train, y_reg_train)

    joblib.dump(reg, MODEL_REG_PATH)
    print(f"Saved regression model -> '{MODEL_REG_PATH}'")

    y_reg_pred = np.clip(reg.predict(X_reg_test), 0.0, 10.0)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    r2 = r2_score(y_reg_test, y_reg_pred)

    # Critical detection accuracy (< 2.0 minutes to failure)
    crit_actual = (y_reg_test < 2.0).astype(int)
    crit_pred = (y_reg_pred < 2.0).astype(int)
    crit_acc = float((crit_actual == crit_pred).mean() * 100.0)

    print("\nRegressor Performance Metrics:")
    print(f"  MAE (Mean Absolute Error)     : {mae:.3f} minutes")
    print(f"  R² Score                      : {r2:.4f}")
    print(f"  Critical (<2 min) Accuracy    : {crit_acc:.2f}%")

    # Save regression distribution chart
    plt.figure(figsize=(10, 5))
    plt.hist(y_reg_pred, bins=30, alpha=0.6, label="Predicted (min)", color="#ff7f0e")
    plt.hist(y_reg_test, bins=30, alpha=0.6, label="Actual (min)", color="#1f77b4")
    plt.axvline(2.0, color="red", linestyle="--", lw=2, label="Critical Threshold (2.0 min)")
    plt.xlabel("Minutes to Failure", fontsize=11)
    plt.ylabel("Sample Count", fontsize=11)
    plt.title(
        f"Minutes-to-Failure Distribution\nMAE: {mae:.3f} min | R²: {r2:.4f} | Critical Acc: {crit_acc:.1f}%",
        fontsize=12,
        fontweight="bold",
    )
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHART_REGRESSION_DIST, dpi=150)
    plt.close()
    print(f"Saved regression distribution chart -> '{CHART_REGRESSION_DIST}'")

    # -----------------------------------------------------------------------
    # STEP 8 — Feature importance chart
    # -----------------------------------------------------------------------
    print_step(8, "Feature Importance chart")

    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    # Assign group colors: red for RAW, orange for ROLLING, blue for TREND
    color_map = {}
    for feat in RAW_FEATURES:
        color_map[feat] = "#e63946"   # Red
    for feat in ROLLING_FEATURES:
        color_map[feat] = "#f77f00"   # Orange
    for feat in TREND_FEATURES:
        color_map[feat] = "#1d3557"   # Blue

    sorted_features = [ALL_FEATURES[i] for i in sorted_idx]
    sorted_values = [float(importances[i]) for i in sorted_idx]
    bar_colors = [color_map[feat] for feat in sorted_features]

    plt.figure(figsize=(13, 6))
    bars = plt.bar(sorted_features, sorted_values, color=bar_colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, sorted_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.xlabel("Network Performance Indicator", fontsize=11, labelpad=8)
    plt.ylabel("Gini Importance (Weight)", fontsize=11)
    plt.title("Which network metric best predicts failure?", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=35, ha="right", fontsize=9)

    # Custom legend for color code
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e63946", edgecolor="black", label="Raw Features"),
        Patch(facecolor="#f77f00", edgecolor="black", label="Rolling Features"),
        Patch(facecolor="#1d3557", edgecolor="black", label="Trend Features"),
    ]
    plt.legend(handles=legend_elements, loc="upper right", fontsize=10)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHART_SHAP_BAR, dpi=150)
    plt.close()
    print(f"Saved feature importance chart -> '{CHART_SHAP_BAR}'")

    # -----------------------------------------------------------------------
    # STEP 9 — SHAP beeswarm (optional, skip if SHAP unavailable)
    # -----------------------------------------------------------------------
    print_step(9, "SHAP Beeswarm summary plot")

    if SHAP_AVAILABLE:
        try:
            print("Computing SHAP values with TreeExplainer...")
            explainer = shap.TreeExplainer(clf)
            sample_subset = X_clf_test.iloc[: min(300, len(X_clf_test))]
            shap_values = explainer.shap_values(sample_subset)

            # In binary classification shap_values can be a list or 2D array
            if isinstance(shap_values, list) and len(shap_values) == 2:
                sv_to_plot = shap_values[1]
            else:
                sv_to_plot = shap_values

            plt.figure(figsize=(11, 7))
            shap.summary_plot(
                sv_to_plot,
                sample_subset,
                feature_names=ALL_FEATURES,
                show=False,
            )
            plt.title("SHAP Beeswarm — Impact of Network Metrics on Failure Prediction", fontsize=12, fontweight="bold", pad=10)
            plt.tight_layout()
            plt.savefig(CHART_SHAP_BEESWARM, dpi=150)
            plt.close()
            print(f"Saved SHAP beeswarm plot -> '{CHART_SHAP_BEESWARM}'")
        except Exception as e:
            print(f"WARNING: SHAP beeswarm generation encountered an error and was skipped: {e}")
    else:
        print("WARNING: SHAP package is not available. Skipping SHAP beeswarm plot.")

    # -----------------------------------------------------------------------
    # STEP 10 — Final summary
    # -----------------------------------------------------------------------
    print_step(10, "Final Summary")

    print("Project: Explainable AI-Based Predictive Failure Detection for Network Devices Using XGBoost and SHAP")
    print("Features: 15 network performance indicators")
    print("Model 1: XGBoost Classifier — predicts Normal/Failure")
    print("Model 2: XGBoost Regressor — predicts minutes to failure")

    print("\nModel Evaluation Summary:")
    print(f"  Classifier Accuracy      : {acc * 100:.2f}%")
    print(f"  Classifier Precision     : {prec * 100:.2f}%")
    print(f"  Classifier Recall        : {rec * 100:.2f}%")
    print(f"  Classifier F1-Score      : {f1 * 100:.2f}%")
    print(f"  Classifier ROC-AUC       : {roc_auc:.4f}")
    print(f"  Regressor MAE            : {mae:.3f} minutes")
    print(f"  Regressor R²             : {r2:.4f}")
    print(f"  Critical Window Accuracy : {crit_acc:.2f}%")

    print("\nArtifacts Created:")
    artifacts = [
        ("XGBoost Classifier", MODEL_CLF_PATH),
        ("XGBoost Regressor", MODEL_REG_PATH),
        ("Confusion Matrix", CHART_CONFUSION_MATRIX),
        ("ROC Curve Chart", CHART_ROC_CURVE),
        ("Regression Distribution", CHART_REGRESSION_DIST),
        ("Feature Importance Bar", CHART_SHAP_BAR),
    ]
    if SHAP_AVAILABLE and os.path.exists(CHART_SHAP_BEESWARM):
        artifacts.append(("SHAP Beeswarm Plot", CHART_SHAP_BEESWARM))

    for label, path in artifacts:
        print(f"  [OK] {label:<25} -> {path}")

    print("\nNext step: streamlit run live_dashboard.py\n")


if __name__ == "__main__":
    main()
