"""
Explainable AI-Based Predictive Failure Detection for
Network Devices Using XGBoost and SHAP

System layer: Model Training Layer + Explainability Layer.

Algorithms / techniques:
    - SMOTE minority oversampling (classifier only)
    - XGBClassifier (device failure vs normal)
    - XGBRegressor (minutes to failure)
    - Stratified train/test split, ROC-AUC, confusion matrix
    - XGBoost feature_importances_
    - SHAP TreeExplainer beeswarm on held-out samples

Inputs:
    - training_data.csv with ALL_FEATURES and is_failure / minutes_to_failure

Outputs:
    - failure_model.pkl, regression_model.pkl
    - confusion_matrix.png, roc_curve.png, shap_summary_bar.png,
      shap_beeswarm.png, regression_distribution.png
    - Console Accuracy, Precision, Recall, F1, ROC-AUC, MAE, R²

Research reference:
    Alghamdi et al. (2025), IJISRT,
    "Artificial Intelligence for Predictive Failures of Network Devices"
"""

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, auc, classification_report,
    confusion_matrix, f1_score, mean_absolute_error, precision_score,
    r2_score, recall_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from features import ALL_FEATURES, RAW_FEATURES, ROLLING_FEATURES, TREND_FEATURES

DATA_FILE = "training_data.csv"
MODEL_FILE = "failure_model.pkl"
REGRESSION_FILE = "regression_model.pkl"

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# Print a training-pipeline section header (data load, SMOTE, XGBoost, SHAP).
def sep(title=""):
    print("\n" + "=" * 55)
    if title:
        print(title)
        print("=" * 55)


# Train explainable device-failure models: SMOTE → XGBoost → SHAP plots.
def main():
    sep("STEP 1: Load network-device training data")
    df = pd.read_csv(DATA_FILE)
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        print("Missing columns:", missing)
        print("Run: python generate_training_data.py")
        return

    print(f"Rows: {len(df)}  Normal: {(df.is_failure==0).sum()}  "
          f"Failure: {(df.is_failure==1).sum()}")
    X = df[ALL_FEATURES].fillna(0)
    y_cls = df["is_failure"]
    y_reg = df["minutes_to_failure"]

    sep("STEP 2: SMOTE (classifier only)")
    X_bal, y_bal = SMOTE(random_state=42).fit_resample(X, y_cls)
    print(f"Balanced Normal={(y_bal==0).sum()} Failure={(y_bal==1).sum()}")

    X_tr, X_te, yc_tr, yc_te = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal)
    X_r_tr, X_r_te, yr_tr, yr_te = train_test_split(
        X, y_reg, test_size=0.2, random_state=42)

    sep("STEP 3: XGBoost classifier")
    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42,
    )
    clf.fit(X_tr, yc_tr)
    joblib.dump(clf, MODEL_FILE)
    yc_pred = clf.predict(X_te)
    yc_prob = clf.predict_proba(X_te)[:, 1]
    acc = accuracy_score(yc_te, yc_pred)
    f1 = f1_score(yc_te, yc_pred)
    print(classification_report(yc_te, yc_pred, target_names=["Normal", "Failure"]))
    print(f"Accuracy {acc*100:.2f}%  Precision {precision_score(yc_te, yc_pred)*100:.2f}%  "
          f"Recall {recall_score(yc_te, yc_pred)*100:.2f}%  F1 {f1*100:.2f}%")

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(
        confusion_matrix(yc_te, yc_pred),
        display_labels=["Normal", "Failure"],
    ).plot(cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix — Network Device Failure\nXGBoost", fontweight="bold")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(yc_te, yc_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, "darkorange", lw=2, label=f"XGBoost (AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "navy", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC — Network Device Failure Classifier", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    plt.close()

    sep("STEP 4: XGBoost regressor (minutes to failure)")
    reg = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42,
    )
    reg.fit(X_r_tr, yr_tr)
    joblib.dump(reg, REGRESSION_FILE)
    yr_pred = np.clip(reg.predict(X_r_te), 0, 10)
    mae = mean_absolute_error(yr_te, yr_pred)
    r2 = r2_score(yr_te, yr_pred)
    crit_acc = ((yr_te < 2).astype(int) == (yr_pred < 2).astype(int)).mean() * 100
    print(f"MAE={mae:.3f} min  R2={r2:.4f}  <2min detection={crit_acc:.1f}%")

    plt.figure(figsize=(10, 5))
    plt.hist(yr_pred, bins=30, alpha=0.6, label="Predicted", color="orange")
    plt.hist(yr_te, bins=30, alpha=0.6, label="Actual", color="blue")
    plt.axvline(2, color="red", linestyle="--", label="Critical (2 min)")
    plt.xlabel("Minutes to device failure")
    plt.ylabel("Count")
    plt.title(f"Minutes-to-Failure  MAE={mae:.3f}  R²={r2:.4f}", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("regression_distribution.png", dpi=150)
    plt.close()

    sep("STEP 5: Feature importance")
    fi = clf.feature_importances_
    idx = np.argsort(fi)[::-1]
    colors = (["#ff4444"] * len(RAW_FEATURES)
              + ["#ff8800"] * len(ROLLING_FEATURES)
              + ["#0088ff"] * len(TREND_FEATURES))
    plt.figure(figsize=(13, 6))
    names = [ALL_FEATURES[i] for i in idx]
    vals = [fi[i] for i in idx]
    bars = plt.bar(names, vals, color=[colors[i] for i in idx])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                 f"{v:.3f}", ha="center", fontsize=8)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.title("XGBoost Feature Importance — Network Device Symptoms\n"
              "Red=raw  Orange=rolling  Blue=trend", fontweight="bold")
    plt.tight_layout()
    plt.savefig("shap_summary_bar.png", dpi=150)
    plt.close()

    if SHAP_AVAILABLE:
        try:
            ex = shap.TreeExplainer(clf)
            smp = X_te.iloc[: min(300, len(X_te))]
            sv = ex.shap_values(smp)
            plt.figure(figsize=(11, 7))
            shap.summary_plot(sv, smp, feature_names=ALL_FEATURES, show=False)
            plt.title("SHAP — which network symptoms drive failure", fontweight="bold")
            plt.tight_layout()
            plt.savefig("shap_beeswarm.png", dpi=150)
            plt.close()
            print("shap_beeswarm.png saved")
        except Exception as e:
            print("SHAP chart skipped:", e)

    sep("DONE")
    print(f"Classifier -> {MODEL_FILE}  Acc={acc*100:.2f}%  AUC={roc_auc:.4f}")
    print(f"Regressor  -> {REGRESSION_FILE}  MAE={mae:.3f}")
    print("Next: python -m streamlit run live_dashboard.py")


if __name__ == "__main__":
    main()
