# Explainable AI-Based Predictive Failure Detection for Network Devices Using XGBoost and SHAP

A B.Tech final-year system that predicts **router, switch, and firewall failures** from live network symptoms, classifies risk with **XGBoost**, and explains every alert with **SHAP** so operators can act before the path goes down.

---

## 1. Project Title and Description

**Title:** Explainable AI-Based Predictive Failure Detection for Network Devices Using XGBoost and SHAP

**One-line description:** Predict imminent network-device failure from nine client-observed path metrics, rank the suspected device with SHAP, and raise a dashboard alert with minutes-to-failure estimate.

---

## 2. Problem Statement

Enterprise connectivity depends on a small set of infrastructure devices—**routers**, **switches**, and **firewalls**. When any of these fail, the cost is not a single host going offline. Sessions drop, SLA credits are triggered, VoIP and transaction systems stall, and engineers spend hours isolating whether the fault is the access point, a switch port, the default gateway, or the WAN/firewall path.

Traditional Network Management Systems (NMS) and SNMP threshold monitoring are **reactive**. They typically:

- Fire only after a hard outage (ping timeout, interface down, CPU trap on the box itself).
- Treat each metric independently, so a slow rise in gateway RTT plus jitter is ignored until packet loss is already severe.
- Provide **no explanation** of *which* symptom caused the alert, which delays root-cause analysis.
- Often confuse **host health** (PC CPU/RAM) with **device health**. A laptop under load is not evidence that the router is failing.

The operational gap is therefore: **detect degrading network-device behaviour early, from observable path symptoms, and explain the prediction in language an operator can trust.**

This project addresses that gap. It does **not** monitor PC CPU or RAM. It observes ICMP RTT to the default gateway and DNS, Wi-Fi radio quality, NIC/switch-port error rates, jitter, and path throughput—symptoms of failing network devices as seen from the client side.

---

## 3. Proposed Solution

The proposed system is an **explainable, two-model pipeline** wrapped in a real-time Streamlit dashboard.

### XGBoost for classification

A gradient-boosted **XGBoost classifier** maps fifteen engineered features (nine raw metrics plus rolling means and short-term trends) to a binary label: **Normal** vs **Device failure**. Gradient boosting is well suited to this tabular, mixed-scale, mildly correlated feature set. Class imbalance is corrected with **SMOTE** before training so rare failure signatures are not drowned by healthy samples.

A companion **XGBoost regressor** estimates **minutes to failure**, giving operators a lead-time window rather than a binary flag.

### SHAP for explainability

**SHapley Additive exPlanations (SHAP)** attribute each prediction to individual metrics. A high SHAP value on `router_latency_ms` and `router_packet_loss` points to the **router**; elevated `nic_errors_per_sec` with collapsing `tx_rate_mbps` points to a **switch / AP**; high `dns_latency_ms` with a healthy gateway points to the **firewall / ISP path**. Operators see *why* the model fired, not only *that* it fired.

### Real-time monitoring

The dashboard either:

- **Simulates** independent failure signatures for Router, Switch, and Firewall; or
- **Collects live measurements** from the host (gateway ping, DNS ping, Wi-Fi RSSI/TX rate, NIC counters).

Predictions, health scores, topology colour, optional email/buzzer alerts, recommended network-side healing (DNS flush, port flags—not process killing), SLA, and PDF reports are updated continuously.

---

## 4. System Architecture

```
 ┌──────────────────────┐
 │  Network devices     │
 │  Router / Switch /   │
 │  Firewall / Wi-Fi AP │
 └──────────┬───────────┘
            │ ICMP, DNS, RSSI, NIC counters
            ▼
 ┌──────────────────────┐
 │  Metrics collection  │
 │  Simulator  OR       │
 │  real_network_       │
 │  collector.py        │
 └──────────┬───────────┘
            │ 9 raw + rolling/trend features
            ▼
 ┌──────────────────────┐
 │  XGBoost models      │
 │  Classifier: fail?   │
 │  Regressor: minutes  │
 └──────────┬───────────┘
            │ probability + lead time
            ▼
 ┌──────────────────────┐
 │  SHAP explanation    │
 │  Metric → device map │
 │  Root-cause text     │
 └──────────┬───────────┘
            │ ranked symptoms + suspected device
            ▼
 ┌──────────────────────┐
 │  Dashboard + alerts  │
 │  Streamlit UI        │
 │  Email / buzzer      │
 │  Healing + SLA + PDF │
 └──────────────────────┘
```

**Data-flow summary:** device path → metric capture → feature engineering → XGBoost classification and regression → SHAP attribution → operator dashboard and alerts.

---

## 5. Features

Nine **raw network metrics** are monitored. They describe **device path health**, not host resource usage.

| # | Metric | What it measures | Device it primarily indicates |
|---|--------|------------------|-------------------------------|
| 1 | `router_latency_ms` | ICMP RTT to the default gateway | **Router** |
| 2 | `router_packet_loss` | Percentage of gateway pings that time out | **Router** |
| 3 | `dns_latency_ms` | ICMP RTT to configured DNS (WAN / ISP) | **Firewall / ISP** |
| 4 | `dns_packet_loss` | Percentage of DNS pings that time out | **Firewall / ISP** |
| 5 | `rssi_dbm` | Wi-Fi received signal strength | **Wi-Fi AP / Router** |
| 6 | `tx_rate_mbps` | Wi-Fi link rate (drops when RF quality falls) | **Wi-Fi AP / Switch** |
| 7 | `jitter_ms` | Standard deviation of recent gateway RTTs | **Router** |
| 8 | `nic_errors_per_sec` | NIC / switch-port error counter rate | **Switch** |
| 9 | `traffic_kbps` | Throughput through the local NIC (path load) | **Firewall / path overload** |

**Derived features (used by the model, not additional sensors):** five-sample rolling means (`router_latency_rolling5`, `dns_latency_rolling5`, `rssi_rolling5`) and trends (`router_trend`, `dns_trend`, `rssi_trend`). Total model input: **15 features**.

---

## 6. Methodology

### 6.1 Data collection

Labelled training rows are generated by `generate_training_data.py` using device-specific simulators:

- **Router** signatures: gateway latency spikes, router packet loss, jitter, Wi-Fi AP fade.
- **Switch** signatures: NIC/switch-port errors, TX-rate collapse.
- **Firewall** signatures: DNS/WAN latency, DNS loss, traffic overload.

Each sample is tagged with `is_failure`, `failure_type`, and `minutes_to_failure`. Rolling and trend columns are computed per device over a window of five readings. Live inference uses the same schema from `real_network_collector.py` (gateway, DNS, radio, NIC)—never PC CPU or RAM.

### 6.2 SMOTE balancing

Failure events are rarer than healthy samples. **SMOTE** (`imbalanced-learn`) oversamples the minority class **for the classifier only**, producing a balanced training set while leaving the regressor on the original time-to-failure distribution.

### 6.3 XGBoost training

The classifier (`XGBClassifier`) is trained with 200 estimators, max depth 6, learning rate 0.1, and column/row subsampling. An 80/20 stratified split is used for evaluation. Artefacts include `failure_model.pkl`, a confusion matrix, ROC curve, and feature-importance plot.

### 6.4 SHAP integration

A `shap.TreeExplainer` is fitted to the trained trees. Global beeswarm plots show which symptoms drive failure across the test set. At runtime, SHAP values are mapped through `METRIC_TO_DEVICE` so each alert names the **suspected device class**.

### 6.5 Regression model

An `XGBRegressor` predicts `minutes_to_failure` (clipped to a 0–10 minute horizon). Mean Absolute Error (MAE) and R² are reported; a critical-window check evaluates whether samples under two minutes are identified as such. The saved model is `regression_model.pkl`.

---

## 7. Model Performance

Fill this table after running `python train_model.py`. Values are printed in the console (`Accuracy`, `Precision`, `Recall`, `F1`, ROC-AUC).

| Metric | Score |
|--------|-------|
| Accuracy | — |
| Precision | — |
| Recall | — |
| F1-Score | — |
| ROC-AUC | — |

**Regression (optional, same training run):** MAE (minutes) — ; R² — ; detection of &lt; 2 min window — %.

Plots written by training: `confusion_matrix.png`, `roc_curve.png`, `shap_summary_bar.png`, `shap_beeswarm.png`, `regression_distribution.png`.

---

## 8. Installation

**Requirements:** Python 3.10+ (3.11 recommended), pip, and (for live Wi-Fi metrics on Windows) a wireless NIC.

```bash
git clone <repository-url>
cd MiniProject_V1

python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

Core packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `shap`, `joblib`, `streamlit`, `streamlit-autorefresh`, `plotly`, `psutil`, `reportlab`, `matplotlib`.

---

## 9. How to Run

### Step 1 — Generate training data

```bash
python generate_training_data.py
```

Writes `training_data.csv` (simulated Router / Switch / Firewall readings with labels).

### Step 2 — Train the models

```bash
python train_model.py
```

Produces `failure_model.pkl`, `regression_model.pkl`, and evaluation figures. Confirm metrics and copy them into Section 7.

### Step 3 — Run the dashboard

```bash
python -m streamlit run live_dashboard.py
```

Log in (default local accounts: `admin` / `admin123`, `engineer` / `eng123`, `viewer` / `view123`). Use simulated device tabs or the live **My Network** collector. Change default passwords before any shared or production deployment.

---

## 10. Project Structure

```
MiniProject_V1/
├── README.md                  # This document
├── requirements.txt           # Python dependencies
├── .gitignore                 # Virtualenv, bytecode, and env files
│
├── features.py                # Nine raw metrics, rolling/trend schema, metric→device map
├── generate_training_data.py  # Builds labelled CSV from device simulators
├── train_model.py             # SMOTE, XGBoost classifier/regressor, SHAP plots
├── live_dashboard.py          # Streamlit UI: predict, explain, alert, topology
│
├── network_simulator.py       # Device-specific failure signatures (not PC health)
├── real_network_collector.py  # Live gateway/DNS/Wi-Fi/NIC measurements
├── network_detector.py        # Wi-Fi vs LAN detection helper
│
├── root_cause.py              # Rule overlay: symptoms → failing device class
├── health_scorer.py           # Health score from network symptoms + failure probability
├── system_status.py           # Live healthy/degraded/down state per device
├── networktopology_map.py     # Plotly topology coloured by device health
│
├── database.py                # SQLite: readings, alerts, SLA, healing logs
├── logger.py                  # CSV + SQLite logger for path status
├── failure_logger.py          # Persist predicted device failures
├── sla_monitor.py             # SLA = uptime readings / total readings
│
├── self_healing.py            # Network-side recovery recommendations (DNS flush, port flags)
├── healing_memory.py          # CSV memory of issue → action success
├── email_alert.py             # SMTP alert on predicted device failure
├── buzzer.py                  # Local audible alarm (Windows beep)
├── report_pdf.py              # Explainable PDF incident report
├── chat_assistant.py          # Intent assistant (status, SHAP, SLA, healing)
├── auth.py                    # Role-based dashboard login
│
├── training_data.csv          # Generated labelled training set
├── network_logs.csv           # Runtime path-status log
├── healing_memory.csv         # Healing action history
├── failure_model.pkl          # Trained XGBoost classifier (after Step 2)
└── regression_model.pkl       # Trained minutes-to-failure regressor (after Step 2)
```

Runtime files created on first dashboard use (not always in source control): `network_monitor.db`, `failure_history.csv`, and training plots (`confusion_matrix.png`, `roc_curve.png`, `shap_beeswarm.png`, etc.).

---

## 11. Research Paper Reference

This implementation is intended to accompany an academic paper on **explainable predictive maintenance of network devices**. Suggested citations for related work and methods:

1. T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD)*, 2016, pp. 785–794.

2. S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

3. N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “SMOTE: Synthetic minority over-sampling technique,” *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357, 2002.

4. J. A. G. Berral *et al.*, “Towards energy-aware scheduling in data centers using machine learning,” *Proc. Int. Conf. Energy-Efficient Computing and Networking*, 2010. *(context: ML for infrastructure health)*

5. Authors, “Explainable AI-Based Predictive Failure Detection for Network Devices Using XGBoost and SHAP,” *[Venue / Journal — to be filled]*, 2026.

Use IEEE or ACM style consistently in the manuscript. After publication, replace entry 5 with the DOI and full bibliographic record.

---

## 12. License

This project is released under the **MIT License**.

```
MIT License

Copyright (c) 2026 MiniProject_V1 authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
