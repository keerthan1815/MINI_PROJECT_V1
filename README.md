# Explainable AI-Based Predictive Network Failure Detection Using XGBoost and SHAP

A B.Tech final-year system that predicts **network failures** — including path congestion, DNS/WAN outages, physical layer faults, Wi-Fi degradation, and traffic overload — from nine live path metrics, classifies failure risk with **XGBoost**, and explains every alert with **SHAP** so operators can act before connectivity is lost.

![Tests](https://img.shields.io/badge/tests-112%20passed-brightgreen) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 1. Project Title and Description

**Title:** Explainable AI-Based Predictive Network Failure Detection Using XGBoost and SHAP

**One-line description:** Predict imminent network failures from nine client-observed path metrics, identify the failure scenario with SHAP, and raise a real-time dashboard alert with a minutes-to-failure estimate.

---

## 2. Problem Statement

Modern enterprise and campus networks depend on a layered infrastructure of routers, switches, firewalls, and wireless access points. When a **network failure** occurs — whether due to congestion, packet storm, DNS outage, Wi-Fi signal degradation, or a physical layer fault — the impact is immediate: sessions drop, SLA credits are triggered, VoIP and transaction systems stall, and engineers spend hours correlating symptoms to isolate the root cause.

Traditional Network Management Systems (NMS) and SNMP threshold monitoring are **reactive**. They typically:

- Trigger only after a hard failure event (ping timeout, interface down, CPU trap).
- Treat each telemetry metric independently, so a slow rise in gateway RTT combined with increasing jitter is ignored until packet loss is already catastrophic.
- Provide **no explanation** of *which* symptom or path segment caused the alert, delaying root-cause isolation.
- Cannot predict a failure before it becomes a full outage — operators have no warning window.

The operational gap is therefore: **detect degrading network behaviour early, from observable path telemetry, classify the failure type, and explain the prediction in language an operator can trust and act upon.**

This project addresses that gap. It models **eight distinct network failure scenarios**:

| Scenario | Description |
|----------|-------------|
| `router_congestion` | Path to gateway is congested — elevated RTT and jitter |
| `router_unreachable` | Router not responding — gateway timeout, full packet loss |
| `internet_outage` | DNS/WAN unreachable — internet connectivity lost |
| `wifi_degradation` | RF signal weakening — RSSI drop, TX rate collapse |
| `physical_layer_fault` | Cable or switch-port errors — NIC error spike |
| `network_overload` | Bandwidth saturation — traffic spike, RTT and DNS slow |
| `dns_slowdown` | DNS server sluggish — ISP or resolver issue |
| `packet_storm` | High packet loss across all paths — broadcast/DDoS storm |

The system observes **only network path metrics** (ICMP RTT, DNS RTT, Wi-Fi RSSI/TX rate, NIC error counters, jitter, traffic throughput). It does **not** monitor host CPU or RAM — those are PC resources, not network health indicators.

---

## 3. Proposed Solution

The proposed system is an **explainable, two-model predictive pipeline** wrapped in a real-time Streamlit dashboard.

### XGBoost for classification

A gradient-boosted **XGBoost classifier** maps fifteen engineered features (nine raw path metrics plus rolling means and short-term trends) to a binary label: **Healthy** vs **Network Failure Imminent**. Gradient boosting is well suited to this tabular, mixed-scale, correlated feature set. Class imbalance is corrected with **SMOTE** before training so rare failure signatures are not drowned by healthy samples.

A companion **XGBoost regressor** estimates **minutes to failure**, giving operators a lead-time window rather than a binary flag.

### SHAP for explainability

**SHapley Additive exPlanations (SHAP)** attribute each prediction to individual metrics, enabling operators to understand *why* the model fired:

- High SHAP on `router_latency` + `router_packet_loss` → **Router congestion / unreachable**
- Elevated `nic_errors` + collapsing `tx_rate` → **Physical layer fault / Switch port**
- High `dns_latency` with healthy gateway → **DNS outage / ISP issue**
- Low `rssi` + low `tx_rate` + `router_packet_loss` → **Wi-Fi degradation**
- Spike in `traffic` + elevated RTT/DNS → **Network overload / Packet storm**

### Real-time monitoring

The dashboard operates in two modes:

- **Simulation mode:** Independently streams all 8 failure scenarios with realistic metric degradation and recovery state machines.
- **Live mode:** Collects real measurements from the host (gateway ping, DNS ping, Wi-Fi RSSI/TX rate, NIC error counters) via `real_network_collector.py`.

Predictions, health scores, network topology colour-coding, email/buzzer alerts, recommended network-side healing actions (DNS flush, port flags — no process killing), SLA tracking, and PDF incident reports are updated continuously.

---

## 4. System Architecture

```
 ┌─────────────────────────────┐
 │  Network Path Under Monitor │
 │  8 Failure Scenarios        │
 │  (congestion, outage, etc.) │
 └──────────────┬──────────────┘
                │ ICMP RTT, DNS RTT, RSSI, TX rate, NIC counters
                ▼
 ┌─────────────────────────────┐
 │  Metrics Collection Layer   │
 │  network_simulator.py  OR   │
 │  real_network_collector.py  │
 └──────────────┬──────────────┘
                │ 9 raw metrics → rolling/trend feature engineering
                ▼
 ┌─────────────────────────────┐
 │  XGBoost Prediction Models  │
 │  Classifier: failure? (0/1) │
 │  Regressor: minutes to fail │
 └──────────────┬──────────────┘
                │ failure probability + lead time
                ▼
 ┌─────────────────────────────┐
 │  SHAP Explainability Layer  │
 │  Metric → scenario mapping  │
 │  Root-cause text generation │
 └──────────────┬──────────────┘
                │ ranked symptoms + failure scenario
                ▼
 ┌─────────────────────────────┐
 │  Dashboard + Alerting       │
 │  Streamlit UI               │
 │  Email / audible buzzer     │
 │  Self-healing + SLA + PDF   │
 └─────────────────────────────┘
```

**Data-flow summary:** Live network path → telemetry capture → feature engineering → XGBoost failure classification and time regression → SHAP attribution → operator dashboard with real-time alerts.

---

## 5. Network Metrics (Feature Set)

Nine **raw network path metrics** are monitored. They describe **connection health**, not host resource usage.

| # | Feature Name | What it measures | Failure scenarios indicated |
|---|--------------|------------------|----------------------------|
| 1 | `router_latency` | ICMP RTT to the default gateway (ms) | Router congestion, Router unreachable, Network overload |
| 2 | `router_packet_loss` | % of gateway pings that time out | Router unreachable, Packet storm, Physical layer fault |
| 3 | `dns_latency` | ICMP RTT to configured DNS server (ms) | Internet outage, DNS slowdown, Network overload |
| 4 | `dns_packet_loss` | % of DNS pings that time out | Internet outage, DNS slowdown |
| 5 | `rssi` | Wi-Fi received signal strength (dBm) | Wi-Fi degradation |
| 6 | `tx_rate` | Wi-Fi link rate — drops when RF quality falls (Mbps) | Wi-Fi degradation, Physical layer fault |
| 7 | `jitter` | Std. dev. of recent gateway RTTs — path instability (ms) | Router congestion, Packet storm |
| 8 | `nic_errors` | NIC / switch-port error counter rate (errors/sec) | Physical layer fault, Packet storm |
| 9 | `traffic` | Throughput through the local NIC (kbps) | Network overload, Packet storm |

**Derived features (six, computed over a 5-sample rolling window — not additional sensors):**
- Rolling means: `router_latency_rolling5`, `dns_latency_rolling5`, `rssi_rolling5`
- Short-term trends: `router_latency_trend`, `dns_latency_trend`, `rssi_trend`

**Total model input: 15 features.**

---

## 6. Methodology

### 6.1 Data collection and simulation

Training data is generated by `generate_training_data.py` by running all 8 failure scenarios through `NetworkSimulator`, each with a realistic state machine:

- **Pre-failure phase** (10–20 readings): gradual metric degradation with proportional SHAP-visible deltas.
- **Hard failure phase** (8–18 readings): full symptom impact — packet loss at 100%, RSSI at floor, RTT at 9999ms sentinel.
- **Recovery phase**: automatic `_reset_baseline()` — simulator returns to healthy metrics.

Each sample is tagged with `is_failure` (binary), `failure_type` (8-class), and `minutes_to_failure` (regression target, clipped to 0–10 minutes). Rolling and trend columns are computed per scenario over a window of 5 readings via `pandas.rolling()`.

### 6.2 SMOTE balancing

Network failure events are rarer than healthy samples. **SMOTE** (`imbalanced-learn`) oversamples the minority class **for the classifier only**, producing a balanced training set while leaving the regressor on the true `minutes_to_failure` distribution.

### 6.3 XGBoost training

The classifier (`XGBClassifier`) uses 200 estimators, max depth 6, learning rate 0.1, and column/row subsampling (0.8 each). An 80/20 stratified split is used for evaluation. Training outputs: `failure_model.pkl`, `confusion_matrix.png`, `roc_curve.png`, and `shap_summary_bar.png`.

### 6.4 SHAP integration

A `shap.TreeExplainer` is fitted to the trained XGBoost trees. Global beeswarm plots (written to `shap_beeswarm.png`) show which path symptoms drive failure predictions across the test set. At runtime, per-prediction SHAP values are mapped through `METRIC_TO_DEVICE` so each alert names the **suspected failure scenario** with supporting metric evidence.

### 6.5 Regression model

An `XGBRegressor` predicts `minutes_to_failure` (clipped 0–10 minute horizon). Reported metrics: MAE, R², and detection rate for the critical < 2-minute window. Saved to `regression_model.pkl`.

---

## 7. Automated Testing

The project includes a comprehensive automated test suite with **112 tests** covering all core modules.

```
tests/
├── conftest.py                        # 17 shared fixtures (simulators, readings, DB, DataFrames)
├── test_network_simulator.py          # Simulator schema, bounds, injection, all 8 scenarios
├── test_health_scorer.py              # Health score correctness, label boundaries
├── test_root_cause.py                 # diagnose() correctness for all failure patterns
├── test_integration.py               # End-to-end pipeline: simulator → features → XGBoost.predict()
├── test_edge_cases_and_exceptions.py  # 41 edge case and exception handling fixtures
└── test_alerts_and_healing.py        # 18 tests for email alert, buzzer, self-healing routing
```

Run the test suite:

```bash
python -m pytest --tb=short -v
```

Run with coverage:

```bash
python -m pytest --cov=. --cov-config=.coveragerc --cov-report=term-missing
```

A **GitHub Actions CI workflow** (`.github/workflows/python-tests.yml`) automatically runs the full test suite on Python 3.11 and 3.12 on every push and pull request.

---

## 8. Model Performance

Fill this table after running `python train_model.py`. Values are printed to the console.

| Metric | Score |
|--------|-------|
| Accuracy | — |
| Precision | — |
| Recall | — |
| F1-Score | — |
| ROC-AUC | — |

**Regression (same training run):** MAE (minutes) — ; R² — ; detection rate for < 2-min window — %.

Plots written by training: `confusion_matrix.png`, `roc_curve.png`, `shap_summary_bar.png`, `shap_beeswarm.png`, `regression_distribution.png`.

---

## 9. Installation

**Requirements:** Python 3.10+ (3.11 recommended), pip, and (for live Wi-Fi metrics on Windows) a wireless NIC.

```bash
git clone https://github.com/keerthan1815/MINI_PROJECT_V1.git
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

## 10. How to Run

### Step 1 — Generate training data

```bash
python generate_training_data.py
```

Runs all 8 failure scenarios through `NetworkSimulator` and writes `training_data.csv` (~24,000 labelled samples).

### Step 2 — Train the models

```bash
python train_model.py
```

Produces `failure_model.pkl`, `regression_model.pkl`, and all evaluation charts. Copy printed metrics into Section 8.

### Step 3 — Run the dashboard

```bash
python -m streamlit run live_dashboard.py
```

Log in (default accounts: `admin` / `admin123`, `engineer` / `eng123`, `viewer` / `view123`). Use simulation tabs for any of the 8 failure scenarios, or switch to the live **My Network** collector. Change default passwords before any shared or production deployment.

### Step 4 — Run automated tests (optional)

```bash
python -m pytest --tb=short -v
```

---

## 11. Project Structure

```
MiniProject_V1/
├── README.md                   # This document
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Test discovery and marker configuration
├── .coveragerc                 # Coverage measurement settings
├── .gitignore                  # Virtualenv, bytecode, env files
├── .github/
│   └── workflows/
│       └── python-tests.yml    # GitHub Actions CI (Python 3.11 + 3.12)
│
├── config.py                   # Single source of truth: feature names, thresholds, paths
├── features.py                 # Feature schema, metric→scenario mapping, SHAP labels
├── generate_training_data.py   # Simulates all 8 scenarios and builds labelled CSV
├── train_model.py              # SMOTE, XGBoost classifier/regressor, SHAP plots
├── live_dashboard.py           # Streamlit UI: predict, explain, alert, topology
│
├── network_simulator.py        # 8 failure scenario state machines (not PC health)
├── real_network_collector.py   # Live gateway/DNS/Wi-Fi/NIC measurements
│
├── root_cause.py               # Rule overlay: symptoms → failure scenario diagnosis
├── health_scorer.py            # Network path health score 0–100
├── system_status.py            # Live healthy/degraded/down state per scenario
│
├── database.py                 # SQLite: readings, alerts, SLA, healing logs
├── failure_logger.py           # Persist predicted network failures
├── sla_monitor.py              # SLA = uptime readings / total readings
│
├── self_healing.py             # Network-side recovery (DNS flush, port flags)
├── healing_memory.py           # CSV memory of issue → action → success
├── email_alert.py              # SMTP alert on predicted network failure
├── buzzer.py                   # Local audible alarm (Windows beep)
├── report_pdf.py               # Explainable PDF network failure incident report
├── chat_assistant.py           # Intent assistant (status, SHAP, SLA, healing queries)
├── auth.py                     # Role-based dashboard login
│
├── tests/
│   ├── conftest.py                        # Shared fixtures
│   ├── test_network_simulator.py          # Simulator unit tests (all 8 scenarios)
│   ├── test_health_scorer.py              # Health scoring unit tests
│   ├── test_root_cause.py                 # Root-cause diagnosis unit tests
│   ├── test_integration.py               # End-to-end pipeline integration tests
│   ├── test_edge_cases_and_exceptions.py  # Edge cases and exception handling
│   └── test_alerts_and_healing.py        # Alert and healing module tests
│
├── training_data.csv           # Generated labelled training set
├── network_logs.csv            # Runtime path-status log
├── healing_memory.csv          # Healing action history
├── failure_model.pkl           # Trained XGBoost classifier
└── regression_model.pkl        # Trained minutes-to-failure regressor
```

---

## 12. Research References

1. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD)*, 2016, pp. 785–794.

2. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

3. N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357, 2002.

4. Alghamdi et al., "Artificial Intelligence for Predictive Failures of Network Devices," *IJISRT*, 2025.

5. Authors, "Explainable AI-Based Predictive Network Failure Detection Using XGBoost and SHAP," *[Venue / Journal — to be filled]*, 2026.

---

## 13. License

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
