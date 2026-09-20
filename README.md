# 🌐 Explainable AI (XAI) Network Device Failure Prediction & Automated Self-Healing

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%2B%20SHAP-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![SNMP](https://img.shields.io/badge/Protocol-SNMP%20v1%2Fv2c-green.svg)](https://pysnmp.readthedocs.io/)

A predictive maintenance and automated network management system that detects, predicts, and isolates failures in **Network Forwarding Hardware** (*Routers, Switches, WiFi Access Points, Firewalls/ISP Gateways*) prior to total connectivity outage.

Unlike traditional host-monitoring tools that inspect PC CPU or RAM, this project evaluates client-side network symptoms and direct SNMP telemetry to pinpoint network infrastructure degradation with explainable AI (SHAP).

---

## 🚀 Key Features

* **🤖 Machine Learning Failure Prediction**:
  * **XGBoost Classifier**: Classifies network states into `NORMAL` vs `FAILURE` based on 15 network symptoms.
  * **Regression Model**: Predicts estimated **Minutes to Failure** (Lead time window of 3–8 minutes).
* **🔍 Explainable AI (SHAP)**:
  * Generates waterfall and bar attribution charts explaining *why* a failure alert was triggered and *which metric* caused it.
* **📡 Dual Data Collection Modes**:
  * **Real Mode**: Probes live gateway latency, DNS performance, RSSI signal strength via `netsh wlan`, and NIC hardware error counters using `psutil`.
  * **Simulated Mode**: Emulates specific failure modes (Router Overload, WiFi Signal Degradation, Switch Port Errors, Firewall WAN Bottlenecks).
* **🔌 Direct SNMP Monitoring**:
  * Queries real network hardware via standard MIB-II SNMP OIDs (`sysDescr`, `sysUpTime`, `ifInOctets`, `ifOutOctets`, `ifInErrors`, `ifOutErrors`, `ifSpeed`, `hrProcessorLoad`).
* **🛠️ Automated Self-Healing**:
  * Triggers target remedies based on root cause: DNS cache flushing, adapter resets, switch port overload logs, and IT notifications.
* **📊 Experimental Validation Dashboard**:
  * Streamlit interface with 10 interactive pages including real-time SLA metrics, topology maps, PDF reporting, AI Assistant, and **Research Questions (RQ1–RQ5)**.

---

## 📐 Network Component & Symptom Mapping

| Target Device | Network Symptoms Tracked | Detection Method |
| :--- | :--- | :--- |
| **Router** | Gateway RTT, Packet Loss, Jitter | Ping to Default Gateway IP |
| **WiFi Access Point** | RSSI Signal Strength (dBm), TX Rate (Mbps), Link Quality | `netsh wlan show interfaces` |
| **Switch / Bridge** | Physical NIC Frame & CRC Errors, Interface Traffic | `psutil` NIC counter deltas |
| **Firewall / ISP** | WAN Latency, Public DNS Packet Loss | Ping to Primary/Secondary DNS |

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10+ installed
* Git

### 1. Clone the Repository
```bash
git clone <your-github-repo-url>
cd MINI_try1
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 3. (Optional) Train / Re-train ML Models
```bash
python train_model.py
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run live_dashboard.py
```

---

## 📊 Dashboard Modules

1. **Live Monitor**: Real-time status cards, XGBoost confidence, minutes to failure gauge, SHAP explanation charts.
2. **SNMP Monitor**: On-demand SNMP polling form and historical telemetry logs.
3. **Research Questions (RQ1–RQ5)**: Empirical validation addressing predictive accuracy, feature importance, root-cause localization, self-healing ROI, and multi-device scalability.
4. **Topology**: Interactive graph visualization of network path health.
5. **Analytics**: Historical failure trends and confusion matrix metrics.
6. **Logs & Healing**: Event tables and automated remediation action logs.
7. **Reports**: Instant downloadable PDF SLA and health status reports.
8. **AI Assistant**: Natural language Q&A interface for network diagnostic queries.

---

## 📝 License
Distributed under the MIT License.
