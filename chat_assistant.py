"""Intent-based assistant about network DEVICE failure (XGBoost + SHAP)."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sla_monitor import calculate_sla
from system_status import get_overall_status, status

TRAINING_DATA = [
    "network status", "is the internet working", "network down",
    "connection ok", "wifi status",
    "router status", "switch status", "firewall status",
    "which device is failing", "router health", "device failure",
    "ai prediction", "xgboost accuracy", "model performance",
    "failure prediction", "how accurate is the model",
    "shap explanation", "why failure", "which metric caused failure",
    "explain prediction", "feature importance",
    "self healing", "what was healed", "healing history",
    "sla percentage", "uptime", "availability",
    "recent alerts", "which device is critical",
    "help", "what can you do",
]

LABELS = (
    ["NETWORK"] * 5
    + ["DEVICE"] * 6
    + ["AI_MODEL"] * 5
    + ["SHAP"] * 5
    + ["HEALING"] * 3
    + ["SLA"] * 3
    + ["ALERTS"] * 2
    + ["HELP"] * 2
)

_vec = TfidfVectorizer()
_X = _vec.fit_transform(TRAINING_DATA)


def detect_intent(text):
    v = _vec.transform([text.lower()])
    sim = cosine_similarity(v, _X)
    idx = sim.argmax()
    if sim[0][idx] < 0.15:
        return "UNKNOWN"
    return LABELS[idx]


def _emoji(state):
    return {"healthy": "🟢", "warning": "🟡", "failure": "🔴"}.get(state, "⚪")


def handle_query(user_input, context=None):
    intent = detect_intent(user_input)
    overall = get_overall_status()

    if intent == "NETWORK":
        return {"response": f"""
**Network path health:** {overall}

This system does **not** score your PC. It scores:
- Router (gateway RTT / loss / jitter)
- Switch (NIC errors, TX collapse)
- Firewall / ISP (DNS RTT / loss, overload)
"""}

    if intent == "DEVICE":
        lines = [
            f"{_emoji(status.get(d, 'healthy'))} **{d}**: {status.get(d, 'healthy').upper()}"
            for d in ["Router", "Switch", "Firewall"]
        ]
        return {"response": "**Device status**\n\n" + "\n".join(lines)}

    if intent == "AI_MODEL":
        return {"response": """
**Model:** XGBoost classifier + minutes-to-failure regressor

**Inputs (15):** gateway RTT, router loss, DNS RTT, DNS loss, RSSI,
TX rate, jitter, NIC errors/s, traffic + 5-sample rolling/trend.

PC CPU and RAM are **not** features. A healthy laptop with a dying
router still produces a failure prediction.
"""}

    if intent == "SHAP":
        return {"response": """
**SHAP** attributes each alert to a symptom:

- High `router_latency_ms` / `router_packet_loss` → **Router**
- High `nic_errors_per_sec` / low `tx_rate_mbps` → **Switch / AP**
- High `dns_latency_ms` with a healthy gateway → **Firewall / ISP**
"""}

    if intent == "HEALING":
        return {"response": """
Healing is network-side only: flush DNS cache, flag switch ports,
recommend a router check. It does not kill PC processes.
"""}

    if intent == "SLA":
        try:
            sla, up, dn = calculate_sla()
            return {"response": f"**SLA** {sla}%  (up {up} / down {dn})"}
        except Exception:
            return {"response": "SLA not available yet."}

    if intent == "ALERTS":
        try:
            from database import get_alerts
            alerts = get_alerts(limit=5)
            if not alerts:
                return {"response": "No recent device-failure alerts."}
            lines = [f"- {a['timestamp']} | **{a['device']}** | {a['severity']}"
                     for a in alerts]
            return {"response": "**Recent alerts**\n\n" + "\n".join(lines)}
        except Exception:
            return {"response": "Alerts not available yet."}

    if intent == "HELP":
        return {"response": (
            "Ask: network status, which device is failing, "
            "why failure, model accuracy, SLA, recent alerts."
        )}

    return {"response": "Try: `which device is failing`, `why failure`, `help`."}
