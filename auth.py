"""
Explainable AI-Based Predictive Failure Detection for
Network Devices Using XGBoost and SHAP

System layer: Dashboard Layer (role-based access to the monitoring UI).

Algorithms / techniques:
    - Local username/password lookup
    - Role gating (admin / engineer / viewer)

Inputs:
    - Username and password from the Streamlit login form

Outputs:
    - Role string on success, or None if credentials are invalid

Research reference:
    Alghamdi et al. (2025), IJISRT,
    "Artificial Intelligence for Predictive Failures of Network Devices"
"""

users = {
    "admin": {
        "password": "admin123",
        "role": "admin"
    },
    "engineer": {
        "password": "eng123",
        "role": "engineer"
    },
    "viewer": {
        "password": "view123",
        "role": "viewer"
    }
}


# Admit an operator to the failure-prediction dashboard and return their role.
def login(username, password):
    if username in users:
        if users[username]["password"] == password:
            return users[username]["role"]
    return None
