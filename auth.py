# auth.py

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

def login(username, password):
    if username in users:
        if users[username]["password"] == password:
            return users[username]["role"]
    return None