import csv
import os

MEMORY_FILE = "healing_memory.csv"


def init_memory():
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["issue", "action", "success"])


def save_memory(issue, action, success):
    with open(MEMORY_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([issue, action, success])


def get_best_action(issue):

    if not os.path.exists(MEMORY_FILE):
        return None

    best_action = None
    best_score = 0

    with open(MEMORY_FILE, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["issue"] == issue and row["success"] == "1":
                return row["action"]

    return best_action