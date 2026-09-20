"""
run_project.py — Single execution entry point for reviewers and operators.

Provides an interactive CLI launcher for:
  Step 1: python generate_training_data.py (Generate 24k synthetic samples)
  Step 2: python train_model.py            (Train XGBoost models & generate SHAP artifacts)
  Step 3: streamlit run live_dashboard.py  (Launch real-time web monitoring UI)
"""

import subprocess
import sys


def print_banner():
    print("=" * 70)
    print(" Explainable AI-Based Predictive Failure Detection for Network Devices")
    print(" Single Entry Point Runner")
    print("=" * 70)
    print("Available Steps:")
    print("  [1] Generate Training Data    -> python generate_training_data.py")
    print("  [2] Train ML Models & SHAP    -> python train_model.py")
    print("  [3] Launch Streamlit Dashboard -> streamlit run live_dashboard.py")
    print("  [4] Run All Steps (1 -> 2 -> 3)")
    print("  [0] Exit")
    print("=" * 70)


def execute_command(cmd: list):
    """Run a subprocess command with live output streaming."""
    print(f"\n>> Executing: {' '.join(cmd)}\n" + "-" * 60)
    try:
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"\n[ERROR] Command exited with code {proc.returncode}")
            return False
        return True
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
        return False
    except Exception as e:
        print(f"\n[ERROR] Failed to launch command: {e}")
        return False


def main():
    print_banner()

    # If an argument is provided via command line, use it directly
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
    else:
        try:
            choice = input("Enter choice [1-4, or 0 to exit]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return

    py = sys.executable

    if choice == "1":
        execute_command([py, "generate_training_data.py"])
    elif choice == "2":
        execute_command([py, "train_model.py"])
    elif choice == "3":
        execute_command([py, "-m", "streamlit", "run", "live_dashboard.py"])
    elif choice in ("4", "all"):
        ok1 = execute_command([py, "generate_training_data.py"])
        if ok1:
            ok2 = execute_command([py, "train_model.py"])
            if ok2:
                execute_command([py, "-m", "streamlit", "run", "live_dashboard.py"])
    elif choice in ("0", "exit", "q"):
        print("Exiting.")
    else:
        print(f"Unknown option '{choice}'. Please select 1, 2, 3, 4, or 0.")


if __name__ == "__main__":
    main()
