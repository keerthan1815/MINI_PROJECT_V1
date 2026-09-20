import threading
import platform

running = False

IS_WINDOWS = platform.system().lower() == "windows"

if IS_WINDOWS:
    import winsound


def _alarm():
    while running:
        if IS_WINDOWS:
            winsound.Beep(4000, 1500)
        else:
            import time
            time.sleep(1.5)


def start_buzzer():
    global running
    if not running:
        running = True
        threading.Thread(target=_alarm, daemon=True).start()


def stop_buzzer():
    global running
    running = False
