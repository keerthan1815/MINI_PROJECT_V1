import smtplib
from email.mime.text import MIMEText

SENDER_EMAIL = "haddadisanad814@gmail.com"
APP_PASSWORD = "iaqx cbqb ulnb zhle"
RECEIVER_EMAIL = "haddadisanad814@gmail.com"


def send_alert(device, prediction, reason, metrics=None):
    metrics = metrics or {}
    try:
        message = f"""
NETWORK DEVICE FAILURE ALERT

Device / path : {device}
Prediction    : {prediction}
Reason        : {reason}

Gateway RTT   : {metrics.get('router_latency_ms')} ms
Router loss   : {metrics.get('router_packet_loss')} %
DNS RTT       : {metrics.get('dns_latency_ms')} ms
DNS loss      : {metrics.get('dns_packet_loss')} %
RSSI          : {metrics.get('rssi_dbm')} dBm
Jitter        : {metrics.get('jitter_ms')} ms
NIC errors/s  : {metrics.get('nic_errors_per_sec')}
Traffic       : {metrics.get('traffic_kbps')} KB/s
"""
        msg = MIMEText(message)
        msg["Subject"] = "Network device failure predicted (XGBoost)"
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=10)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print("Email alert sent")
    except Exception as e:
        print("Email failed:", e)
