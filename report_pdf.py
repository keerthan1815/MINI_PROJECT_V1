"""
PDF report for explainable network-device failure prediction.
"""

import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from database import get_alerts, get_failure_count_by_device
from sla_monitor import calculate_sla
from system_status import get_overall_status, status


def generate_pdf_report(report_type="daily"):
    filename = f"{report_type}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4,
                            rightMargin=2 * cm, leftMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    story = []
    title_style = ParagraphStyle(
        "title", parent=styles["Title"], fontSize=16, alignment=TA_CENTER)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=13)
    normal = styles["Normal"]

    story.append(Paragraph(
        "Explainable AI Prediction of Failure for Network Devices", title_style))
    story.append(Paragraph(f"{report_type.upper()} REPORT — XGBoost + SHAP", h1))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y  %H:%M:%S')}", normal))
    story.append(HRFlowable(width="100%", thickness=1.5,
                             color=colors.HexColor("#0f3460")))
    story.append(Spacer(1, 0.3 * cm))

    sla, up, dn = calculate_sla()
    story.append(Paragraph("1. Scope", h1))
    story.append(Paragraph(
        "This report covers <b>network device</b> failure (router, switch, "
        "WiFi AP, firewall/ISP path), inferred from client-side network "
        "symptoms. Host CPU and RAM are not used.",
        normal))
    story.append(Spacer(1, 0.25 * cm))

    story.append(Paragraph("2. Executive summary", h1))
    story.append(_table([
        ["Metric", "Value"],
        ["Overall path health", get_overall_status()],
        ["SLA", f"{sla}%"],
        ["Uptime / downtime samples", f"{up} / {dn}"],
        ["Models", "XGBoost classifier + minutes-to-failure regressor"],
        ["Features", "15 network-device symptoms (no PC metrics)"],
    ]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("3. Device status", h1))
    rows = [["Device", "Status"]]
    label = {"healthy": "Healthy", "warning": "Warning", "failure": "Failure"}
    for device in ["Router", "Switch", "Firewall"]:
        rows.append([device, label.get(status.get(device, "healthy"), "Unknown")])
    story.append(_table(rows))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("4. Alerts by observed device", h1))
    counts = get_failure_count_by_device()
    if counts:
        fb = [["Device", "Alerts"]] + [[d, str(c)] for d, c in counts.items()]
        story.append(_table(fb))
    else:
        story.append(Paragraph("No failure alerts recorded yet.", normal))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("5. Recent alerts", h1))
    alerts = get_alerts(limit=10)
    if alerts:
        al = [["Time", "Device", "Severity", "Lead time"]]
        for a in alerts:
            al.append([
                str(a.get("timestamp", ""))[:19],
                a.get("device", ""),
                a.get("severity", ""),
                a.get("lead_time", ""),
            ])
        story.append(_table(al))
    else:
        story.append(Paragraph("No alerts yet.", normal))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("6. How failure is known", h1))
    story.append(Paragraph(
        "A router is predicted to fail when gateway RTT, packet loss, or "
        "jitter rise. A switch is implicated by NIC/PHY error rate and link "
        "rate collapse. A firewall/ISP path is implicated when DNS/WAN RTT "
        "rises while the local gateway stays healthy. SHAP attributes each "
        "alert to those symptoms.",
        normal))
    story.append(Spacer(1, 0.2 * cm))

    if os.path.exists("shap_summary_bar.png"):
        story.append(Image("shap_summary_bar.png", width=14 * cm, height=7 * cm))
    if os.path.exists("confusion_matrix.png"):
        story.append(Spacer(1, 0.2 * cm))
        story.append(Image("confusion_matrix.png", width=11 * cm, height=8 * cm))

    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#0f3460")))
    story.append(Paragraph(
        "XGBoost + SHAP — network device failure prediction",
        ParagraphStyle("f", parent=normal, fontSize=9,
                       textColor=colors.grey, alignment=TA_CENTER)))
    doc.build(story)
    return filename


def _table(data):
    t = Table(data, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#e8f0fe")]),
    ]))
    return t
