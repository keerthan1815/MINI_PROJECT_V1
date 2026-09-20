"""
networktopology_map.py - Interactive Plotly network topology.

Shows Router → Switch → Firewall → PC1/PC2/Server
Color coded by live device health from system_status.py

Called by live_dashboard.py on the Network Topology page.
"""

import plotly.graph_objects as go
from system_status import status


# Fixed positions for each node on the canvas
NODE_POSITIONS = {
    "Internet":  (0,   2),
    "Router":    (2,   2),
    "Switch":    (4,   2),
    "Firewall":  (4,   0),
    "Server":    (6,   3),
    "PC1":       (6,   2),
    "PC2":       (6,   1),
}

# Connections (edges)
EDGES = [
    ("Internet",  "Router"),
    ("Router",    "Switch"),
    ("Switch",    "Firewall"),
    ("Switch",    "Server"),
    ("Switch",    "PC1"),
    ("Switch",    "PC2"),
]

NODE_ICONS = {
    "Internet":  "📡",
    "Router":    "🔀",
    "Switch":    "🔌",
    "Firewall":  "🛡️",
    "Server":    "🖥️",
    "PC1":       "💻",
    "PC2":       "💻",
}

STATUS_COLOR = {
    "healthy": "#00cc44",
    "warning": "#ffaa00",
    "failure": "#ff3333",
}


def create_topology_figure():
    """Returns a Plotly Figure showing the live network topology."""

    edge_x, edge_y = [], []
    for src, dst in EDGES:
        x0, y0 = NODE_POSITIONS[src]
        x1, y1 = NODE_POSITIONS[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=2, color="#888"),
        hoverinfo="none"
    )

    node_x, node_y, node_colors, node_labels, node_hover = [], [], [], [], []

    for node, (x, y) in NODE_POSITIONS.items():
        node_x.append(x)
        node_y.append(y)
        s = status.get(node, "healthy")
        node_colors.append(STATUS_COLOR.get(s, "#aaa"))
        icon = NODE_ICONS.get(node, "")
        node_labels.append(f"{icon} {node}")
        node_hover.append(f"{node}: {s.upper()}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=40,
            color=node_colors,
            line=dict(width=2, color="#333")
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="🌐 Live Network Topology",
            titlefont_size=18,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=420,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white")
        )
    )
    return fig


def show_topology():
    """
    Called from live_dashboard.py.
    Renders the topology chart and a status legend.
    """
    import streamlit as st
    fig = create_topology_figure()
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Device Status")
    cols = st.columns(len(NODE_POSITIONS))
    for i, (node, _) in enumerate(NODE_POSITIONS.items()):
        s = status.get(node, "healthy")
        icon = {"healthy": "🟢", "warning": "🟡", "failure": "🔴"}.get(s, "⚪")
        cols[i].markdown(f"**{node}**<br>{icon} {s.upper()}", unsafe_allow_html=True)
