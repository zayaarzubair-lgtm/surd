"""All Plotly chart-building functions for the SURD dashboard."""

import plotly.graph_objects as go

COLOURS = {
    "unique": "#d62828",
    "redundant": "#003049",
    "synergy": "#f77f00",
    "leak": "#888888",
    "bg": "rgba(0,0,0,0)",
}


def make_overview_fig(result: dict) -> go.Figure:
    """Stacked bar showing unique, redundant, and synergy totals."""
    labels = ["Unique", "Redundant", "Synergy"]
    values = [result["total_unique"], result["total_redundant"], result["total_synergy"]]
    colours = [COLOURS["unique"], COLOURS["redundant"], COLOURS["synergy"]]

    fig = go.Figure(data=[go.Bar(
        x=labels, y=values,
        marker_color=colours,
        text=[f"{v:.4f}" for v in values],
        textposition="auto",
    )])
    fig.update_layout(
        title="SURD Causal Decomposition — Totals",
        yaxis_title="Information (bits)",
        template="plotly_white",
        height=400,
    )
    return fig


def make_breakdown_fig(result: dict) -> go.Figure:
    """Bar chart showing every individual SURD component."""
    labels = []
    values = []
    colours = []

    # Unique contributions.
    for name, val in result.get("unique", {}).items():
        labels.append(f"U({name})")
        values.append(val)
        colours.append(COLOURS["unique"])

    # Redundant contributions.
    for name, val in result.get("redundant_breakdown", {}).items():
        labels.append(f"R({name.replace('|', ',')})")
        values.append(val)
        colours.append(COLOURS["redundant"])

    # Synergistic contributions.
    for name, val in result.get("synergy_breakdown", {}).items():
        labels.append(f"S({name.replace('|', ',')})")
        values.append(val)
        colours.append(COLOURS["synergy"])

    # Normalise so bars sum to 1 (matching the paper's convention).
    total = sum(values) if sum(values) > 0 else 1
    norm_values = [v / total for v in values]

    fig = go.Figure(data=[go.Bar(
        x=labels, y=norm_values,
        marker_color=colours,
        text=[f"{v:.3f}" for v in norm_values],
        textposition="auto",
    )])
    fig.update_layout(
        title="SURD Decomposition — All Components (normalised)",
        yaxis_title="Fraction of total causal information",
        template="plotly_white",
        height=420,
    )
    return fig


def make_leak_fig(result: dict) -> go.Figure:
    """Simple bar showing the information leak."""
    leak = result.get("info_leak", 0)

    fig = go.Figure(data=[go.Bar(
        x=["Information leak"],
        y=[leak],
        marker_color=COLOURS["leak"],
        text=[f"{leak:.2%}"],
        textposition="auto",
        width=[0.4],
    )])
    fig.update_layout(
        title="Causality Leak — Unexplained by Observed Variables",
        yaxis_title="Fraction of target entropy",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        height=300,
    )
    return fig


def make_pie_fig(result: dict) -> go.Figure:
    """Donut chart showing the share of unique, redundant, and synergy."""
    labels = ["Unique (total)", "Redundant (total)", "Synergy (total)"]
    values = [result["total_unique"], result["total_redundant"], result["total_synergy"]]
    colours = [COLOURS["unique"], COLOURS["redundant"], COLOURS["synergy"]]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colours),
        textinfo="label+percent", hole=0.35,
    )])
    fig.update_layout(
        title="Causal information share",
        template="plotly_white", height=360,
    )
    return fig


def make_lag_sweep_fig(sweep_results: list[dict]) -> go.Figure:
    """Line chart showing how U/R/S change across tau values."""
    taus = [r["tau"] for r in sweep_results]
    unique = [r["total_unique"] for r in sweep_results]
    redundant = [r["total_redundant"] for r in sweep_results]
    synergy = [r["total_synergy"] for r in sweep_results]
    leak = [r["info_leak"] for r in sweep_results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=taus, y=unique, name="Unique",
        mode="lines+markers", marker_color=COLOURS["unique"]))
    fig.add_trace(go.Scatter(
        x=taus, y=redundant, name="Redundant",
        mode="lines+markers", marker_color=COLOURS["redundant"]))
    fig.add_trace(go.Scatter(
        x=taus, y=synergy, name="Synergy",
        mode="lines+markers", marker_color=COLOURS["synergy"]))
    fig.add_trace(go.Scatter(
        x=taus, y=leak, name="Info leak",
        mode="lines+markers", marker_color=COLOURS["leak"],
        line=dict(dash="dash")))
    fig.update_layout(
        title="Lag sweep — causal decomposition vs time lag (tau)",
        xaxis_title="Time lag (tau)",
        yaxis_title="Information (bits) / Leak fraction",
        template="plotly_white",
        height=450,
    )
    return fig


def make_heatmap_fig(result: dict, field: str) -> go.Figure:
    """Heatmap of pairwise values from redundant or synergy breakdown."""
    breakdown = result.get(field, {})
    if not breakdown:
        fig = go.Figure()
        fig.update_layout(title="No pairwise data available")
        return fig

    # Collect all agent names from the breakdown keys.
    agents = []
    for key in breakdown:
        for name in key.split("|"):
            if name not in agents:
                agents.append(name)

    n = len(agents)
    matrix = [[0.0] * n for _ in range(n)]

    for key, val in breakdown.items():
        parts = key.split("|")
        if len(parts) == 2:
            i = agents.index(parts[0])
            j = agents.index(parts[1])
            matrix[i][j] = val
            matrix[j][i] = val

    label = "Redundancy" if "redundant" in field else "Synergy"
    colour = "Blues" if "redundant" in field else "Oranges"

    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=agents, y=agents,
        colorscale=colour,
        text=[[f"{v:.4f}" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size": 11},
    ))
    fig.update_layout(
        title=f"Pairwise {label.lower()} heatmap",
        template="plotly_white",
        height=400,
    )
    return fig


def make_evaluation_fig(eval_results: list[dict]) -> go.Figure:
    """Grouped bar chart comparing SURD across evaluation datasets."""
    names = [r["dataset"] for r in eval_results]
    total_unique = [r["result"]["total_unique"] for r in eval_results]
    redundant = [r["result"]["total_redundant"] for r in eval_results]
    synergy = [r["result"]["total_synergy"] for r in eval_results]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Unique", x=names, y=total_unique,
                         marker_color=COLOURS["unique"]))
    fig.add_trace(go.Bar(name="Redundant", x=names, y=redundant,
                         marker_color=COLOURS["redundant"]))
    fig.add_trace(go.Bar(name="Synergy", x=names, y=synergy,
                         marker_color=COLOURS["synergy"]))
    fig.update_layout(
        barmode="group",
        title="Evaluation — Expected vs observed causal decomposition",
        yaxis_title="Information (bits)",
        template="plotly_white", height=400,
    )
    return fig


def make_significance_fig(sig_result: dict) -> go.Figure:
    """Histograms of null distributions with the observed value marked."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Unique", "Redundant", "Synergy", "Information leak"),
    )

    components = [
        ("unique", 1, 1, COLOURS["unique"]),
        ("redundant", 1, 2, COLOURS["redundant"]),
        ("synergy", 2, 1, COLOURS["synergy"]),
        ("leak", 2, 2, COLOURS["leak"]),
    ]

    for name, row, col, colour in components:
        null_dist = sig_result["null_distributions"][name]
        observed = sig_result["observed"][name]
        p_val = sig_result["p_values"][name]

        # Histogram of null distribution.
        fig.add_trace(
            go.Histogram(x=null_dist, marker_color=colour, opacity=0.6,
                         showlegend=False, nbinsx=20),
            row=row, col=col,
        )
        # Vertical line at observed value.
        fig.add_vline(x=observed, line_dash="dash", line_color="black",
                      line_width=2, annotation_text=f"observed (p={p_val:.3f})",
                      annotation_position="top right",
                      row=row, col=col)

    fig.update_layout(
        title="Permutation test — null distributions vs observed values",
        template="plotly_white",
        height=600, showlegend=False,
    )
    return fig
