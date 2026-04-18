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
