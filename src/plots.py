"""All Plotly chart-building functions live here."""

import plotly.graph_objects as go

COLOURS = {
    "unique": "#3b82f6",
    "redundant": "#f59e0b",
    "synergy": "#10b981",
    "bg": "rgba(0,0,0,0)",
}


def make_overview_fig(result: dict) -> go.Figure:
    """Stacked bar chart showing Unique / Redundant / Synergy per source."""
    sources = list(result["unique"].keys())
    unique_vals = [result["unique"][s] for s in sources]
    n = len(sources)

    # Spread redundant and synergy evenly across bars.
    red_each = result["redundant"] / n if n else 0
    syn_each = result["synergy"] / n if n else 0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Unique", x=sources, y=unique_vals,
        marker_color=COLOURS["unique"],
    ))
    fig.add_trace(go.Bar(
        name="Redundant", x=sources, y=[red_each] * n,
        marker_color=COLOURS["redundant"],
    ))
    fig.add_trace(go.Bar(
        name="Synergy", x=sources, y=[syn_each] * n,
        marker_color=COLOURS["synergy"],
    ))
    fig.update_layout(
        barmode="stack",
        title="PID Decomposition — Overview",
        xaxis_title="Source variable",
        yaxis_title="Information (bits)",
        template="plotly_white",
        plot_bgcolor=COLOURS["bg"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
        height=420,
    )
    return fig


def make_pie_fig(result: dict) -> go.Figure:
    """Pie chart showing the share of unique, redundant, and synergy."""
    total_unique = sum(result["unique"].values())
    labels = ["Unique (total)", "Redundant", "Synergy"]
    values = [total_unique, result["redundant"], result["synergy"]]
    colours = [COLOURS["unique"], COLOURS["redundant"], COLOURS["synergy"]]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colours),
        textinfo="label+percent", hole=0.35,
    )])
    fig.update_layout(
        title="Information share",
        template="plotly_white", height=360,
    )
    return fig


def make_heatmap_fig(result: dict, field: str = "pairwise_synergy") -> go.Figure:
    """Heatmap of pairwise synergy (or redundancy) between sources."""
    pw = result.get(field)
    if not pw:
        return _empty_figure("Pairwise data requires more than 2 sources.")

    sources = list(result["unique"].keys())
    n = len(sources)
    z = [[0.0] * n for _ in range(n)]

    for key_str, val in pw.items():
        a, b = key_str.split("|")
        i, j = sources.index(a), sources.index(b)
        z[i][j] = val
        z[j][i] = val

    title = "Pairwise synergy" if "synergy" in field else "Pairwise redundancy"
    fig = go.Figure(data=go.Heatmap(
        z=z, x=sources, y=sources,
        colorscale="Teal" if "synergy" in field else "YlOrRd",
        text=[[f"{v:.4f}" for v in row] for row in z],
        texttemplate="%{text}",
        hovertemplate="(%{x}, %{y}): %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, template="plotly_white", height=420,
    )
    return fig


def make_evaluation_fig(eval_results: list[dict]) -> go.Figure:
    """Grouped bar chart comparing PID across evaluation datasets."""
    names = [r["dataset"] for r in eval_results]
    total_unique = [sum(r["result"]["unique"].values()) for r in eval_results]
    redundant = [r["result"]["redundant"] for r in eval_results]
    synergy = [r["result"]["synergy"] for r in eval_results]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Unique", x=names, y=total_unique,
                         marker_color=COLOURS["unique"]))
    fig.add_trace(go.Bar(name="Redundant", x=names, y=redundant,
                         marker_color=COLOURS["redundant"]))
    fig.add_trace(go.Bar(name="Synergy", x=names, y=synergy,
                         marker_color=COLOURS["synergy"]))
    fig.update_layout(
        barmode="group",
        title="Evaluation — Expected vs observed decomposition",
        yaxis_title="Information (bits)",
        template="plotly_white", height=400,
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    """Blank chart with a centred message."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16))
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        template="plotly_white", height=300,
    )
    return fig
