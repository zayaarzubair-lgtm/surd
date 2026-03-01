

import json
import plotly.graph_objects as go

# -- Colour palette used across all charts --
COLOURS = {
    "unique": "#3b82f6",
    "redundant": "#f59e0b",
    "synergy": "#10b981",
    "bg": "rgba(0,0,0,0)",
}


def make_overview_fig(result: dict) -> go.Figure:
    """Build a stacked bar chart showing Unique / Redundant / Synergy."""
    sources = list(result["unique"].keys())
    unique_vals = [result["unique"][s] for s in sources]
    n = len(sources)

    # Spread redundant and synergy evenly across sources for the stacked view.
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
        title="SURD Decomposition — Overview",
        xaxis_title="Source variable",
        yaxis_title="Information (bits)",
        template="plotly_white",
        plot_bgcolor=COLOURS["bg"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=420,
    )
    return fig


def make_heatmap_fig(result: dict) -> go.Figure:
    """Build a heatmap of pairwise synergy between sources."""
    pw = result.get("pairwise_synergy")
    if not pw:
        return _empty_figure("Pairwise synergy requires more than 2 sources.")

    sources = list(result["unique"].keys())
    n = len(sources)
    z = [[0.0] * n for _ in range(n)]

    for key_str, val in pw.items():
        a, b = key_str.split("|")
        i, j = sources.index(a), sources.index(b)
        z[i][j] = val
        z[j][i] = val

    fig = go.Figure(data=go.Heatmap(
        z=z, x=sources, y=sources,
        colorscale="Teal",
        text=[[f"{v:.3f}" for v in row] for row in z],
        texttemplate="%{text}",
        hovertemplate="(%{x}, %{y}): %{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title="Pairwise Synergy Heatmap",
        template="plotly_white",
        height=420,
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    """Return a blank chart with an annotation message."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16))
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        template="plotly_white", height=300,
    )
    return fig


def result_to_json(result: dict) -> str:
    """Convert an analysis result dict to downloadable JSON text."""
    safe = {**result}
    # pairwise keys are "A|B" strings already, so it serialises fine.
    return json.dumps(safe, indent=2)
