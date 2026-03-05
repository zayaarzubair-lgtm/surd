"""Handles exporting results to JSON, CSV, and chart HTML files."""

import json
import os
from datetime import datetime

import plotly.graph_objects as go

from src.logger import get_logger

logger = get_logger(__name__)

ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts"
)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def export_json(result: dict, filename: str | None = None) -> str:
    """Save result dict as JSON. Returns the file path."""
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"surd_result_{ts}.json"
    path = os.path.join(ARTIFACTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Exported JSON to %s", path)
    return path


def export_chart_html(fig: go.Figure, filename: str | None = None) -> str:
    """Save a Plotly figure as a standalone HTML file. Returns the path."""
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chart_{ts}.html"
    path = os.path.join(ARTIFACTS_DIR, filename)
    fig.write_html(path, include_plotlyjs="cdn")
    logger.info("Exported chart HTML to %s", path)
    return path
