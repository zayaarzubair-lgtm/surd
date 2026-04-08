"""Handles exporting results to JSON, CSV, and chart HTML files."""

import json
import csv
import io
import os
from datetime import datetime

from src.logger import get_logger

logger = get_logger(__name__)

ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts"
)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def result_to_json(result: dict) -> str:
    """Convert a result dict to downloadable JSON text."""
    return json.dumps(result, indent=2)


def result_to_csv(result: dict) -> str:
    """Convert key numbers to a small CSV summary string."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["component", "variables", "value_bits"])

    for name, val in result.get("unique", {}).items():
        writer.writerow(["unique", name, f"{val:.6f}"])

    for name, val in result.get("redundant_breakdown", {}).items():
        writer.writerow(["redundant", name, f"{val:.6f}"])

    for name, val in result.get("synergy_breakdown", {}).items():
        writer.writerow(["synergy", name, f"{val:.6f}"])

    writer.writerow(["info_leak", "all", f"{result.get('info_leak', 0):.6f}"])

    return buf.getvalue()


def export_json(result: dict, filename: str | None = None) -> str:
    """Save result dict as JSON file. Returns the file path."""
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"surd_result_{ts}.json"
    path = os.path.join(ARTIFACTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Exported JSON to %s", path)
    return path


def export_chart_html(fig, filename: str | None = None) -> str:
    """Save a Plotly figure as standalone HTML. Returns the path."""
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chart_{ts}.html"
    path = os.path.join(ARTIFACTS_DIR, filename)
    fig.write_html(path, include_plotlyjs="cdn")
    logger.info("Exported chart HTML to %s", path)
    return path
