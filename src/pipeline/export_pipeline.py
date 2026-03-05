"""Handles exporting results to files (JSON for now)."""

import json
import os
from datetime import datetime

from src.logger import get_logger

logger = get_logger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def export_json(result: dict, filename: str | None = None) -> str:
    """Save a result dict as a JSON file in the artifacts folder.

    Returns the file path that was written.
    """
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"surd_result_{ts}.json"

    path = os.path.join(ARTIFACTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Exported results to %s", path)
    return path
