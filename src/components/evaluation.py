"""Runs the analysis on synthetic datasets and checks the results make sense."""

import os
import json
import pandas as pd

from src.pipeline.analysis_pipeline import run_analysis
from src.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")

# What we expect from each test dataset.
EXPECTED = {
    "XOR": {
        "file": "synthetic_xor.csv",
        "sources": ["X1", "X2"],
        "target": "Y",
        "expect": "High synergy — neither source alone predicts the target.",
        "dominant": "synergy",
    },
    "Redundancy": {
        "file": "synthetic_redundancy.csv",
        "sources": ["X1", "X2"],
        "target": "Y",
        "expect": "High redundancy — both sources carry the same information.",
        "dominant": "redundant",
    },
    "Unique": {
        "file": "synthetic_unique.csv",
        "sources": ["X1", "X2"],
        "target": "Y",
        "expect": "High unique for X1 — only X1 predicts the target.",
        "dominant": "unique",
    },
}


def run_evaluation(bins: int = 5) -> list[dict]:
    """Run PID on all synthetic datasets and return results with interpretations."""
    results = []

    for name, spec in EXPECTED.items():
        path = os.path.join(DATA_DIR, spec["file"])
        if not os.path.exists(path):
            logger.warning("Missing evaluation dataset: %s", path)
            continue

        df = pd.read_csv(path)
        result = run_analysis(
            df,
            sources=spec["sources"],
            target=spec["target"],
            params={"bins": bins, "missing_strategy": "drop"},
        )

        # Check if the dominant component is actually the largest.
        total_unique = sum(result["unique"].values())
        components = {
            "unique": total_unique,
            "redundant": result["redundant"],
            "synergy": result["synergy"],
        }
        actual_dominant = max(components, key=components.get)
        passed = actual_dominant == spec["dominant"]

        entry = {
            "dataset": name,
            "result": result,
            "expected": spec["expect"],
            "dominant_expected": spec["dominant"],
            "dominant_actual": actual_dominant,
            "passed": passed,
        }
        results.append(entry)
        logger.info("Eval '%s': expected=%s, got=%s, pass=%s",
                     name, spec["dominant"], actual_dominant, passed)

    # Save to artifacts.
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    out_path = os.path.join(ARTIFACTS_DIR, "evaluation_results.json")
    safe = []
    for r in results:
        safe.append({
            "dataset": r["dataset"],
            "passed": r["passed"],
            "dominant_expected": r["dominant_expected"],
            "dominant_actual": r["dominant_actual"],
            "unique": r["result"]["unique"],
            "redundant": r["result"]["redundant"],
            "synergy": r["result"]["synergy"],
        })
    with open(out_path, "w") as f:
        json.dump(safe, f, indent=2)

    return results
