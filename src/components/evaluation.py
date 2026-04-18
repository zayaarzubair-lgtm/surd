"""Runs SURD on the four benchmark datasets from the paper and checks results.

The datasets come from Martínez-Sánchez et al. (2024), Table 1:
mediator, confounder, synergistic collider, and redundant collider.
Each has a known dominant causal structure that SURD should detect.
"""

import os
import json
import pandas as pd

from src.pipeline.analysis_pipeline import run_analysis
from src.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")

# Each test defines: which file, which columns, what to look for.
EXPECTED = {
    "Mediator": {
        "file": "mediator.csv",
        "target": "q1",
        "agents": ["q1", "q2", "q3"],
        "expect": "q2 should be the dominant causal agent for q1 — it is the mediator in the chain q3→q2→q1.",
        "check": "q2_causes_q1",
    },
    "Confounder": {
        "file": "confounder.csv",
        "target": "q1",
        "agents": ["q1", "q2", "q3"],
        "expect": "Synergistic causality from q1+q3 should be significant — q3 confounds.",
        "check": "synergy_present",
    },
    "Synergistic collider": {
        "file": "synergistic_collider.csv",
        "target": "q1",
        "agents": ["q1", "q2", "q3"],
        "expect": "Synergy from q2+q3 should dominate — they only cause q1 together.",
        "check": "synergy_dominant",
    },
    "Redundant collider": {
        "file": "redundant_collider.csv",
        "target": "q1",
        "agents": ["q1", "q2", "q3"],
        "expect": "Redundancy should be significant — q2 and q3 carry the same information.",
        "check": "redundancy_present",
    },
}


def run_evaluation(tau: int = 1, nbins: int = 8) -> list[dict]:
    """Run SURD on all benchmark datasets and return results with pass/fail."""
    results = []

    for name, spec in EXPECTED.items():
        path = os.path.join(DATA_DIR, spec["file"])
        if not os.path.exists(path):
            logger.warning("Missing evaluation dataset: %s", path)
            continue

        df = pd.read_csv(path)
        result = run_analysis(
            df,
            target=spec["target"],
            agents=spec["agents"],
            params={"tau": tau, "nbins": nbins, "missing_strategy": "drop"},
        )

        # Check if the expected causal structure is detected.
        passed = _check_result(result, spec["check"])

        entry = {
            "dataset": name,
            "result": result,
            "expected": spec["expect"],
            "check_type": spec["check"],
            "passed": passed,
        }
        results.append(entry)
        logger.info("Eval '%s': check=%s, pass=%s", name, spec["check"], passed)

    # Save to artifacts.
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    out_path = os.path.join(ARTIFACTS_DIR, "evaluation_results.json")
    safe = []
    for r in results:
        safe.append({
            "dataset": r["dataset"],
            "passed": r["passed"],
            "check_type": r["check_type"],
            "total_unique": r["result"]["total_unique"],
            "total_redundant": r["result"]["total_redundant"],
            "total_synergy": r["result"]["total_synergy"],
            "info_leak": r["result"]["info_leak"],
        })
    with open(out_path, "w") as f:
        json.dump(safe, f, indent=2)

    return results


def _check_result(result: dict, check_type: str) -> bool:
    """Verify the SURD result matches what the paper predicts."""
    tu = result["total_unique"]
    tr = result["total_redundant"]
    ts = result["total_synergy"]

    if check_type == "q2_causes_q1":
        # q2 should appear in the biggest unique or redundant component.
        all_components = {}
        all_components.update(result.get("unique", {}))
        all_components.update(result.get("redundant_breakdown", {}))
        if not all_components:
            return False
        top_key = max(all_components, key=all_components.get)
        return "q2" in top_key

    if check_type == "unique_dominant":
        return tu > tr and tu > ts

    if check_type == "synergy_dominant":
        return ts > tu and ts > tr

    if check_type == "synergy_present":
        total = tu + tr + ts
        return ts > 0.05 * total if total > 0 else False

    if check_type == "redundancy_present":
        total = tu + tr + ts
        return tr > 0.05 * total if total > 0 else False

    return False
