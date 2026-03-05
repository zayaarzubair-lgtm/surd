"""Main pipeline — the single function any UI calls.

Ties together cleaning, PID computation, and explanation.
No UI code lives here.
"""

import pandas as pd

from src.components.data_ingestion import validate_columns, dataset_summary
from src.components.data_transformation import handle_missing, discretise
from src.components.surd_engine import compute_surd
from src.components.explanation import generate_explanation
from src.logger import get_logger

logger = get_logger(__name__)


def run_analysis(
    df: pd.DataFrame,
    sources: list[str],
    target: str,
    params: dict | None = None,
) -> dict:
    """Run the full PID analysis and return a results dict.

    Args:
        df: Raw uploaded DataFrame.
        sources: Column names to use as X variables (2–6).
        target: Column name to use as Y.
        params: Optional settings dict:
            - bins (int): discretisation bins (default 5).
            - missing_strategy (str): 'drop' or 'mean'.
            - explanation_mode (str): 'plain' or 'technical'.

    Returns:
        dict with: unique, redundant, synergy, pairwise_synergy,
                   pairwise_redundancy, method, meta, warnings,
                   explanation.
    """
    params = params or {}
    bins = params.get("bins", 5)
    missing_strategy = params.get("missing_strategy", "drop")
    explanation_mode = params.get("explanation_mode", "plain")

    logger.info("Pipeline started: sources=%s, target=%s", sources, target)

    # Step 1 — Validate.
    warnings = validate_columns(df, sources, target)

    # Step 2 — Subset.
    cols = sources + [target]
    subset = df[cols].copy()

    # Step 3 — Handle missing values.
    subset = handle_missing(subset, strategy=missing_strategy)

    if len(subset) < 10:
        warnings.append("Very few rows after cleaning — results may be unreliable.")

    # Step 4 — Discretise continuous columns into bins.
    df_binned = discretise(subset, columns=cols, bins=bins)

    # Step 5 — Compute PID.
    surd = compute_surd(df_binned, sources, target)

    # Step 6 — Metadata.
    summary = dataset_summary(subset)
    meta = {
        "n_rows": summary["rows"],
        "n_cols": summary["columns"],
        "sources": sources,
        "target": target,
        "params": {"bins": bins, "missing_strategy": missing_strategy},
    }

    # Step 7 — Assemble.
    result = {**surd, "meta": meta, "warnings": warnings}
    result["explanation"] = generate_explanation(result, mode=explanation_mode)

    logger.info("Pipeline finished. Method: %s", result.get("method"))
    return result
