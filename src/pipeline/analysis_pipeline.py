"""Main pipeline that ties together data cleaning, SURD, and explanation.

Any UI (Streamlit, Dash, etc.) just calls run_analysis() and gets
back a plain dict — no UI code lives here.
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
    """Run the full SURD analysis and return a results dict.

    Args:
        df: Raw uploaded DataFrame.
        sources: Column names to use as X variables.
        target: Column name to use as Y.
        params: Optional settings dict with keys:
            - bins (int): number of bins for discretisation (default 5).
            - missing_strategy (str): 'drop' or 'mean' (default 'drop').

    Returns:
        dict with keys: unique, redundant, synergy, pairwise_synergy,
                        meta, warnings, explanation.
    """
    params = params or {}
    bins = params.get("bins", 5)
    missing_strategy = params.get("missing_strategy", "drop")

    logger.info("Pipeline started: sources=%s, target=%s", sources, target)

    # Step 1 — Validate chosen columns.
    warnings = validate_columns(df, sources, target)

    # Step 2 — Keep only the columns we need.
    cols = sources + [target]
    subset = df[cols].copy()

    # Step 3 — Handle missing values.
    subset = handle_missing(subset, strategy=missing_strategy)

    if len(subset) < 10:
        warnings.append("Very few rows left after cleaning — results may not be meaningful.")

    # Step 4 — Discretise.
    df_binned = discretise(subset, columns=cols, bins=bins)

    # Step 5 — Compute SURD.
    surd = compute_surd(df_binned, sources, target)

    # Step 6 — Build metadata.
    summary = dataset_summary(subset)
    meta = {
        "n_rows": summary["rows"],
        "n_cols": summary["columns"],
        "sources": sources,
        "target": target,
        "params": {"bins": bins, "missing_strategy": missing_strategy},
    }

    # Step 7 — Assemble full result.
    result = {**surd, "meta": meta, "warnings": warnings}
    result["explanation"] = generate_explanation(result)

    logger.info("Pipeline finished successfully.")
    return result
