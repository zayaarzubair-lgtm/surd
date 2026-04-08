"""Main pipeline — the single function any UI calls.

Ties together data cleaning, SURD computation, and explanation.
No UI code lives here. Both the Streamlit and future Dash apps
call run_analysis() and get back a plain dict.
"""

import pandas as pd

from src.components.data_ingestion import validate_columns, dataset_summary
from src.components.data_transformation import handle_missing
from src.components.surd_engine import run_surd
from src.components.explanation import generate_explanation
from src.logger import get_logger

logger = get_logger(__name__)


def run_analysis(
    df: pd.DataFrame,
    target: str,
    agents: list[str],
    params: dict | None = None,
) -> dict:
    """Run the full SURD analysis and return a results dict.

    Args:
        df:      Raw uploaded DataFrame (rows must be in time order).
        target:  Column name for the variable whose future I'm predicting.
        agents:  Column names for variables whose past might cause the target.
        params:  Optional settings dict:
            - tau (int): time lag in rows (default 1).
            - nbins (int): histogram bins per variable (default 10).
            - missing_strategy (str): 'drop' or 'mean' (default 'drop').
            - explanation_mode (str): 'plain' or 'technical' (default 'plain').

    Returns:
        dict with all SURD results, metadata, warnings, and explanation.
    """
    params = params or {}
    tau = params.get("tau", 1)
    nbins = params.get("nbins", 10)
    missing_strategy = params.get("missing_strategy", "drop")
    explanation_mode = params.get("explanation_mode", "plain")

    logger.info("Pipeline started: target=%s, agents=%s, tau=%d", target, agents, tau)

    # Step 1 — Validate the chosen columns.
    warnings = validate_columns(df, agents, target)

    # Step 2 — Keep only the columns I need (deduplicate if target is also an agent).
    cols = list(dict.fromkeys([target] + agents))
    subset = df[cols].copy()

    # Step 3 — Handle missing values.
    subset = handle_missing(subset, strategy=missing_strategy)

    if len(subset) < 50:
        warnings.append("Very few rows after cleaning — SURD needs enough data for reliable probability estimates.")

    # Step 4 — Run the SURD decomposition.
    surd_result = run_surd(subset, target=target, agents=agents,
                           tau=tau, nbins=nbins)

    # Step 5 — Add metadata about the dataset.
    summary = dataset_summary(subset)
    surd_result["meta"]["n_rows_original"] = summary["rows"]
    surd_result["meta"]["n_cols"] = summary["columns"]
    surd_result["meta"]["missing_strategy"] = missing_strategy

    # Step 6 — Attach warnings.
    surd_result["warnings"] = warnings

    # Step 7 — Generate explanation text.
    surd_result["explanation"] = generate_explanation(surd_result, mode=explanation_mode)

    logger.info("Pipeline finished. Method: %s", surd_result.get("method"))
    return surd_result
