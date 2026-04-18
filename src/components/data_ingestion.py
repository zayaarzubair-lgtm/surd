"""Loads CSV data and checks that it looks reasonable for SURD analysis."""

import pandas as pd

from src.logger import get_logger
from src.exception import DataValidationError

logger = get_logger(__name__)


def load_csv(file) -> pd.DataFrame:
    """Read a CSV file into a DataFrame.

    Accepts either a file path string or a Streamlit uploaded file object.
    """
    try:
        df = pd.read_csv(file)
        logger.info("Loaded CSV with shape %s", df.shape)
        return df
    except Exception as exc:
        raise DataValidationError(f"Could not read CSV: {exc}") from exc


def validate_columns(df: pd.DataFrame, sources: list[str], target: str) -> list[str]:
    """Check that the chosen columns exist and are numeric.

    Returns a list of warning strings (empty if everything is fine).
    """
    warnings = []
    all_cols = sources + [target]

    for col in all_cols:
        if col not in df.columns:
            raise DataValidationError(f"Column '{col}' not found in the dataset.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            warnings.append(f"'{col}' is not numeric — results may be unreliable.")

    # In SURD, the target's own past can be an agent (self-causation).
    # Only check that columns are unique within sources.
    if len(set(sources)) != len(sources):
        raise DataValidationError("Duplicate columns in source list.")

    if len(df) < 50:
        warnings.append("Dataset has very few rows — SURD needs enough data to estimate probabilities reliably.")

    logger.info("Column validation passed with %d warning(s)", len(warnings))
    return warnings


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return quick stats about the dataset."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "numeric_columns": list(df.select_dtypes("number").columns),
    }
