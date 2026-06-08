"""Loads CSV data and checks that it looks reasonable"""

import pandas as pd

from src.logger import get_logger
from src.exception import DataValidationError

logger = get_logger(__name__)


def load_csv(file) -> pd.DataFrame:
    """Read a CSV file-like object into a DataFrame

    Args:
        file: An uploaded file (Streamlit UploadedFile) or a file path string.

    Returns:
        A pandas DataFrame
    """
    try:
        df = pd.read_csv(file)
        logger.info("Loaded CSV with shape %s", df.shape)
        return df
    except Exception as exc:
        raise DataValidationError(f"Could not read CSV: {exc}") from exc


def validate_columns(df: pd.DataFrame, sources: list[str], target: str) -> list[str]:
    """Check that the chosen columns exist and are numeric

    Returns a list of warning strings (empty if everything is fine)
    """
    warnings = []
    all_cols = sources + [target]

    for col in all_cols:
        if col not in df.columns:
            raise DataValidationError(f"Column '{col}' not found in the dataset.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            warnings.append(f"'{col}' is not numeric — results may be unreliable.")

    if target in sources:
        raise DataValidationError("Target column must not also be a source.")

    logger.info("Column validation passed with %d warning(s)", len(warnings))
    return warnings


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return quick stats about the dataset"""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "numeric_columns": list(df.select_dtypes("number").columns),
    }
