"""Handles missing values and turns continuous numbers into bins"""

import pandas as pd
import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)


def handle_missing(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """Deal with missing values using the chosen strategy

    Args:
        df: Input DataFrame (only the columns we care about)
        strategy: 'drop' removes rows with NaNs, 'mean' fills them with column averages
    """
    if strategy == "mean":
        filled = df.fillna(df.mean(numeric_only=True))
        logger.info("Filled missing values with column means.")
        return filled

    # Default: drop rows that have any NaN.
    cleaned = df.dropna()
    logger.info("Dropped rows with missing values (%d -> %d rows).", len(df), len(cleaned))
    return cleaned


def discretise(df: pd.DataFrame, columns: list[str], bins: int = 5) -> pd.DataFrame:
    """Cut numeric columns into equal-width bins

    This is a simple placeholder — a real SURD implementation may need
    a smarter binning approach
    """
    out = df.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.cut(out[col], bins=bins, labels=False, duplicates="drop")
    logger.info("Discretised %d columns into %d bins.", len(columns), bins)
    return out
