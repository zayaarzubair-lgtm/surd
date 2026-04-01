"""Handles missing values and builds time-lagged arrays for SURD.

SURD needs pairs of past and future observations. This module
takes a time-series DataFrame and constructs those pairs.
"""

import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


def handle_missing(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """Deal with missing values.

    'drop' removes rows with any NaN. 'mean' fills gaps with column averages.
    """
    if strategy == "mean":
        filled = df.fillna(df.mean(numeric_only=True))
        logger.info("Filled missing values with column means.")
        return filled

    cleaned = df.dropna()
    logger.info("Dropped rows with missing values (%d -> %d rows).", len(df), len(cleaned))
    return cleaned


def build_lagged_array(target_series: np.ndarray, agent_series: list[np.ndarray],
                       tau: int) -> np.ndarray:
    """Stack the target's future with the agents' past.

    Row 0 = target shifted forward by tau steps (the future).
    Rows 1..n = each agent's values at the earlier time (the past).
    Columns = time points where both past and future are available.

    This is the format the official SURD algorithm expects.
    """
    if tau < 1:
        raise ValueError("Time lag tau must be at least 1.")

    future = target_series[tau:]
    rows = [future]
    for agent in agent_series:
        rows.append(agent[:-tau] if tau > 0 else agent)

    stacked = np.vstack(rows)
    logger.info("Built lagged array: %d variables x %d timepoints (tau=%d).",
                stacked.shape[0], stacked.shape[1], tau)
    return stacked


def build_histogram(Y: np.ndarray, nbins: int) -> np.ndarray:
    """Build a joint histogram from the lagged array.

    Y has shape (n_variables, n_timepoints). The histogram has
    one dimension per variable, each with nbins bins.
    """
    hist, _ = np.histogramdd(Y.T, bins=nbins)
    return hist
