"""My wrapper around the official SURD algorithm.

This handles: extracting columns from a DataFrame, building time-lagged
arrays, computing the histogram, calling the official surd() function,
and packaging the results into a dict that the rest of my dashboard uses.

The actual SURD algorithm is in src/components/surd_lib/ and comes from
https://github.com/ALD-Lab/SURD (Martínez-Sánchez et al., 2024).
"""

import numpy as np
import pandas as pd

from src.components.surd_lib.surd_core import surd
from src.components.data_transformation import build_lagged_array, build_histogram
from src.logger import get_logger

logger = get_logger(__name__)


def run_surd(df: pd.DataFrame, target: str, agents: list[str],
             tau: int = 1, nbins: int = 10) -> dict:
    """Run the full SURD causal decomposition.

    Args:
        df:     DataFrame with time-series data (rows in time order).
        target: Name of the target column (variable whose future I'm predicting).
        agents: Names of the agent columns (variables whose past might cause the target).
        tau:    Time lag in number of rows.
        nbins:  Number of histogram bins per variable.

    Returns:
        dict with unique, redundant, synergy breakdowns, info_leak,
        mutual info, raw SURD outputs, method name, and metadata.
    """
    n_agents = len(agents)
    if n_agents < 1:
        raise ValueError("Need at least 1 agent variable.")
    if tau < 1:
        raise ValueError("Lag tau must be at least 1.")

    # Extract the columns as numpy arrays.
    target_series = df[target].values.astype(float)
    agent_series = [df[a].values.astype(float) for a in agents]

    # Step 1 — Build the time-lagged array.
    Y = build_lagged_array(target_series, agent_series, tau)
    n_timepoints = Y.shape[1]

    # Step 2 — Build the joint histogram.
    hist = build_histogram(Y, nbins)

    # Step 3 — Run the official SURD decomposition.
    I_R, I_S, MI, info_leak = surd(hist)

    logger.info("SURD complete. Info leak = %.4f", info_leak)

    # Step 4 — Translate tuple keys into readable names.
    unique = {}
    redundant = {}
    synergy = {}

    for key, value in I_R.items():
        label = _key_to_label(key, agents)
        if len(key) == 1:
            unique[label] = round(float(value), 6)
        else:
            redundant[label] = round(float(value), 6)

    for key, value in I_S.items():
        label = _key_to_label(key, agents)
        synergy[label] = round(float(value), 6)

    mi_named = {}
    for key, value in MI.items():
        label = _key_to_label(key, agents)
        mi_named[label] = round(float(value), 6)

    # Step 5 — Summary totals.
    total_unique = sum(unique.values())
    total_redundant = sum(redundant.values())
    total_synergy = sum(synergy.values())

    return {
        "unique": unique,
        "redundant_breakdown": redundant,
        "synergy_breakdown": synergy,
        "total_unique": round(total_unique, 6),
        "total_redundant": round(total_redundant, 6),
        "total_synergy": round(total_synergy, 6),
        "info_leak": round(float(info_leak), 6),
        "mutual_info": mi_named,
        "I_R_raw": {str(k): float(v) for k, v in I_R.items()},
        "I_S_raw": {str(k): float(v) for k, v in I_S.items()},
        "MI_raw": {str(k): float(v) for k, v in MI.items()},
        "method": "SURD (Martínez-Sánchez et al., 2024)",
        "meta": {
            "tau": tau,
            "nbins": nbins,
            "n_agents": n_agents,
            "n_timepoints": n_timepoints,
            "target": target,
            "agent_names": agents,
        },
    }


def _key_to_label(key: tuple, agent_names: list[str]) -> str:
    """Convert a SURD tuple key like (1,2) into agent names like 'temp|humid'."""
    parts = []
    for idx in key:
        agent_idx = idx - 1
        if 0 <= agent_idx < len(agent_names):
            parts.append(agent_names[agent_idx])
        else:
            parts.append(f"var{idx}")
    return "|".join(parts)
