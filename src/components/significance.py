"""Statistical significance testing for SURD results via permutation.

The idea: if we shuffle each agent's time series, we destroy any
genuine causal relationship while preserving the marginal distributions.
By running SURD many times on shuffled data, we build a null distribution
for what the components look like when there is no real causality.
The p-value tells us how likely the observed result would be under the
null hypothesis of no causality.
"""

import numpy as np
import pandas as pd

from src.components.surd_engine import run_surd
from src.logger import get_logger

logger = get_logger(__name__)


def permutation_test(df: pd.DataFrame, target: str, agents: list[str],
                     tau: int = 1, nbins: int = 8,
                     n_permutations: int = 100,
                     seed: int = 42) -> dict:
    """Run SURD on shuffled data to test significance of observed values.

    Args:
        df:             DataFrame with the original time series.
        target:         Target column name.
        agents:         Agent column names.
        tau:            Time lag.
        nbins:          Histogram bins.
        n_permutations: Number of shuffles to perform.
        seed:           Random seed for reproducibility.

    Returns:
        dict with observed values, null distributions, and p-values
        for each component.
    """
    rng = np.random.default_rng(seed)

    # Run SURD on the real data first.
    observed = run_surd(df, target=target, agents=agents,
                        tau=tau, nbins=nbins)
    obs_unique = observed["total_unique"]
    obs_redundant = observed["total_redundant"]
    obs_synergy = observed["total_synergy"]
    obs_leak = observed["info_leak"]

    logger.info("Starting permutation test with %d shuffles", n_permutations)

    # Build null distributions by shuffling each agent independently.
    null_unique = []
    null_redundant = []
    null_synergy = []
    null_leak = []

    for i in range(n_permutations):
        shuffled = df.copy()
        # Shuffle each agent column independently. Target stays put.
        for agent in agents:
            if agent != target:
                shuffled[agent] = rng.permutation(shuffled[agent].values)

        try:
            r = run_surd(shuffled, target=target, agents=agents,
                         tau=tau, nbins=nbins)
            null_unique.append(r["total_unique"])
            null_redundant.append(r["total_redundant"])
            null_synergy.append(r["total_synergy"])
            null_leak.append(r["info_leak"])
        except Exception as exc:
            logger.warning("Permutation %d failed: %s", i, exc)

        if (i + 1) % 20 == 0:
            logger.info("Permutation %d/%d complete", i+1, n_permutations)

    # P-value: fraction of null samples >= observed.
    # For leak it is fraction <= observed (we expect leak to be high under null).
    def p_value(observed_val: float, null_dist: list[float],
                direction: str = "greater") -> float:
        null_arr = np.array(null_dist)
        if direction == "greater":
            return float((null_arr >= observed_val).sum() + 1) / (len(null_arr) + 1)
        return float((null_arr <= observed_val).sum() + 1) / (len(null_arr) + 1)

    p_unique = p_value(obs_unique, null_unique, "greater")
    p_redundant = p_value(obs_redundant, null_redundant, "greater")
    p_synergy = p_value(obs_synergy, null_synergy, "greater")
    p_leak = p_value(obs_leak, null_leak, "less")

    return {
        "observed": {
            "unique": obs_unique,
            "redundant": obs_redundant,
            "synergy": obs_synergy,
            "leak": obs_leak,
        },
        "null_distributions": {
            "unique": null_unique,
            "redundant": null_redundant,
            "synergy": null_synergy,
            "leak": null_leak,
        },
        "null_means": {
            "unique": float(np.mean(null_unique)) if null_unique else 0.0,
            "redundant": float(np.mean(null_redundant)) if null_redundant else 0.0,
            "synergy": float(np.mean(null_synergy)) if null_synergy else 0.0,
            "leak": float(np.mean(null_leak)) if null_leak else 0.0,
        },
        "null_stds": {
            "unique": float(np.std(null_unique)) if null_unique else 0.0,
            "redundant": float(np.std(null_redundant)) if null_redundant else 0.0,
            "synergy": float(np.std(null_synergy)) if null_synergy else 0.0,
            "leak": float(np.std(null_leak)) if null_leak else 0.0,
        },
        "p_values": {
            "unique": p_unique,
            "redundant": p_redundant,
            "synergy": p_synergy,
            "leak": p_leak,
        },
        "n_permutations": len(null_unique),
        "tau": tau,
        "nbins": nbins,
    }


def interpret_pvalue(p: float) -> str:
    """Convert a p-value into a readable significance label."""
    if p < 0.001:
        return "highly significant (p < 0.001)"
    if p < 0.01:
        return "very significant (p < 0.01)"
    if p < 0.05:
        return "significant (p < 0.05)"
    if p < 0.1:
        return "marginally significant (p < 0.1)"
    return "not significant (p >= 0.1)"
