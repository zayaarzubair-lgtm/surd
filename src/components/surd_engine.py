"""Produces SURD-style results (Unique / Redundant / Synergy)

Gives a Fake result at the moment
look at research papers to figure out which method to use
"""

import numpy as np
from itertools import combinations

from src.logger import get_logger

logger = get_logger(__name__)

SEED = 42


def compute_surd(df_binned, sources: list[str], target: str) -> dict:

    rng = np.random.default_rng(SEED)
    n = len(sources)

    # Dummy unique information per source (bigger for first source).
    raw = rng.uniform(0.05, 0.35, size=n)
    raw = np.sort(raw)[::-1]
    unique = {src: round(float(raw[i]), 4) for i, src in enumerate(sources)}

    # Dummy redundant and synergy totals.
    redundant = round(float(rng.uniform(0.02, 0.15)), 4)
    synergy = round(float(rng.uniform(0.01, 0.10)), 4)

    # Pairwise synergy (only when >2 sources).
    pairwise = None
    if n > 2:
        pairwise = {}
        for a, b in combinations(sources, 2):
            key = f"{a}|{b}"
            pairwise[key] = round(float(rng.uniform(0.005, 0.06)), 4)

    logger.info("SURD computed (dummy) for %d sources -> target '%s'.", n, target)
    return {
        "unique": unique,
        "redundant": redundant,
        "synergy": synergy,
        "pairwise_synergy": pairwise,
    }
