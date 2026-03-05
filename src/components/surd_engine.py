"""Computes Partial Information Decomposition for 2 sources and 1 target.

Method: I_min (Williams & Beer, 2010).
For >2 sources we run PID on every pair and aggregate.
"""

import numpy as np
import pandas as pd
from itertools import combinations

from src.logger import get_logger

logger = get_logger(__name__)


# ── Low-level information theory helpers ─────────────────────────────

def _prob_table(series: pd.Series) -> dict:
    """Count how often each value appears and return probabilities."""
    counts = series.value_counts()
    total = counts.sum()
    return (counts / total).to_dict()


def _joint_prob(col_a: pd.Series, col_b: pd.Series) -> dict:
    """Joint probability of two columns as {(a,b): p} dict."""
    paired = list(zip(col_a, col_b))
    n = len(paired)
    counts = {}
    for pair in paired:
        counts[pair] = counts.get(pair, 0) + 1
    return {k: v / n for k, v in counts.items()}


def _mutual_info(x: pd.Series, y: pd.Series) -> float:
    """Mutual information I(X;Y) in bits."""
    px = _prob_table(x)
    py = _prob_table(y)
    pxy = _joint_prob(x, y)

    mi = 0.0
    for (xv, yv), pj in pxy.items():
        if pj > 0 and px.get(xv, 0) > 0 and py.get(yv, 0) > 0:
            mi += pj * np.log2(pj / (px[xv] * py[yv]))
    return max(mi, 0.0)


def _conditional_prob_y_given_x(x: pd.Series, y: pd.Series) -> dict:
    """Returns {(x_val, y_val): P(y|x)} for every observed pair."""
    pxy = _joint_prob(x, y)
    px = _prob_table(x)
    cond = {}
    for (xv, yv), pj in pxy.items():
        if px.get(xv, 0) > 0:
            cond[(xv, yv)] = pj / px[xv]
    return cond


def _joint_mi(x1: pd.Series, x2: pd.Series, y: pd.Series) -> float:
    """Mutual information I(X1,X2 ; Y) — treat (X1,X2) as one joint variable."""
    joint_x = pd.Series(list(zip(x1, x2)), index=x1.index)
    return _mutual_info(joint_x, y)


# ── I_min redundancy (Williams & Beer 2010) ─────────────────────────

def _specific_info(x: pd.Series, y: pd.Series, y_val) -> float:
    """Specific information I(X; Y=y) = sum_x p(x|y) * log2(p(x|y)/p(x))."""
    px = _prob_table(x)
    cond = _conditional_prob_y_given_x(y, x)

    si = 0.0
    for xv in px:
        p_x_given_y = cond.get((y_val, xv), 0.0)
        if p_x_given_y > 0 and px[xv] > 0:
            si += p_x_given_y * np.log2(p_x_given_y / px[xv])
    return si


def _i_min(x1: pd.Series, x2: pd.Series, y: pd.Series) -> float:
    """I_min redundancy: sum_y p(y) * min(I_spec(X1;y), I_spec(X2;y))."""
    py = _prob_table(y)
    redundancy = 0.0
    for yv, p_yv in py.items():
        si1 = _specific_info(x1, y, yv)
        si2 = _specific_info(x2, y, yv)
        redundancy += p_yv * min(si1, si2)
    return max(redundancy, 0.0)


# ── Public API ───────────────────────────────────────────────────────

def compute_pid_2source(df: pd.DataFrame, src1: str, src2: str,
                        target: str) -> dict:
    """PID for exactly 2 sources and 1 target.

    Returns dict with unique_x1, unique_x2, redundant, synergy (all in bits).
    """
    x1 = df[src1]
    x2 = df[src2]
    y = df[target]

    mi_x1_y = _mutual_info(x1, y)
    mi_x2_y = _mutual_info(x2, y)
    mi_joint = _joint_mi(x1, x2, y)
    redundant = _i_min(x1, x2, y)

    unique_x1 = max(mi_x1_y - redundant, 0.0)
    unique_x2 = max(mi_x2_y - redundant, 0.0)
    synergy = max(mi_joint - unique_x1 - unique_x2 - redundant, 0.0)

    return {
        "unique_x1": round(unique_x1, 6),
        "unique_x2": round(unique_x2, 6),
        "redundant": round(redundant, 6),
        "synergy": round(synergy, 6),
        "mi_x1_y": round(mi_x1_y, 6),
        "mi_x2_y": round(mi_x2_y, 6),
        "mi_joint": round(mi_joint, 6),
    }


def compute_surd(df_binned: pd.DataFrame, sources: list[str],
                 target: str) -> dict:
    """Run PID-based decomposition and return a standard results dict.

    For 2 sources: direct PID.
    For >2 sources: PID on every pair, then aggregate.
    """
    n = len(sources)

    if n == 2:
        pid = compute_pid_2source(df_binned, sources[0], sources[1], target)
        unique = {
            sources[0]: pid["unique_x1"],
            sources[1]: pid["unique_x2"],
        }
        logger.info("PID computed for 2 sources -> target '%s'.", target)
        return {
            "unique": unique,
            "redundant": pid["redundant"],
            "synergy": pid["synergy"],
            "pairwise_synergy": None,
            "pairwise_redundancy": None,
            "method": "PID I_min (Williams & Beer 2010), 2-source",
        }

    # More than 2 sources: pairwise mode.
    pairwise_syn = {}
    pairwise_red = {}
    unique_accum = {s: [] for s in sources}

    for a, b in combinations(sources, 2):
        pid = compute_pid_2source(df_binned, a, b, target)
        key = f"{a}|{b}"
        pairwise_syn[key] = pid["synergy"]
        pairwise_red[key] = pid["redundant"]
        unique_accum[a].append(pid["unique_x1"])
        unique_accum[b].append(pid["unique_x2"])

    # Average unique across all pairs each source appeared in.
    unique = {s: round(float(np.mean(vals)), 6)
              for s, vals in unique_accum.items()}

    # Overall redundancy and synergy = average across pairs.
    redundant = round(float(np.mean(list(pairwise_red.values()))), 6)
    synergy = round(float(np.mean(list(pairwise_syn.values()))), 6)

    logger.info("PID (pairwise) computed for %d sources -> '%s'.", n, target)
    return {
        "unique": unique,
        "redundant": redundant,
        "synergy": synergy,
        "pairwise_synergy": pairwise_syn,
        "pairwise_redundancy": pairwise_red,
        "method": f"PID I_min pairwise over {n} sources",
    }
