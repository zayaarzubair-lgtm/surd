"""Compare SURD results against transfer entropy on the same datasets.

Transfer entropy (Schreiber, 2000) measures how much knowing one variable's
past reduces uncertainty about another variable's future, conditional on
the target's own past. It is a single number per source-target pair.

SURD's advantage: it can tell you whether two sources carry the same
information (redundancy) or only matter together (synergy). Transfer
entropy cannot.

This module computes transfer entropy from each agent to the target and
compares it with the SURD breakdown.
"""

import numpy as np
import pandas as pd

from src.components.surd_engine import run_surd
from src.logger import get_logger

logger = get_logger(__name__)


def _bin_series(values: np.ndarray, nbins: int) -> np.ndarray:
    """Discretise a continuous series into integer bins for PyInform."""
    edges = np.linspace(values.min(), values.max() + 1e-9, nbins + 1)
    binned = np.digitize(values, edges) - 1
    binned = np.clip(binned, 0, nbins - 1)
    return binned.astype(int)


def transfer_entropy_naive(source: np.ndarray, target: np.ndarray,
                            k: int = 1) -> float:
    """Compute transfer entropy using histograms (no PyInform dependency).

    TE(source -> target) = H(target_t+1 | target_t..target_t-k+1)
                         - H(target_t+1 | target_t..target_t-k+1, source_t)

    This is a histogram-based estimator. PyInform would be more accurate
    but adds a C dependency. For benchmark comparisons this is sufficient.
    """
    n = len(target)
    if n < k + 2:
        return 0.0

    # Build joint observations: target[t+1], target[t..t-k+1], source[t]
    target_future = target[k:]
    source_past = source[k-1:-1] if k > 0 else source[:-1]

    # Target history (last k values)
    target_hist = []
    for i in range(k):
        target_hist.append(target[i:n - k + i])
    target_hist = np.array(target_hist).T  # shape (n-k, k)

    # Encode target history as a single integer per row.
    nbins = max(int(target.max()) + 1, int(source.max()) + 1)
    target_hist_int = np.zeros(len(target_hist), dtype=int)
    for col in range(k):
        target_hist_int = target_hist_int * nbins + target_hist[:, col]

    # Joint distributions
    joint_full = np.zeros((nbins, nbins ** k, nbins))   # future, hist, source
    joint_no_src = np.zeros((nbins, nbins ** k))        # future, hist
    marg_hist = np.zeros(nbins ** k)                    # hist
    marg_hist_src = np.zeros((nbins ** k, nbins))       # hist, source

    n_obs = len(target_future)
    for i in range(n_obs):
        f = int(target_future[i])
        h = int(target_hist_int[i])
        s = int(source_past[i])
        joint_full[f, h, s] += 1
        joint_no_src[f, h] += 1
        marg_hist[h] += 1
        marg_hist_src[h, s] += 1

    joint_full /= n_obs
    joint_no_src /= n_obs
    marg_hist /= n_obs
    marg_hist_src /= n_obs

    # Compute the two conditional entropies and subtract.
    te = 0.0
    for f in range(nbins):
        for h in range(nbins ** k):
            for s in range(nbins):
                p_fhs = joint_full[f, h, s]
                p_hs = marg_hist_src[h, s]
                p_fh = joint_no_src[f, h]
                p_h = marg_hist[h]
                if p_fhs > 0 and p_hs > 0 and p_fh > 0 and p_h > 0:
                    te += p_fhs * np.log2((p_fhs * p_h) / (p_hs * p_fh))

    return max(te, 0.0)


def compare_surd_te(df: pd.DataFrame, target: str, agents: list[str],
                     tau: int = 1, nbins: int = 8,
                     te_history: int = 1) -> dict:
    """Run both SURD and transfer entropy on the same dataset.

    Returns a dictionary suitable for tabular display in the dashboard.
    """
    # SURD result.
    surd = run_surd(df, target=target, agents=agents, tau=tau, nbins=nbins)

    # Transfer entropy from each agent to the target.
    target_binned = _bin_series(df[target].values, nbins)
    te_results = {}
    for agent in agents:
        if agent == target:
            te_results[agent] = float("nan")  # self-TE is awkward; skip
            continue
        agent_binned = _bin_series(df[agent].values, nbins)
        te = transfer_entropy_naive(agent_binned, target_binned, k=te_history)
        te_results[agent] = round(te, 6)

    logger.info("Transfer entropy: %s", te_results)

    # Sum of unique contributions from SURD for non-target agents.
    surd_unique = surd["unique"]

    return {
        "transfer_entropy": te_results,
        "surd_unique": {a: surd_unique.get(a, 0) for a in agents if a != target},
        "surd_total_synergy": surd["total_synergy"],
        "surd_total_redundant": surd["total_redundant"],
        "surd_synergy_breakdown": surd.get("synergy_breakdown", {}),
        "surd_redundant_breakdown": surd.get("redundant_breakdown", {}),
        "tau": tau,
        "nbins": nbins,
        "te_history": te_history,
    }


def explain_comparison(comparison: dict) -> str:
    """Build a paragraph explaining what the comparison reveals."""
    te = comparison["transfer_entropy"]
    sus = comparison["surd_unique"]

    # Find agents where TE is high but SURD unique is low.
    surprises = []
    for agent in te:
        if np.isnan(te[agent]):
            continue
        te_val = te[agent]
        u_val = sus.get(agent, 0)
        if te_val > 0.05 and u_val < 0.5 * te_val:
            surprises.append((agent, te_val, u_val))

    lines = [
        "**Transfer entropy results (one number per agent):**",
    ]
    for agent, val in te.items():
        if np.isnan(val):
            lines.append(f"- TE({agent} -> target): N/A (self-causation)")
        else:
            lines.append(f"- TE({agent} -> target) = {val:.4f} bits")

    lines.append("")
    lines.append("**SURD unique contributions (same agents):**")
    for agent, val in sus.items():
        lines.append(f"- U({agent}) = {val:.4f} bits")

    lines.append("")
    if surprises:
        lines.append("**Where the methods disagree:**")
        for agent, te_val, u_val in surprises:
            lines.append(
                f"- TE reports {agent} carries {te_val:.4f} bits of causal "
                f"information about the target, but SURD's unique "
                f"contribution from {agent} is only {u_val:.4f} bits. "
                f"The difference suggests the information {agent} appears "
                f"to carry is actually shared with other agents (redundancy) "
                f"or only emerges in combination with other agents (synergy)."
            )
    else:
        lines.append("Transfer entropy and SURD's unique contributions broadly agree on this dataset.")

    lines.append("")
    lines.append(
        f"SURD identifies {comparison['surd_total_synergy']:.4f} bits of "
        f"synergistic causality and {comparison['surd_total_redundant']:.4f} "
        f"bits of redundancy that transfer entropy cannot detect on its own. "
        f"This is the practical advantage of the decomposition: it tells you "
        f"not just that variables affect the target, but how they share or "
        f"combine that influence."
    )

    return "\n".join(lines)
