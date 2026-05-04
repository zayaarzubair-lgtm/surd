"""Interpretation helpers: short explanations of what numbers mean.

Used across the dashboard to give users context next to results.
The goal is to make a number less intimidating by explaining what
it represents and how to read it.

All functions are defensive about input types: dictionaries from
run_surd / run_lag_sweep, plain floats, and lists are all handled
without crashing the dashboard.
"""

from typing import Any
import math


def _safe_get(d: Any, *keys, default: float = 0.0) -> float:
    """Try a sequence of keys until one works. Returns default if none do."""
    if not isinstance(d, dict):
        return default
    for key in keys:
        if key in d:
            try:
                return float(d[key])
            except (TypeError, ValueError):
                continue
    return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce to float, returning default for None/NaN/non-numeric."""
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


# ─── Total component interpretations ───

def interpret_unique(value: float, total: float = None) -> str:
    """Explain what total unique means."""
    value = _safe_float(value)
    if value < 0.02:
        return ("Near zero. None of the agents alone carry meaningful causal "
                "information about the target. The action is in combinations "
                "or in unmeasured factors.")
    if total and total > 0:
        share = value / total
        if share > 0.6:
            return (f"Most of the causal information ({value:.4f} bits) comes "
                    f"from individual agents acting on their own. This is "
                    f"the simplest causal pattern.")
        if share > 0.3:
            return (f"A meaningful share of the information ({value:.4f} bits) "
                    f"comes from individual agents. The rest is in combinations "
                    f"or in factors not measured.")
    return (f"Each individual agent contributes some causal information about "
            f"the target on its own. The total across all agents is {value:.4f} bits.")


def interpret_redundant(value: float, total: float = None) -> str:
    """Explain what total redundant means."""
    value = _safe_float(value)
    if value < 0.02:
        return ("Near zero. The agents do not carry overlapping information. "
                "Each one tells you something different.")
    if total and total > 0:
        share = value / total
        if share > 0.4:
            return (f"A large fraction of the causal information ({value:.4f} bits) "
                    f"is duplicated across agents. They are partially redundant "
                    f"with each other; you could drop some without losing much.")
    return (f"Some agents carry overlapping causal information ({value:.4f} bits "
            f"shared). They are partially redundant with each other.")


def interpret_synergy(value: float, total: float = None) -> str:
    """Explain what total synergy means."""
    value = _safe_float(value)
    if value < 0.02:
        return ("Near zero. The agents do not need to be observed together "
                "to predict the target. Each can be analysed separately.")
    if total and total > 0:
        share = value / total
        if share > 0.5:
            return (f"Most of the causal information ({value:.4f} bits) only "
                    f"emerges when agents are observed together. Looking at "
                    f"any single agent in isolation would miss the structure.")
    return (f"Some causal information ({value:.4f} bits) only emerges when "
            f"agents are observed together. They interact in ways that no "
            f"single agent can reveal.")


def interpret_leak(value: float) -> str:
    """Explain what info leak means."""
    value = _safe_float(value)
    pct = value * 100
    if value < 0.1:
        return (f"Only {pct:.1f}% of the target's future is unexplained. The "
                f"selected agents capture almost all the causal information "
                f"the data contains.")
    if value < 0.4:
        return (f"{pct:.1f}% of the target's future is unexplained by these "
                f"agents. There may be additional drivers worth measuring.")
    if value < 0.7:
        return (f"{pct:.1f}% of the target's future is unexplained by these "
                f"agents. A substantial fraction comes from unmeasured factors "
                f"(temperature, time-of-day, external shocks, etc).")
    return (f"{pct:.1f}% of the target's future is unexplained by these agents. "
            f"The selected agents do not carry enough information to predict "
            f"the target reliably; the dominant drivers are not in your data.")


# ─── Per-agent and pair interpretations ───

def interpret_unique_per_agent(agent: str, value: float, all_unique: dict = None) -> str:
    """Explain a single agent's unique contribution."""
    value = _safe_float(value)
    if value < 0.001:
        return (f"{agent} on its own does not predict the target. Either it "
                f"genuinely has no causal effect, or its influence only shows "
                f"up alongside other agents (synergy) or is shared with them "
                f"(redundancy).")
    if all_unique and isinstance(all_unique, dict):
        try:
            max_unique = max(_safe_float(v) for v in all_unique.values())
            if max_unique > 0 and abs(value - max_unique) < 1e-9:
                return (f"{agent} is the strongest individual predictor of the target "
                        f"in this dataset, contributing {value:.4f} bits on its own.")
        except (ValueError, TypeError):
            pass
    return (f"{agent} contributes {value:.4f} bits of unique causal information "
            f"on its own.")


def interpret_pair_synergy(pair: str, value: float, top_pair: tuple = None) -> str:
    """Explain a synergistic pair's value."""
    value = _safe_float(value)
    if top_pair and isinstance(top_pair, (tuple, list)) and len(top_pair) > 0 and pair == top_pair[0]:
        return (f"This is the strongest synergistic pair in the dataset at "
                f"{value:.4f} bits. These agents only reveal their causal role "
                f"when observed together.")
    if value < 0.001:
        return ""
    return (f"Together this pair carries {value:.4f} bits of causal information "
            f"that neither agent provides alone.")


def interpret_pair_redundant(pair: str, value: float, top_pair: tuple = None) -> str:
    """Explain a redundant pair's value."""
    value = _safe_float(value)
    if top_pair and isinstance(top_pair, (tuple, list)) and len(top_pair) > 0 and pair == top_pair[0]:
        return (f"This is the most redundant pair in the dataset, sharing "
                f"{value:.4f} bits of causal information. Knowing one of them "
                f"largely tells you what the other would tell you.")
    if value < 0.001:
        return ""
    return (f"This pair shares {value:.4f} bits of overlapping causal "
            f"information about the target.")


# ─── Statistical interpretations ───

def interpret_pvalue(p: float, component: str = "value") -> str:
    """Explain a p-value with context."""
    try:
        p_float = float(p)
    except (TypeError, ValueError):
        return f"P-value not available for {component}."

    if math.isnan(p_float):
        return f"P-value not available for {component}."

    if p_float < 0.01:
        return (f"The {component} value is highly unlikely to occur by chance "
                f"(p = {p_float:.3f}). It reflects real causal structure in the data, "
                f"not a histogram artefact or noise.")
    if p_float < 0.05:
        return (f"The {component} value is unlikely to occur by chance "
                f"(p = {p_float:.3f}). It probably reflects a real pattern, with a "
                f"small chance of being noise.")
    if p_float < 0.1:
        return (f"The {component} value is borderline (p = {p_float:.3f}). It might "
                f"reflect a real pattern or might be noise; consider running "
                f"more permutations or collecting more data.")
    return (f"The {component} value could plausibly arise by chance "
            f"(p = {p_float:.3f}). It does not stand out from the null distribution, "
            f"so we cannot claim it reflects a real causal pattern.")


# ─── Lag sweep interpretation ───

def interpret_lag_curve(sweep_data: Any) -> str:
    """Explain the lag sweep curve shape.

    sweep_data is a list of dicts from run_lag_sweep, each with keys
    'tau', 'total_unique', 'total_redundant', 'total_synergy', 'info_leak'.
    Defensive: also handles list-of-list and list-of-tuple formats.
    """
    if not sweep_data:
        return "No lag sweep data to interpret yet."

    def get_total(item):
        """Extract the sum of unique + redundant + synergy from an item."""
        if isinstance(item, dict):
            return (
                _safe_get(item, "total_unique", "unique", default=0)
                + _safe_get(item, "total_redundant", "redundant", default=0)
                + _safe_get(item, "total_synergy", "synergy", default=0)
            )
        if isinstance(item, (list, tuple)) and len(item) >= 4:
            try:
                return _safe_float(item[1]) + _safe_float(item[2]) + _safe_float(item[3])
            except (TypeError, ValueError, IndexError):
                return 0.0
        return 0.0

    try:
        first_total = get_total(sweep_data[0])
        last_total = get_total(sweep_data[-1])
    except Exception:
        return "Lag sweep results are available; review the curve and table for the trend."

    if first_total > 3 * last_total and first_total > 0.1:
        return ("Causal information drops sharply as the time lag increases. "
                "This is typical for systems where the dominant effects are "
                "fast: knowing the past one or two steps ahead is informative, "
                "but knowing it ten steps ahead is not. The leak grows with tau "
                "because longer-range prediction is genuinely harder.")
    if last_total > first_total:
        return ("Causal information grows with the time lag. This suggests the "
                "effects in this system act over longer timescales rather than "
                "immediately. Try larger tau values to find where the signal "
                "peaks.")
    return ("Causal information stays roughly stable across the tested lags. "
            "The system's predictive structure does not depend strongly on "
            "how far ahead you look in this range.")


# ─── Transfer entropy comparison interpretation ───

def interpret_te_vs_unique(agent: str, te: float, unique: float) -> str:
    """Explain how transfer entropy compares with SURD's unique contribution."""
    te = _safe_float(te, default=float('nan'))
    unique = _safe_float(unique, default=float('nan'))

    if math.isnan(te) or math.isnan(unique):
        return f"{agent}: comparison not available (likely self-causation)."

    if te < 0.01:
        return f"Both methods agree that {agent} carries little causal information about the target."
    if abs(te - unique) < 0.01:
        return (f"Both methods agree: {agent} contributes around {unique:.4f} bits "
                f"of unique causal information.")
    if te > 2 * unique:
        return (f"Transfer entropy reports {te:.4f} bits for {agent}, but SURD "
                f"shows only {unique:.4f} bits is unique. The difference "
                f"({te - unique:.4f} bits) is information that {agent} shares "
                f"with other agents (redundancy) or only contributes in "
                f"combination with them (synergy).")
    return (f"Transfer entropy reports {te:.4f} bits for {agent}; SURD's unique "
            f"contribution is {unique:.4f} bits. The slight difference reflects "
            f"that some of {agent}'s apparent influence is actually shared "
            f"with other agents.")
