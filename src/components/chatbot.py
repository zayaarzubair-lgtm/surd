"""Answers questions about SURD results.

Mode 1 (default): rule-based pattern matching, no dependencies.
Mode 2 (optional): if Ollama is running locally, uses a local LLM
for more flexible answers. Falls back to mode 1 silently.
"""

import re
import json

from src.logger import get_logger

logger = get_logger(__name__)


# ── Ollama detection and calling ──────────────────────────

def _ollama_available() -> bool:
    """Check if Ollama is running locally."""
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags")
        resp = urllib.request.urlopen(req, timeout=2)
        return resp.status == 200
    except Exception:
        return False


def _ask_ollama(question: str, result: dict, model: str = "llama3.2:3b") -> str | None:
    """Send a question and the SURD result to Ollama. Returns None on failure."""
    try:
        import urllib.request

        # Build a concise context from the result dict.
        context = _build_context(result)

        prompt = (
            "You are a helpful assistant built into a causal analysis dashboard "
            "called SURDview. The user has just run a SURD causal decomposition "
            "on their time-series data. Here are the results:\n\n"
            f"{context}\n\n"
            "Answer the user's question about these results. Be specific, "
            "use the actual numbers, and keep it concise. If the question "
            "is not about the results, say you can only help with SURD analysis.\n\n"
            f"User's question: {question}"
        )

        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read().decode("utf-8"))
        answer = data.get("response", "").strip()
        if answer:
            logger.info("Ollama answered question: %s", question[:50])
            return answer
    except Exception as exc:
        logger.warning("Ollama call failed: %s", exc)
    return None


def _build_context(result: dict) -> str:
    """Turn the result dict into a readable text block for the LLM."""
    meta = result.get("meta", {})
    lines = [
        f"Target variable: {meta.get('target', 'unknown')}",
        f"Agent variables: {', '.join(meta.get('agent_names', []))}",
        f"Time lag (tau): {meta.get('tau', '?')}",
        f"Histogram bins: {meta.get('nbins', '?')}",
        f"Usable time points: {meta.get('n_timepoints', '?')}",
        "",
        "Unique causal contributions (bits):",
    ]
    for name, val in result.get("unique", {}).items():
        lines.append(f"  {name}: {val:.6f}")

    lines.append("")
    lines.append("Redundant causal contributions (bits):")
    for name, val in result.get("redundant_breakdown", {}).items():
        lines.append(f"  {name}: {val:.6f}")

    lines.append("")
    lines.append("Synergistic causal contributions (bits):")
    for name, val in result.get("synergy_breakdown", {}).items():
        lines.append(f"  {name}: {val:.6f}")

    lines.append("")
    lines.append(f"Total unique: {result.get('total_unique', 0):.6f}")
    lines.append(f"Total redundant: {result.get('total_redundant', 0):.6f}")
    lines.append(f"Total synergy: {result.get('total_synergy', 0):.6f}")
    lines.append(f"Information leak: {result.get('info_leak', 0):.4f} "
                 f"({result.get('info_leak', 0)*100:.1f}%)")

    return "\n".join(lines)


# ── Rule-based fallback ───────────────────────────────────

def _matches(text: str, keywords: list[str]) -> bool:
    """Check if any keyword appears in the text."""
    return any(kw in text for kw in keywords)


def _top_agent(unique: dict) -> tuple[str, float]:
    """Return the agent with the highest unique contribution."""
    if not unique:
        return ("none", 0.0)
    name = max(unique, key=unique.get)
    return (name, unique[name])


def _top_pair(breakdown: dict) -> tuple[str, float]:
    """Return the pair with the highest value from a breakdown dict."""
    pairs = {k: v for k, v in breakdown.items() if "|" in k}
    if not pairs:
        return ("none", 0.0)
    name = max(pairs, key=pairs.get)
    return (name, pairs[name])


def _dominant_component(result: dict) -> str:
    """Which component is largest: unique, redundant, or synergy?"""
    tu = result.get("total_unique", 0)
    tr = result.get("total_redundant", 0)
    ts = result.get("total_synergy", 0)
    if ts >= tr and ts >= tu:
        return "synergy"
    if tr >= tu and tr >= ts:
        return "redundancy"
    return "unique"


def _rule_based(question: str, result: dict) -> str:
    """Answer using pattern matching on the question text."""
    q = question.lower().strip()
    meta = result.get("meta", {})
    target = meta.get("target", "the target")
    agents = meta.get("agent_names", [])
    tau = meta.get("tau", 1)
    unique = result.get("unique", {})
    redundant = result.get("redundant_breakdown", {})
    synergy = result.get("synergy_breakdown", {})
    tu = result.get("total_unique", 0)
    tr = result.get("total_redundant", 0)
    ts = result.get("total_synergy", 0)
    leak = result.get("info_leak", 0)
    total = tu + tr + ts
    safe_total = total if total > 0 else 1
    dominant = _dominant_component(result)

    # Check if asking about a specific agent.
    for agent in agents:
        if agent.lower() in q:
            u_val = unique.get(agent, 0)
            syn_pairs = {k: v for k, v in synergy.items() if agent in k}
            red_pairs = {k: v for k, v in redundant.items() if agent in k}
            top_syn = max(syn_pairs.values()) if syn_pairs else 0
            top_red = max(red_pairs.values()) if red_pairs else 0

            return (
                f"{agent} has a unique causal contribution of {u_val:.4f} bits "
                f"to {target}'s future. "
                f"Its strongest synergistic pairing contributes {top_syn:.4f} bits "
                f"and its strongest redundant pairing contributes {top_red:.4f} bits. "
                f"{'It is the most important unique agent.' if agent == _top_agent(unique)[0] else ''}"
            ).strip()

    # Leak questions.
    if _matches(q, ["leak", "unexplained", "hidden", "unobserved", "missing"]):
        if leak < 0.05:
            interpretation = (
                "The leak is very low. The observed agents explain "
                "almost all of the target's future behaviour."
            )
        elif leak < 0.3:
            interpretation = (
                "There is some information leak. The observed agents explain "
                "most of the target, but some influence comes from variables "
                "not included in the analysis."
            )
        else:
            interpretation = (
                "The leak is substantial. A large part of the target's future "
                "is driven by variables you have not included. Consider adding "
                "more agents to the analysis."
            )
        return (
            f"The information leak is {leak:.4f} ({leak:.1%}). "
            f"{interpretation}"
        )

    # Unique questions.
    if _matches(q, ["unique", "individual", "solo", "independent"]):
        best, val = _top_agent(unique)
        return (
            f"Total unique causality is {tu:.4f} bits "
            f"({tu/safe_total*100:.1f}% of explained information). "
            f"{best} has the highest unique contribution at {val:.4f} bits. "
            f"Unique causality means information about {target}'s future that "
            f"only that agent provides and no other agent can substitute."
        )

    # Redundancy questions.
    if _matches(q, ["redundan", "shared", "overlap", "duplicate", "same info"]):
        pair, val = _top_pair(redundant)
        return (
            f"Total redundancy is {tr:.4f} bits "
            f"({tr/safe_total*100:.1f}% of explained information). "
            f"The most redundant pair is {pair.replace('|', ' and ')} "
            f"at {val:.4f} bits. Redundancy means these agents carry "
            f"the same causal information about {target}, so knowing "
            f"one gives you little extra beyond what the other already tells you."
        )

    # Synergy questions.
    if _matches(q, ["synerg", "combined", "together", "joint", "interaction"]):
        pair, val = _top_pair(synergy)
        return (
            f"Total synergy is {ts:.4f} bits "
            f"({ts/safe_total*100:.1f}% of explained information). "
            f"The strongest synergistic pair is {pair.replace('|', ' and ')} "
            f"at {val:.4f} bits. Synergy means these agents only provide "
            f"causal information about {target} when observed together. "
            f"Neither alone carries that information."
        )

    # Summary / overview (after specific component questions).
    if _matches(q, ["summary", "overview", "results", "explain", "tell me",
                     "what happened", "what did you find", "what does it mean"]):
        best_agent, best_val = _top_agent(unique)
        return (
            f"I analysed how {len(agents)} agents causally influence {target}'s "
            f"future at lag tau={tau}. "
            f"The dominant component is {dominant} "
            f"({ts:.4f} bits synergy, {tr:.4f} bits redundancy, "
            f"{tu:.4f} bits unique). "
            f"{best_agent} has the highest unique contribution ({best_val:.4f} bits). "
            f"The information leak is {leak:.1%}, meaning "
            f"{'the agents explain almost everything' if leak < 0.1 else 'a substantial fraction of the target is driven by unmeasured variables' if leak > 0.3 else 'some of the target is driven by unmeasured variables'}."
        )

    # Why questions.
    if _matches(q, ["why", "reason", "cause", "how come"]):
        if "synergy" in q or "high" in q:
            pair, val = _top_pair(synergy)
            return (
                f"Synergy is {ts:.4f} bits ({ts/safe_total*100:.1f}% of the total). "
                f"The {pair.replace('|', ' and ')} pair contributes {val:.4f} bits. "
                f"This happens when agents interact nonlinearly to influence "
                f"{target}. Neither agent alone carries this information. "
                f"In the SURD paper, the synergistic collider (sin(q2*q3)) "
                f"is the textbook example of this pattern."
            )
        if "redundan" in q:
            pair, val = _top_pair(redundant)
            return (
                f"Redundancy is {tr:.4f} bits. The {pair.replace('|', ' and ')} "
                f"pair is the most redundant at {val:.4f} bits. This means these "
                f"agents carry overlapping causal information about {target}. "
                f"If they are measuring similar quantities, redundancy is expected."
            )
        if "leak" in q:
            return (
                f"The leak is {leak:.1%}. This means {leak*100:.0f}% of {target}'s "
                f"future cannot be predicted from the observed agents. Either "
                f"there are other variables influencing {target} that are not "
                f"included, or there is inherent randomness in the system."
            )
        return (
            f"The dominant component is {dominant}. "
            f"Total unique={tu:.4f}, redundant={tr:.4f}, synergy={ts:.4f}, "
            f"leak={leak:.1%}. Could you be more specific about which part "
            f"you want explained?"
        )

    # Parameters.
    if _matches(q, ["tau", "lag", "parameter", "setting", "bins", "config"]):
        return (
            f"The analysis was run with tau={tau} (time lag in steps) and "
            f"{meta.get('nbins', '?')} histogram bins per variable. "
            f"There are {meta.get('n_timepoints', '?')} usable time points "
            f"after applying the lag. Increasing tau looks further into "
            f"the future. More bins gives finer probability estimates but "
            f"needs more data to be reliable."
        )

    # Significance / p-value questions.
    if _matches(q, ["signif", "p-val", "p val", "permutation", "null hypoth",
                     "statistic", "by chance", "random chance"]):
        return (
            f"Statistical significance is tested through permutation: I shuffle "
            f"each agent's time series many times to break any real causal links, "
            f"then run SURD on the shuffled data to build a null distribution. "
            f"P-values measure how often the shuffled result exceeds the observed "
            f"value. Open the Significance tab and click 'Run permutation test' "
            f"to see the null distributions and p-values for the current results. "
            f"100 permutations gives reasonable estimates; more is slower but "
            f"more reliable."
        )

    # Method.
    if _matches(q, ["method", "surd", "algorithm", "how does it work", "what is surd"]):
        return (
            "SURD (Synergistic-Unique-Redundant Decomposition) was published "
            "by Martinez-Sanchez, Arranz, and Lozano-Duran in Nature "
            "Communications in 2024. It measures how much knowing the past "
            "of each agent reduces uncertainty about the target's future, "
            "then decomposes that causal information into unique, redundant, "
            "and synergistic components. It also estimates the information "
            "leak from unobserved variables. The algorithm works by computing "
            "specific mutual information for every subset of agents, sorting "
            "them, and assigning incremental gains to redundancy or synergy."
        )

    # Comparison / which is biggest.
    if _matches(q, ["compar", "biggest", "most", "dominant", "main", "primary"]):
        return (
            f"The dominant component is {dominant}. "
            f"Unique: {tu:.4f} bits ({tu/safe_total*100:.1f}%). "
            f"Redundant: {tr:.4f} bits ({tr/safe_total*100:.1f}%). "
            f"Synergy: {ts:.4f} bits ({ts/safe_total*100:.1f}%). "
            f"Information leak: {leak:.1%}."
        )

    # Fallback.
    return (
        f"I can answer questions about the SURD results. Try asking about "
        f"specific agents ({', '.join(agents)}), the synergy, redundancy, "
        f"unique contributions, the information leak, the parameters, "
        f"or ask for a summary of the results."
    )


# ── Public interface ──────────────────────────────────────

def get_chatbot_mode() -> str:
    """Return 'ollama' if available, otherwise 'rules'."""
    if _ollama_available():
        return "ollama"
    return "rules"


def chat(question: str, result: dict, mode: str = "auto") -> str:
    """Answer a question about SURD results.

    mode='auto' tries Ollama first, falls back to rules.
    mode='ollama' uses only Ollama (returns error if unavailable).
    mode='rules' uses only pattern matching.
    """
    if result is None:
        return "Run an analysis first and I can answer questions about the results."

    if not question.strip():
        return "Ask me something about the results."

    if mode == "auto":
        if _ollama_available():
            answer = _ask_ollama(question, result)
            if answer:
                return answer
        return _rule_based(question, result)

    if mode == "ollama":
        if not _ollama_available():
            return ("Ollama is not running. Start it with 'ollama serve' in a "
                    "terminal, or switch to rule-based mode.")
        answer = _ask_ollama(question, result)
        return answer if answer else "Ollama did not return an answer."

    return _rule_based(question, result)
