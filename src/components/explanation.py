"""Turns SURD results into readable text — plain or technical mode."""


def generate_explanation(result: dict, mode: str = "plain") -> str:
    """Build a human-readable summary of the SURD results."""
    if mode == "technical":
        return _technical(result)
    return _plain(result)


def _plain(result: dict) -> str:
    """Non-technical explanation anyone can understand."""
    meta = result["meta"]
    unique = result["unique"]
    target = meta["target"]
    agents = meta["agent_names"]
    tau = meta["tau"]

    # Find the agent with the most unique causal contribution.
    if unique:
        best = max(unique, key=unique.get)
        worst = min(unique, key=unique.get)
    else:
        best = worst = "N/A"

    total = result["total_unique"] + result["total_redundant"] + result["total_synergy"]
    safe_total = total if total > 0 else 1

    syn_pct = result["total_synergy"] / safe_total * 100
    red_pct = result["total_redundant"] / safe_total * 100
    leak_pct = result["info_leak"] * 100

    lines = [
        f"### What the analysis found\n",
        f"I analysed how **{len(agents)} variables** causally influence the "
        f"future of **{target}** using a time lag of **{tau} step(s)** "
        f"({meta['n_timepoints']} usable time points).\n",
        f"- **{best}** has the strongest unique causal influence "
        f"({unique.get(best, 0):.4f} bits).",
        f"- **{worst}** has the weakest unique influence "
        f"({unique.get(worst, 0):.4f} bits).",
        f"- **{red_pct:.1f}%** of the causal information is redundant — "
        "meaning multiple agents carry the same predictive power.",
        f"- **{syn_pct:.1f}%** is synergistic — "
        "causal information that only appears when agents are combined.",
        f"- **{leak_pct:.1f}%** of the target's future is unexplained — "
        "suggesting hidden causes not included in the analysis.",
    ]

    # Highlight the top synergistic pair if available.
    syn = result.get("synergy_breakdown", {})
    if syn:
        top_pair = max(syn, key=syn.get)
        lines.append(
            f"- The strongest synergistic combination is **{top_pair.replace('|', ' + ')}** "
            f"({syn[top_pair]:.4f} bits)."
        )

    if result.get("warnings"):
        lines.append("\n### Warnings")
        for w in result["warnings"]:
            lines.append(f"- {w}")

    return "\n".join(lines)


def _technical(result: dict) -> str:
    """Detailed explanation with method info and exact values."""
    meta = result["meta"]
    unique = result["unique"]
    target = meta["target"]
    agents = meta["agent_names"]
    method = result.get("method", "Unknown")
    tau = meta["tau"]

    total = result["total_unique"] + result["total_redundant"] + result["total_synergy"]

    lines = [
        f"### Technical summary\n",
        f"**Method:** {method}\n",
        f"**Target:** {target} (future, t+{tau})\n",
        f"**Agents:** {', '.join(agents)} (past, t)\n",
        f"**Parameters:** tau={tau}, nbins={meta['nbins']}, "
        f"n_timepoints={meta['n_timepoints']}\n",
        "**SURD identity:**\n",
        "H(Q_j⁺) = ΔI_redundant + ΔI_unique + ΔI_synergistic + ΔI_leak\n",
        "**Unique causal contributions (bits):**\n",
    ]

    for name, val in unique.items():
        lines.append(f"- Unique({name} → {target}) = {val:.6f}")

    lines.append(f"\n**Redundant causal contributions (bits):**\n")
    for name, val in result.get("redundant_breakdown", {}).items():
        lines.append(f"- Redundant({name.replace('|', ', ')}) = {val:.6f}")

    lines.append(f"\n**Synergistic causal contributions (bits):**\n")
    for name, val in result.get("synergy_breakdown", {}).items():
        lines.append(f"- Synergy({name.replace('|', ', ')}) = {val:.6f}")

    lines.append(f"\n**Totals:**")
    lines.append(f"- Total unique = {result['total_unique']:.6f}")
    lines.append(f"- Total redundant = {result['total_redundant']:.6f}")
    lines.append(f"- Total synergy = {result['total_synergy']:.6f}")
    lines.append(f"- **I(Q_j⁺ ; Q) ≈ {total:.6f}**")
    lines.append(f"- Information leak = {result['info_leak']:.6f} "
                 f"({result['info_leak']*100:.1f}% of target entropy)\n")

    lines.append(
        "**How SURD works:** Causality is measured as the increments of "
        "information gained about the target's future from observing the "
        "agents' past. The algorithm computes specific mutual information "
        "for every subset of agents, sorts them, and assigns incremental "
        "gains to redundancy (when a single agent provides the increment) "
        "or synergy (when a combination is needed). The leak captures "
        "the fraction of the target's uncertainty that no observed agent "
        "can explain, indicating hidden or unmeasured causes."
    )

    if result.get("warnings"):
        lines.append("\n**Warnings:**")
        for w in result["warnings"]:
            lines.append(f"- {w}")

    return "\n".join(lines)
