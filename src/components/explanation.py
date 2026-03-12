"""Turns PID results into readable text — plain or technical."""


def generate_explanation(result: dict, mode: str = "plain") -> str:
    """Build a human-readable summary.

    Args:
        result: Full analysis result dict.
        mode: 'plain' for non-technical, 'technical' for more detail.
    """
    if mode == "technical":
        return _technical(result)
    return _plain(result)


def _plain(result: dict) -> str:
    """Non-technical explanation anyone can understand."""
    meta = result["meta"]
    unique = result["unique"]
    sources = meta["sources"]
    target = meta["target"]

    best_src = max(unique, key=unique.get)
    worst_src = min(unique, key=unique.get)

    total_unique = sum(unique.values())
    total = total_unique + result["redundant"] + result["synergy"]
    if total == 0:
        total = 1  # avoid division by zero

    syn_pct = result["synergy"] / total * 100
    red_pct = result["redundant"] / total * 100

    lines = [
        f"### What the analysis found\n",
        f"We looked at how **{len(sources)} variables** work together to "
        f"predict **{target}** ({meta['n_rows']} data points).\n",
        f"- **{best_src}** is the most useful source on its own "
        f"({unique[best_src]:.4f} bits of unique information).",
        f"- **{worst_src}** contributes the least unique information "
        f"({unique[worst_src]:.4f} bits).",
        f"- **{red_pct:.1f}%** of the information is redundant — "
        "meaning multiple sources tell you the same thing.",
        f"- **{syn_pct:.1f}%** is synergistic — "
        "information that only appears when sources are combined.",
    ]

    if result.get("pairwise_synergy"):
        pw = result["pairwise_synergy"]
        top_pair = max(pw, key=pw.get)
        a, b = top_pair.split("|")
        lines.append(
            f"- The pair **{a}** + **{b}** has the strongest synergy "
            f"({pw[top_pair]:.4f} bits)."
        )

    if result["warnings"]:
        lines.append("\n### Warnings")
        for w in result["warnings"]:
            lines.append(f"- {w}")

    return "\n".join(lines)


def _technical(result: dict) -> str:
    """More detailed explanation with formulas and method info."""
    meta = result["meta"]
    unique = result["unique"]
    sources = meta["sources"]
    target = meta["target"]
    method = result.get("method", "Unknown")

    total_unique = sum(unique.values())
    mi_total = total_unique + result["redundant"] + result["synergy"]

    lines = [
        f"### Technical summary\n",
        f"**Method:** {method}\n",
        f"**Variables:** {len(sources)} sources → target `{target}` "
        f"| {meta['n_rows']} rows | bins={meta['params']['bins']}\n",
        "**PID identity:**  "
        "I(X₁,X₂;Y) = Unique(X₁) + Unique(X₂) + Redundancy + Synergy\n",
        "**Computed values (bits):**\n",
    ]

    for s in sources:
        lines.append(f"- Unique({s}) = {unique[s]:.6f}")

    lines.append(f"- Redundancy = {result['redundant']:.6f}")
    lines.append(f"- Synergy = {result['synergy']:.6f}")
    lines.append(f"- **Total I(sources;target) ≈ {mi_total:.6f}**\n")

    lines.append(
        "**How it works:** Redundancy is measured using I_min — for each "
        "outcome y, we take the minimum specific information that either "
        "source provides, then average over all outcomes. Unique information "
        "is what remains after subtracting redundancy from each source's "
        "mutual information with the target. Synergy is whatever the joint "
        "mutual information has beyond the sum of unique and redundant parts."
    )

    if result.get("pairwise_synergy"):
        lines.append("\n**Pairwise results:**\n")
        for key, val in result["pairwise_synergy"].items():
            a, b = key.split("|")
            red_val = result.get("pairwise_redundancy", {}).get(key, 0)
            lines.append(f"- {a} + {b}: synergy={val:.6f}, redundancy={red_val:.6f}")

    if result["warnings"]:
        lines.append("\n**Warnings:**")
        for w in result["warnings"]:
            lines.append(f"- {w}")

    return "\n".join(lines)
