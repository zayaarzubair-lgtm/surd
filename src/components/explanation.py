"""Turns SURD results into a plain-English paragraph a student can read"""


def generate_explanation(result: dict) -> str:
    """Build a human-readable summary of the analysis results.

    Args:
        result: The full analysis result dict from run_analysis().

    Returns:
        A multi-line explanation string.
    """
    meta = result["meta"]
    unique = result["unique"]
    sources = meta["sources"]
    target = meta["target"]

    # Find the most and least informative source.
    best_src = max(unique, key=unique.get)
    worst_src = min(unique, key=unique.get)

    lines = [
        f"**Analysis of {len(sources)} source(s) predicting `{target}`** "
        f"({meta['n_rows']} rows, {meta['n_cols']} columns after cleaning)\n",

        f"- **Most informative source:** `{best_src}` "
        f"(unique information = {unique[best_src]:.4f} bits).",

        f"- **Least informative source:** `{worst_src}` "
        f"(unique information = {unique[worst_src]:.4f} bits).",

        f"- **Redundancy across sources:** {result['redundant']:.4f} bits — "
        "this is information shared by multiple sources.",

        f"- **Synergy:** {result['synergy']:.4f} bits — "
        "this is extra information that only appears when sources are combined.",
    ]

    if result["pairwise_synergy"]:
        pw = result["pairwise_synergy"]
        top_pair = max(pw, key=pw.get)
        a, b = top_pair.split("|")
        lines.append(
            f"- **Strongest pairwise synergy:** `{a}` + `{b}` "
            f"({pw[top_pair]:.4f} bits)."
        )

    # Append any warnings.
    if result["warnings"]:
        lines.append("\n**Warnings:**")
        for w in result["warnings"]:
            lines.append(f"  - {w}")

    return "\n".join(lines)
