"""Shared helpers for serialisation and formatting."""

import json
import csv
import io


def result_to_json(result: dict) -> str:
    """Convert an analysis result dict to downloadable JSON text."""
    return json.dumps(result, indent=2)


def result_to_csv(result: dict) -> str:
    """Convert key numbers to a small CSV summary string."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["component", "source", "value_bits"])

    for src, val in result["unique"].items():
        writer.writerow(["unique", src, f"{val:.6f}"])

    writer.writerow(["redundant", "all", f"{result['redundant']:.6f}"])
    writer.writerow(["synergy", "all", f"{result['synergy']:.6f}"])

    if result.get("pairwise_synergy"):
        for pair, val in result["pairwise_synergy"].items():
            writer.writerow(["pairwise_synergy", pair, f"{val:.6f}"])

    if result.get("pairwise_redundancy"):
        for pair, val in result["pairwise_redundancy"].items():
            writer.writerow(["pairwise_redundancy", pair, f"{val:.6f}"])

    return buf.getvalue()
