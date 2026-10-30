"""Create the midterm-report figures for the two standalone methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MIDTERM_METHODS = ("tfidf", "embedding")
METHOD_NAME = {"tfidf": "TF-IDF", "embedding": "Gemini embedding"}


def learning_curve(result: dict, output: Path) -> None:
    rows = [row for row in result["rows"] if row["method"] in MIDTERM_METHODS]
    proportions = sorted({int(row["proportion"]) for row in rows})
    fig, axis = plt.subplots(figsize=(6.6, 3.5))
    for method in MIDTERM_METHODS:
        means = []
        deviations = []
        for proportion in proportions:
            values = [float(row["macro_f1"]) for row in rows if row["method"] == method and int(row["proportion"]) == proportion]
            means.append(float(np.mean(values)))
            deviations.append(float(np.std(values, ddof=1)))
        axis.errorbar(proportions, means, yerr=deviations, marker="o", capsize=3, label=METHOD_NAME[method])
    axis.set_xlabel("Training-pool proportion (%)")
    axis.set_ylabel("Test macro-F1")
    axis.set_xticks(proportions)
    axis.set_ylim(0, 0.85)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.results.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "midterm_learning_curve.pdf"
    learning_curve(result, output)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
