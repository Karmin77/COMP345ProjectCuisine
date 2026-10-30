"""Create compact CSV/PNG summaries from an experiment results.json."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


def summarize(source: Path, output_dir: Path) -> None:
    with source.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    rows = result["rows"]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = result.get("summary", [])
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["proportion", "method", "mean_macro_f1", "std_macro_f1", "fold_scores"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    methods = sorted({row["method"] for row in rows})
    proportions = sorted({int(row["proportion"]) for row in rows})
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    for method in methods:
        means = []
        deviations = []
        for proportion in proportions:
            values = [float(row["macro_f1"]) for row in rows if row["method"] == method and int(row["proportion"]) == proportion]
            means.append(float(np.mean(values)) if values else np.nan)
            deviations.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        axis.errorbar(proportions, means, yerr=deviations, marker="o", capsize=3, label=method)
    axis.set_xlabel("Training-pool proportion (%)")
    axis.set_ylabel("Test macro-F1")
    axis.set_title("COMP 345 learning curves")
    axis.set_xticks(proportions)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=180)
    plt.close(fig)

    bootstrap = result.get("primary_bootstrap_embedding_minus_tfidf")
    if bootstrap:
        print(
            "100% embedding minus TF-IDF: "
            f"{bootstrap['difference']:.4f} "
            f"(95% group bootstrap [{bootstrap['ci_95_low']:.4f}, {bootstrap['ci_95_high']:.4f}])"
        )
    print(f"wrote {output_dir / 'summary.csv'}")
    print(f"wrote {output_dir / 'learning_curves.png'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize(args.results, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
