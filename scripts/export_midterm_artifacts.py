"""Export the midterm subset of saved results: the TF-IDF and Gemini arms only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

MIDTERM_METHODS = ("tfidf", "embedding")
ERROR_CASE_COLUMNS = (
    "fold",
    "dataset_index",
    "recipe_id",
    "true_class",
    "ingredients",
    "normalized_group",
    "tfidf_pred",
    "embedding_pred",
    "tfidf_confidence",
    "embedding_confidence",
)
MIDTERM_OPTIONS = ("seed", "quick", "proportions", "bootstrap_replicates")


def export_results(source: Path, output: Path) -> None:
    with source.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    result["rows"] = [row for row in result["rows"] if row["method"] in MIDTERM_METHODS]
    result["summary"] = [row for row in result["summary"] if row["method"] in MIDTERM_METHODS]
    result["options"] = {key: result["options"][key] for key in MIDTERM_OPTIONS if key in result["options"]}
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


def filter_rows(source: Path, output: Path) -> None:
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if row["method"] in MIDTERM_METHODS]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_columns(source: Path, output: Path) -> None:
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = [{key: row[key] for key in ERROR_CASE_COLUMNS} for row in csv.DictReader(handle)]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ERROR_CASE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="full run output, e.g. artifacts/results")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    failure = args.results_dir / "failure-analysis"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    export_results(args.results_dir / "results.json", args.output_dir / "results.json")
    filter_rows(failure / "per_class_100.csv", args.output_dir / "per_class_100.csv")
    for method in MIDTERM_METHODS:
        name = f"confusion_{method}_100.csv"
        shutil.copyfile(failure / name, args.output_dir / name)
    select_columns(failure / "error_cases_100.csv", args.output_dir / "error_cases_100.csv")
    for path in sorted(args.output_dir.iterdir()):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
