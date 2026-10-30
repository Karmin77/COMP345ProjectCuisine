"""Generate reproducible per-class metrics, confusion matrices, and error cases.

The script consumes the official full experiment artifacts only.  It does not
call Gemini or retrain any model, so it is safe to run after the API key has
been removed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from comp345.data import load_recipes  # noqa: E402
from comp345.splits import PreparedDataset, SplitPlan, make_nested_subsets  # noqa: E402


METHODS = ("tfidf", "embedding", "probability_fusion", "feature_fusion")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _top_pairs(matrix: np.ndarray, labels: list[str], limit: int = 12) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for true_id in range(matrix.shape[0]):
        for predicted_id in range(matrix.shape[1]):
            if true_id != predicted_id and int(matrix[true_id, predicted_id]):
                pairs.append(
                    {
                        "true_class": labels[true_id],
                        "predicted_class": labels[predicted_id],
                        "count": int(matrix[true_id, predicted_id]),
                    }
                )
    return sorted(pairs, key=lambda row: (-row["count"], row["true_class"], row["predicted_class"]))[:limit]


def _group_class_counts(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, n_classes: int) -> np.ndarray:
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    counts = np.zeros((len(unique_groups), 3 * n_classes), dtype=np.float64)
    rows = np.arange(len(y))
    np.add.at(counts, (inverse, y), 1.0)
    np.add.at(counts, (inverse, n_classes + pred), 1.0)
    correct = y == pred
    np.add.at(counts, (inverse[correct], 2 * n_classes + y[correct]), 1.0)
    return counts


def _macro_f1_from_group_counts(counts: np.ndarray, n_classes: int) -> np.ndarray:
    true_support = counts[:, :n_classes]
    pred_support = counts[:, n_classes : 2 * n_classes]
    true_positive = counts[:, 2 * n_classes :]
    denominator = true_support + pred_support
    per_class = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return per_class.mean(axis=1)


def _fast_group_bootstrap(
    y: np.ndarray,
    reference: np.ndarray,
    comparison: np.ndarray,
    groups: np.ndarray,
    strata: np.ndarray,
    n_classes: int,
    *,
    n_bootstrap: int = 2000,
    seed: int = 345,
) -> dict[str, float]:
    """Bootstrap paired macro-F1 using sufficient statistics per group."""
    fold_counts: list[tuple[np.ndarray, np.ndarray]] = []
    observed: list[float] = []
    for stratum in np.unique(strata):
        rows = np.flatnonzero(strata == stratum)
        ref_counts = _group_class_counts(y[rows], reference[rows], groups[rows], n_classes)
        cmp_counts = _group_class_counts(y[rows], comparison[rows], groups[rows], n_classes)
        fold_counts.append((ref_counts, cmp_counts))
        observed.append(
            float(_macro_f1_from_group_counts(ref_counts.sum(axis=0, keepdims=True), n_classes)[0])
        )
    # Replace the per-method observation above with paired differences.
    observed_differences = []
    for ref_counts, cmp_counts in fold_counts:
        ref_score = _macro_f1_from_group_counts(ref_counts.sum(axis=0, keepdims=True), n_classes)[0]
        cmp_score = _macro_f1_from_group_counts(cmp_counts.sum(axis=0, keepdims=True), n_classes)[0]
        observed_differences.append(float(cmp_score - ref_score))

    rng = np.random.default_rng(seed)
    differences = np.zeros(n_bootstrap, dtype=np.float64)
    chunk_size = 32
    for ref_counts, cmp_counts in fold_counts:
        n_groups = len(ref_counts)
        fold_differences = np.empty(n_bootstrap, dtype=np.float64)
        for start in range(0, n_bootstrap, chunk_size):
            stop = min(n_bootstrap, start + chunk_size)
            sampled = rng.integers(0, n_groups, size=(stop - start, n_groups))
            weights = np.zeros((stop - start, n_groups), dtype=np.float32)
            for offset, selected in enumerate(sampled):
                weights[offset] = np.bincount(selected, minlength=n_groups)
            ref_scores = _macro_f1_from_group_counts(weights @ ref_counts, n_classes)
            cmp_scores = _macro_f1_from_group_counts(weights @ cmp_counts, n_classes)
            fold_differences[start:stop] = cmp_scores - ref_scores
        differences += fold_differences / len(fold_counts)
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "difference": float(np.mean(observed_differences)),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "n_bootstrap": int(n_bootstrap),
    }


def analyze(results_path: Path, data_path: Path, splits_path: Path, predictions_path: Path, output_dir: Path) -> None:
    result = json.loads(results_path.read_text(encoding="utf-8"))
    recipes = load_recipes(data_path)
    labels = list(result["label_space"])
    label_to_id = {label: i for i, label in enumerate(labels)}
    output_dir.mkdir(parents=True, exist_ok=True)

    split_manifest = json.loads(splits_path.read_text(encoding="utf-8"))
    dataset = PreparedDataset.from_recipes(recipes)
    split_plan = SplitPlan.load(splits_path, dataset)
    predictions = np.load(predictions_path, allow_pickle=False)
    n_classes = len(labels)

    # The official rows already contain per-fold metrics.  Aggregate them
    # without recomputing model predictions, preserving the registered folds.
    metric_rows: list[dict[str, Any]] = []
    rows_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for row in result["rows"]:
        if int(row["proportion"]) != 100:
            continue
        method = str(row["method"])
        if method not in rows_by_method:
            continue
        rows_by_method[method].append(row)
        for class_id, label in enumerate(labels):
            metric_rows.append(
                {
                    "method": method,
                    "class": label,
                    "fold": int(row["fold"]),
                    "support": int(row["support"][class_id]),
                    "f1": float(row["per_class_f1"][class_id]),
                }
            )

    per_class_rows: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = rows_by_method[method]
        for class_id, label in enumerate(labels):
            values = np.asarray([row["per_class_f1"][class_id] for row in method_rows], dtype=float)
            supports = np.asarray([row["support"][class_id] for row in method_rows], dtype=int)
            per_class_rows.append(
                {
                    "method": method,
                    "class": label,
                    "support_total": int(supports.sum()),
                    "support_mean": float(supports.mean()),
                    "mean_f1": float(values.mean()),
                    "std_f1": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                }
            )
    _write_csv(
        output_dir / "per_class_100.csv",
        per_class_rows,
        ["method", "class", "support_total", "support_mean", "mean_f1", "std_f1"],
    )

    matrices: dict[str, np.ndarray] = {method: np.zeros((n_classes, n_classes), dtype=np.int64) for method in METHODS}
    case_rows: list[dict[str, Any]] = []
    for fold_item in split_manifest["folds"]:
        fold = int(fold_item["fold"])
        test_indices = np.asarray(fold_item["test"], dtype=np.int64)
        key_prefix = f"fold{fold}_p100_"
        truth = np.asarray(predictions[f"truth__{key_prefix}tfidf"], dtype=np.int64)
        if len(truth) != len(test_indices):
            raise ValueError(f"fold {fold} prediction length does not match split manifest")
        method_pred: dict[str, np.ndarray] = {}
        method_prob: dict[str, np.ndarray] = {}
        for method in METHODS:
            method_pred[method] = np.asarray(predictions[f"pred__{key_prefix}{method}"], dtype=np.int64)
            method_prob[method] = np.asarray(predictions[f"prob__{key_prefix}{method}"], dtype=float)
            if len(method_pred[method]) != len(test_indices):
                raise ValueError(f"fold {fold} {method} prediction length mismatch")
            matrices[method] += np.asarray(
                __import__("sklearn").metrics.confusion_matrix(truth, method_pred[method], labels=np.arange(n_classes)),
                dtype=np.int64,
            )

        for position, dataset_index in enumerate(test_indices):
            recipe = recipes[int(dataset_index)]
            true_id = int(truth[position])
            predictions_for_row = {method: int(method_pred[method][position]) for method in METHODS}
            correct = {method: predictions_for_row[method] == true_id for method in METHODS}
            if correct["feature_fusion"] and not correct["tfidf"] and not correct["embedding"]:
                category = "feature_fusion_rescue"
            elif correct["probability_fusion"] and not correct["tfidf"] and not correct["embedding"]:
                category = "probability_fusion_rescue"
            elif correct["embedding"] and not correct["tfidf"] and not correct["feature_fusion"]:
                category = "embedding_only_rescue"
            elif correct["tfidf"] and not correct["embedding"] and not correct["feature_fusion"]:
                category = "tfidf_only_rescue"
            elif not any(correct.values()):
                category = "all_methods_fail"
            elif correct["tfidf"] and correct["embedding"] and not correct["feature_fusion"] and not correct["probability_fusion"]:
                category = "fusion_regression"
            else:
                category = "other"
            row: dict[str, Any] = {
                "fold": fold,
                "dataset_index": int(dataset_index),
                "recipe_id": int(recipe.recipe_id),
                "true_class": labels[true_id],
                "ingredients": " | ".join(recipe.ingredients),
                "normalized_group": recipe.group_key,
                "category": category,
            }
            for method in METHODS:
                row[f"{method}_pred"] = labels[predictions_for_row[method]]
                row[f"{method}_confidence"] = float(np.max(method_prob[method][position]))
            case_rows.append(row)

    matrix_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    confusion_summary: dict[str, list[dict[str, Any]]] = {}
    for method, matrix in matrices.items():
        columns = ["method", "true_class"] + labels
        for true_id, label in enumerate(labels):
            matrix_rows.append({"method": method, "true_class": label, **{labels[p]: int(matrix[true_id, p]) for p in range(n_classes)}})
            denominator = matrix[true_id].sum()
            normalized = matrix[true_id] / denominator if denominator else np.zeros(n_classes)
            normalized_rows.append({"method": method, "true_class": label, **{labels[p]: float(normalized[p]) for p in range(n_classes)}})
        confusion_summary[method] = _top_pairs(matrix, labels)
        _write_csv(output_dir / f"confusion_{method}_100.csv", [row for row in matrix_rows if row["method"] == method], columns)
        _write_csv(output_dir / f"confusion_{method}_100_normalized.csv", [row for row in normalized_rows if row["method"] == method], columns)

    case_fields = [
        "fold", "dataset_index", "recipe_id", "true_class", "ingredients", "normalized_group", "category",
        *[f"{method}_pred" for method in METHODS],
        *[f"{method}_confidence" for method in METHODS],
    ]
    _write_csv(output_dir / "error_cases_100.csv", case_rows, case_fields)

    # Keep a compact, report-ready subset while preserving the full case file.
    selected: list[dict[str, Any]] = []
    categories = ("feature_fusion_rescue", "probability_fusion_rescue", "embedding_only_rescue", "tfidf_only_rescue", "fusion_regression", "all_methods_fail")
    category_method = {
        "feature_fusion_rescue": "feature_fusion",
        "probability_fusion_rescue": "probability_fusion",
        "embedding_only_rescue": "embedding",
        "tfidf_only_rescue": "tfidf",
        "fusion_regression": "probability_fusion",
        "all_methods_fail": "feature_fusion",
    }
    for category in categories:
        candidates = [row for row in case_rows if row["category"] == category]
        successful_method = category_method[category]
        candidates.sort(key=lambda row: (-float(row[f"{successful_method}_confidence"]), int(row["recipe_id"])))
        selected.extend(candidates[:8])
    _write_csv(output_dir / "representative_cases_100.csv", selected, case_fields)

    # Calculate supplementary paired intervals for fusion versus TF-IDF.
    # These are exploratory contrasts; the preregistered primary comparison
    # remains the embedding-minus-TF-IDF interval in results.json.
    bootstrap_inputs: dict[str, list[np.ndarray]] = {name: [] for name in ("y", "tfidf", "embedding", "probability_fusion", "feature_fusion", "groups", "folds")}
    for fold in range(len(split_plan.folds)):
        prefix = f"fold{fold}_p100_"
        bootstrap_inputs["y"].append(np.asarray(predictions[f"truth__{prefix}tfidf"], dtype=np.int64))
        for method in METHODS:
            bootstrap_inputs[method].append(np.asarray(predictions[f"pred__{prefix}{method}"], dtype=np.int64))
        bootstrap_inputs["groups"].append(np.asarray(predictions[f"group__{prefix}tfidf"]))
        bootstrap_inputs["folds"].append(np.full(len(bootstrap_inputs["y"][-1]), fold, dtype=np.int64))
    combined = {key: np.concatenate(value) for key, value in bootstrap_inputs.items()}
    supplementary_bootstrap = {
        method: _fast_group_bootstrap(
            combined["y"],
            combined["tfidf"],
            combined[method],
            combined["groups"],
            combined["folds"],
            n_classes,
        )
        for method in ("probability_fusion", "feature_fusion")
    }

    # Record actual per-class training support in the exploratory 2/5/10%
    # subsets so the low-data results are interpreted with their label counts.
    training_support_rows: list[dict[str, Any]] = []
    low_data_summary: dict[str, Any] = {}
    for outer in split_plan.folds:
        subsets = make_nested_subsets(
            outer.train_pool,
            dataset.labels,
            dataset.groups,
            split_plan.proportions,
            seed=int(result["seed"]) + 10007 * (outer.fold + 1),
        )
        for proportion, indices in subsets.items():
            support = np.bincount(dataset.labels[indices], minlength=n_classes)
            low_data_summary.setdefault(str(proportion), {"missing_class_folds": 0, "minimum_class_support": None})
            low_data_summary[str(proportion)]["missing_class_folds"] += int(np.count_nonzero(support == 0) > 0)
            minimum = int(support.min())
            previous = low_data_summary[str(proportion)]["minimum_class_support"]
            low_data_summary[str(proportion)]["minimum_class_support"] = minimum if previous is None else min(previous, minimum)
            for class_id, label in enumerate(labels):
                training_support_rows.append({"fold": outer.fold, "proportion": proportion, "class": label, "training_support": int(support[class_id])})
    _write_csv(output_dir / "training_class_support.csv", training_support_rows, ["fold", "proportion", "class", "training_support"])

    summary = {
        "dataset": result["dataset_summary"],
        "labels": labels,
        "method_macro_f1": {
            method: {
                "mean": float(np.mean([row["macro_f1"] for row in rows_by_method[method]])),
                "std": float(np.std([row["macro_f1"] for row in rows_by_method[method]], ddof=1)),
                "mean_accuracy": float(np.mean([row["accuracy"] for row in rows_by_method[method]])),
            }
            for method in METHODS
        },
        "category_counts": {category: sum(row["category"] == category for row in case_rows) for category in categories},
        "top_confusions": confusion_summary,
        "supplementary_paired_bootstrap_vs_tfidf": supplementary_bootstrap,
        "training_support_by_proportion": low_data_summary,
    }
    (output_dir / "failure_analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Failure analysis (100% training proportion)",
        "",
        f"The analysis uses {result['dataset_summary']['recipes']:,} labelled recipes and the three official outer test folds. It reads the saved predictions and does not retrain models.",
        "",
        "## Method-level results",
        "",
        "| Method | Mean macro-F1 | Fold SD | Mean accuracy |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        item = summary["method_macro_f1"][method]
        lines.append(f"| {method} | {item['mean']:.4f} | {item['std']:.4f} | {item['mean_accuracy']:.4f} |")
    lines.extend(["", "Exploratory paired group-bootstrap intervals versus TF-IDF:", ""])
    for method, item in supplementary_bootstrap.items():
        lines.append(f"- {method}: difference {item['difference']:.4f}, 95% interval [{item['ci_95_low']:.4f}, {item['ci_95_high']:.4f}] ({item['n_bootstrap']} replicates).")
    lines.append("These fusion contrasts are supplemental; the preregistered primary contrast is embedding minus TF-IDF.")
    lines.extend(["", "## Error-category counts", "", "| Category | Count |", "|---|---:|"])
    for category, count in summary["category_counts"].items():
        lines.append(f"| {category} | {count:,} |")
    lines.extend(["", "## Largest feature-fusion confusions", "", "| True class | Predicted class | Count |", "|---|---|---:|"])
    for pair in confusion_summary["feature_fusion"]:
        lines.append(f"| {pair['true_class']} | {pair['predicted_class']} | {pair['count']:,} |")
    lines.extend(["", "## Per-class macro-F1", "", "The full machine-readable table is in `per_class_100.csv`; the table below reports the mean across outer folds.", "", "| Class | TF-IDF | Embedding | Probability fusion | Feature fusion |", "|---|---:|---:|---:|---:|"])
    for label in labels:
        values = {row["method"]: row["mean_f1"] for row in per_class_rows if row["class"] == label}
        lines.append(f"| {label} | {values['tfidf']:.3f} | {values['embedding']:.3f} | {values['probability_fusion']:.3f} | {values['feature_fusion']:.3f} |")
    lines.extend(["", "## Representative cases", "", "See `representative_cases_100.csv` for compact examples and `error_cases_100.csv` for every test recipe with a category assignment."])
    (output_dir / "failure_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Optional visualization; the numerical artifacts above remain available
    # if Matplotlib is not installed in a fresh checkout.
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for axis, method in zip(axes.ravel(), METHODS):
            matrix = matrices[method].astype(float)
            axis.imshow(matrix, interpolation="nearest", cmap="Blues")
            axis.set_title(method.replace("_", " "))
            axis.set_xlabel("Predicted class")
            axis.set_ylabel("True class")
            axis.set_xticks(range(n_classes), labels, rotation=90, fontsize=6)
            axis.set_yticks(range(n_classes), labels, fontsize=6)
        fig.tight_layout()
        fig.savefig(output_dir / "confusion_matrices_100.png", dpi=180)
        plt.close(fig)
    except ImportError:
        pass

    print(f"wrote failure analysis to {output_dir}")
    print(json.dumps(summary["category_counts"], ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    analyze(args.results, args.data, args.splits, args.predictions, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
