"""Run the scored TF-IDF arm when a Gemini cache is not yet available."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import group_summary
from .experiment import _config_dict, _index_labels, _record, _summary, _train_tfidf_arm, default_probe_candidates
from .metrics import classification_metrics
from .splits import PreparedDataset, SplitPlan, make_nested_subsets


def run_tfidf_baseline(
    dataset: PreparedDataset,
    split_plan: SplitPlan,
    output_dir: str | Path,
    *,
    seed: int = 345,
    proportions: tuple[int, ...] = (2, 5, 10, 20, 40, 60, 80, 100),
    quick: bool = False,
) -> dict[str, Any]:
    """Evaluate TF-IDF over all outer folds without making embedding claims."""

    if any(value not in split_plan.proportions for value in proportions):
        raise ValueError("requested proportions are not present in the split manifest")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    truths: dict[str, np.ndarray] = {}
    groups: dict[str, np.ndarray] = {}
    for outer in split_plan.folds:
        subsets = make_nested_subsets(
            outer.train_pool,
            dataset.labels,
            dataset.groups,
            proportions,
            seed=seed + 10007 * (outer.fold + 1),
        )
        for proportion in proportions:
            train_indices = subsets[proportion]
            option, _, (_, validation_x, test_x), tuned = _train_tfidf_arm(
                train_indices,
                outer.validation,
                outer.test,
                dataset,
                candidates=default_probe_candidates(seed + 100000 * (outer.fold + 1) + proportion, quick=quick),
            )
            test_probabilities = tuned.probe.predict_proba(test_x)
            test_predictions = np.argmax(test_probabilities, axis=1).astype(np.int64)
            metrics = classification_metrics(
                _index_labels(dataset, outer.test), test_predictions, dataset.label_space.n_classes
            )
            rows.append(
                _record(
                    fold=outer.fold,
                    proportion=proportion,
                    method="tfidf",
                    metrics=metrics,
                    validation_macro_f1=tuned.validation_macro_f1,
                    metadata={"tfidf_analyzer": option, "probe_config": _config_dict(tuned.config)},
                )
            )
            key = f"fold{outer.fold}_p{proportion}_tfidf"
            predictions[key] = test_predictions
            probabilities[key] = test_probabilities
            truths[key] = _index_labels(dataset, outer.test)
            groups[key] = dataset.groups[outer.test]
    result: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "methods": ["tfidf"],
        "label_space": list(dataset.label_space.classes),
        "dataset_summary": group_summary(dataset.recipes),
        "split_plan": split_plan.to_dict(),
        "rows": rows,
        "summary": _summary(rows),
        "note": "TF-IDF-only baseline; no embedding or fusion claim is made.",
    }
    with (destination / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    np.savez_compressed(
        destination / "predictions.npz",
        **{f"pred__{key}": value for key, value in predictions.items()},
        **{f"prob__{key}": value for key, value in probabilities.items()},
        **{f"truth__{key}": value for key, value in truths.items()},
        **{f"group__{key}": value.astype(str) for key, value in groups.items()},
    )
    return result
