"""Fixed-label evaluation and paired group bootstrap utilities."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], n_classes: int
) -> dict[str, Any]:
    labels = np.arange(n_classes, dtype=np.int64)
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    supports = np.bincount(np.asarray(y_true, dtype=np.int64), minlength=n_classes)
    return {
        "macro_f1": float(per_class.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_f1": [float(value) for value in per_class],
        "support": [int(value) for value in supports],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def paired_group_bootstrap_difference(
    y_true: Sequence[int],
    reference_pred: Sequence[int],
    comparison_pred: Sequence[int],
    group_ids: Sequence[object],
    n_classes: int,
    *,
    n_bootstrap: int = 2000,
    seed: int = 345,
    strata: Sequence[object] | None = None,
) -> dict[str, float]:
    """Bootstrap complete test groups, preserving paired predictions.

    If ``strata`` is supplied, each replicate samples the same number of
    groups independently inside every stratum.  The experiment passes outer
    fold ids here so the registered interval does not mix the three folds into
    one unrestricted group sample.
    """

    y_true = np.asarray(y_true, dtype=np.int64)
    reference_pred = np.asarray(reference_pred, dtype=np.int64)
    comparison_pred = np.asarray(comparison_pred, dtype=np.int64)
    group_ids = np.asarray(group_ids, dtype=object)
    if not (len(y_true) == len(reference_pred) == len(comparison_pred) == len(group_ids)):
        raise ValueError("bootstrap inputs must have equal lengths")
    if strata is None:
        strata_array = np.zeros(len(y_true), dtype=np.int64)
    else:
        strata_array = np.asarray(strata, dtype=object)
        if len(strata_array) != len(y_true):
            raise ValueError("strata must have the same length as the predictions")
    stratum_rows: list[list[np.ndarray]] = []
    for stratum in np.unique(strata_array):
        rows = np.flatnonzero(strata_array == stratum)
        unique_groups, inverse = np.unique(group_ids[rows], return_inverse=True)
        stratum_rows.append([rows[inverse == i] for i in range(len(unique_groups))])
    if not stratum_rows or any(not group_rows for group_rows in stratum_rows):
        raise ValueError("bootstrap requires at least one group per stratum")
    rng = np.random.default_rng(seed)
    differences = np.empty(n_bootstrap, dtype=np.float64)
    for replicate in range(n_bootstrap):
        selected_parts: list[np.ndarray] = []
        for group_rows in stratum_rows:
            selected_groups = rng.integers(0, len(group_rows), size=len(group_rows))
            selected_parts.extend(group_rows[i] for i in selected_groups)
        selected = np.concatenate(selected_parts)
        selected_strata = strata_array[selected]
        fold_differences = []
        for stratum in np.unique(selected_strata):
            fold_rows = selected[selected_strata == stratum]
            ref = f1_score(y_true[fold_rows], reference_pred[fold_rows], labels=np.arange(n_classes), average="macro", zero_division=0)
            cmp = f1_score(y_true[fold_rows], comparison_pred[fold_rows], labels=np.arange(n_classes), average="macro", zero_division=0)
            fold_differences.append(cmp - ref)
        differences[replicate] = float(np.mean(fold_differences))
    low, high = np.quantile(differences, [0.025, 0.975])
    observed_folds = []
    for stratum in np.unique(strata_array):
        rows = np.flatnonzero(strata_array == stratum)
        observed_folds.append(
            f1_score(y_true[rows], comparison_pred[rows], labels=np.arange(n_classes), average="macro", zero_division=0)
            - f1_score(y_true[rows], reference_pred[rows], labels=np.arange(n_classes), average="macro", zero_division=0)
        )
    observed = float(np.mean(observed_folds))
    return {"difference": float(observed), "ci_95_low": float(low), "ci_95_high": float(high)}
