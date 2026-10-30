"""End-to-end implementation of the Version 2 proposal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from .data import Recipe, group_summary
from .labels import LabelSpace
from .metrics import classification_metrics, paired_group_bootstrap_difference
from .model import FeatureBlocks, ProbeConfig, TorchLinearProbe, TunedProbe, l2_normalize_rows, tune_probe
from .splits import DEFAULT_PROPORTIONS, OuterFold, PreparedDataset, SplitPlan, make_nested_subsets
from .tfidf import ANALYZER_OPTIONS, fit_transform_pair


@dataclass(frozen=True)
class ExperimentOptions:
    seed: int = 345
    quick: bool = False
    proportions: tuple[int, ...] = DEFAULT_PROPORTIONS
    bootstrap_replicates: int = 2000
    skip_probability_fusion: bool = False
    skip_feature_fusion: bool = False


def default_probe_candidates(seed: int, *, quick: bool = False) -> list[ProbeConfig]:
    """Return the small registered validation grid used by the runner."""

    if quick:
        return [ProbeConfig(learning_rate=3e-3, l2=1e-4, batch_size=256, max_epochs=8, patience=2, seed=seed)]
    candidates: list[ProbeConfig] = []
    for learning_rate in (3e-3, 1e-2):
        for l2 in (1e-5, 1e-3):
            candidates.append(
                ProbeConfig(
                    learning_rate=learning_rate,
                    l2=l2,
                    batch_size=512,
                    max_epochs=60,
                    patience=8,
                    seed=seed,
                )
            )
    return candidates


def _config_dict(config: ProbeConfig) -> dict[str, Any]:
    return asdict(config)


def _index_labels(dataset: PreparedDataset, indices: Sequence[int]) -> np.ndarray:
    return dataset.labels[np.asarray(indices, dtype=np.int64)]


def _train_embedding_arm(
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    embedding_matrix: np.ndarray,
    dataset: PreparedDataset,
    *,
    candidates: Sequence[ProbeConfig],
) -> TunedProbe:
    return tune_probe(
        embedding_matrix[train_indices],
        _index_labels(dataset, train_indices),
        embedding_matrix[validation_indices],
        _index_labels(dataset, validation_indices),
        n_classes=dataset.label_space.n_classes,
        candidates=candidates,
    )


def _train_tfidf_arm(
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    dataset: PreparedDataset,
    *,
    candidates: Sequence[ProbeConfig],
) -> tuple[str, Any, Any, TunedProbe]:
    """Select the analyzer and probe using only the current validation set."""

    best: tuple[str, Any, Any, TunedProbe] | None = None
    documents = dataset.texts
    train_docs = [documents[int(index)] for index in train_indices]
    validation_docs = [documents[int(index)] for index in validation_indices]
    test_docs = [documents[int(index)] for index in test_indices]
    y_train = _index_labels(dataset, train_indices)
    y_validation = _index_labels(dataset, validation_indices)
    for option in ANALYZER_OPTIONS:
        vectorizer, train_x, (validation_x, test_x) = fit_transform_pair(
            train_docs, validation_docs, test_docs, option=option
        )
        tuned = tune_probe(
            train_x,
            y_train,
            validation_x,
            y_validation,
            n_classes=dataset.label_space.n_classes,
            candidates=candidates,
        )
        if best is None or tuned.validation_macro_f1 > best[3].validation_macro_f1:
            best = (option, vectorizer, (train_x, validation_x, test_x), tuned)
    assert best is not None
    return best


def _inner_folds(indices: np.ndarray, dataset: PreparedDataset, seed: int) -> list[np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    n_groups = len(np.unique(dataset.groups[indices]))
    if n_groups < 2:
        return [indices]
    n_splits = min(3, n_groups)
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [indices[test] for _, test in splitter.split(np.zeros(len(indices)), dataset.labels[indices], dataset.groups[indices])]
    except (ValueError, RuntimeError):
        splitter = GroupKFold(n_splits=n_splits)
        return [indices[test] for _, test in splitter.split(np.zeros(len(indices)), dataset.labels[indices], dataset.groups[indices])]


def _oof_base_probabilities(
    train_indices: np.ndarray,
    dataset: PreparedDataset,
    embeddings: np.ndarray,
    *,
    analyzer: str,
    tfidf_config: ProbeConfig,
    embedding_config: ProbeConfig,
    seed: int,
) -> np.ndarray:
    """Generate leakage-safe, complete OOF probabilities for stacking."""

    folds = _inner_folds(train_indices, dataset, seed)
    if len(folds) == 1:
        # A one-group smoke dataset cannot support stacking.  This branch is
        # explicit so production runs never silently train on same-row scores.
        raise ValueError("probability fusion requires at least two internal groups")
    oof = np.zeros((len(train_indices), 2 * dataset.label_space.n_classes), dtype=np.float32)
    position = {int(index): i for i, index in enumerate(train_indices)}
    documents = dataset.texts
    for inner_id, held_out in enumerate(folds):
        held_set = set(int(index) for index in held_out)
        inner_train = np.asarray([index for index in train_indices if int(index) not in held_set], dtype=np.int64)
        if not len(inner_train):
            raise ValueError("internal stacking fold left no training examples")
        train_docs = [documents[int(index)] for index in inner_train]
        held_docs = [documents[int(index)] for index in held_out]
        vectorizer, train_x, (held_x,) = fit_transform_pair(train_docs, held_docs, option=analyzer)
        tfidf_probe = TorchLinearProbe(train_x.shape[1], dataset.label_space.n_classes, tfidf_config)
        tfidf_probe.fit(train_x, _index_labels(dataset, inner_train))
        emb_probe = TorchLinearProbe(embeddings.shape[1], dataset.label_space.n_classes, embedding_config)
        emb_probe.fit(embeddings[inner_train], _index_labels(dataset, inner_train))
        tfidf_probabilities = tfidf_probe.predict_proba(held_x)
        embedding_probabilities = emb_probe.predict_proba(embeddings[held_out])
        for row, index in enumerate(held_out):
            oof[position[int(index)]] = np.concatenate((tfidf_probabilities[row], embedding_probabilities[row]))
    if not np.isfinite(oof).all() or np.any(oof.sum(axis=1) <= 0):
        raise RuntimeError("internal stacking produced invalid probabilities")
    return oof


def _record(
    *,
    fold: int,
    proportion: int,
    method: str,
    metrics: Mapping[str, Any],
    validation_macro_f1: float,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "fold": fold,
        "proportion": proportion,
        "method": method,
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "per_class_f1": metrics["per_class_f1"],
        "support": metrics["support"],
        "confusion_matrix": metrics["confusion_matrix"],
        "validation_macro_f1": validation_macro_f1,
    }
    if metadata:
        result.update(metadata)
    return result


def _summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((int(row["proportion"]), str(row["method"])), []).append(float(row["macro_f1"]))
    return [
        {
            "proportion": proportion,
            "method": method,
            "mean_macro_f1": float(np.mean(values)),
            "std_macro_f1": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "fold_scores": values,
        }
        for (proportion, method), values in sorted(grouped.items())
    ]


def run_experiment(
    dataset: PreparedDataset,
    split_plan: SplitPlan,
    embeddings: np.ndarray,
    output_dir: str | Path,
    *,
    options: ExperimentOptions = ExperimentOptions(),
) -> dict[str, Any]:
    """Run all registered folds/proportions and write auditable artifacts."""

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape[0] != dataset.n_samples or embeddings.ndim != 2:
        raise ValueError("embedding matrix must have one finite row per recipe")
    if not np.isfinite(embeddings).all():
        raise ValueError("embedding matrix contains NaN or infinite values")
    requested_proportions = tuple(options.proportions)
    if any(value not in split_plan.proportions for value in requested_proportions):
        raise ValueError("requested experiment proportions are not present in the split manifest")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    prediction_store: dict[str, np.ndarray] = {}
    probability_store: dict[str, np.ndarray] = {}
    truth_store: dict[str, np.ndarray] = {}
    groups_store: dict[str, np.ndarray] = {}

    for outer in split_plan.folds:
        subsets = make_nested_subsets(
            outer.train_pool,
            dataset.labels,
            dataset.groups,
            requested_proportions,
            seed=options.seed + 10007 * (outer.fold + 1),
        )
        validation_indices = outer.validation
        test_indices = outer.test
        for proportion in requested_proportions:
            train_indices = subsets[proportion]
            base_seed = options.seed + 100000 * (outer.fold + 1) + proportion
            candidates = default_probe_candidates(base_seed, quick=options.quick)
            tfidf_option, vectorizer, (train_x, validation_x, test_x), tfidf_tuned = _train_tfidf_arm(
                train_indices,
                validation_indices,
                test_indices,
                dataset,
                candidates=candidates,
            )
            embedding_tuned = _train_embedding_arm(
                train_indices,
                validation_indices,
                embeddings,
                dataset,
                candidates=[replace_seed(candidate, base_seed + 7) for candidate in candidates],
            )
            y_test = _index_labels(dataset, test_indices)
            y_validation = _index_labels(dataset, validation_indices)
            test_groups = dataset.groups[test_indices]

            method_predictions: dict[str, np.ndarray] = {}
            method_probabilities: dict[str, np.ndarray] = {}
            tfidf_test_probabilities = tfidf_tuned.probe.predict_proba(test_x)
            embedding_test_probabilities = embedding_tuned.probe.predict_proba(embeddings[test_indices])
            tfidf_test_pred = np.argmax(tfidf_test_probabilities, axis=1).astype(np.int64)
            embedding_test_pred = np.argmax(embedding_test_probabilities, axis=1).astype(np.int64)
            tfidf_validation_pred = tfidf_tuned.probe.predict(validation_x)
            embedding_validation_pred = embedding_tuned.probe.predict(embeddings[validation_indices])
            tfidf_metrics = classification_metrics(y_test, tfidf_test_pred, dataset.label_space.n_classes)
            embedding_metrics = classification_metrics(y_test, embedding_test_pred, dataset.label_space.n_classes)
            rows.append(_record(fold=outer.fold, proportion=proportion, method="tfidf", metrics=tfidf_metrics, validation_macro_f1=tfidf_tuned.validation_macro_f1, metadata={"tfidf_analyzer": tfidf_option, "probe_config": _config_dict(tfidf_tuned.config)}))
            rows.append(_record(fold=outer.fold, proportion=proportion, method="embedding", metrics=embedding_metrics, validation_macro_f1=embedding_tuned.validation_macro_f1, metadata={"probe_config": _config_dict(embedding_tuned.config), "embedding_model": "gemini-embedding-001", "embedding_task_type": "CLASSIFICATION", "embedding_output_dimensionality": int(embeddings.shape[1])}))
            method_predictions["tfidf"] = tfidf_test_pred
            method_predictions["embedding"] = embedding_test_pred
            method_probabilities["tfidf"] = tfidf_test_probabilities
            method_probabilities["embedding"] = embedding_test_probabilities

            if not options.skip_probability_fusion:
                oof = _oof_base_probabilities(
                    train_indices,
                    dataset,
                    embeddings,
                    analyzer=tfidf_option,
                    tfidf_config=tfidf_tuned.config,
                    embedding_config=embedding_tuned.config,
                    seed=base_seed + 31,
                )
                validation_base = np.concatenate(
                    (
                        tfidf_tuned.probe.predict_proba(validation_x),
                        embedding_tuned.probe.predict_proba(embeddings[validation_indices]),
                    ),
                    axis=1,
                )
                test_base = np.concatenate(
                    (
                        tfidf_tuned.probe.predict_proba(test_x),
                        embedding_tuned.probe.predict_proba(embeddings[test_indices]),
                    ),
                    axis=1,
                )
                fusion_tuned = tune_probe(
                    oof,
                    _index_labels(dataset, train_indices),
                    validation_base,
                    y_validation,
                    n_classes=dataset.label_space.n_classes,
                    candidates=[replace_seed(candidate, base_seed + 47) for candidate in candidates],
                )
                fusion_probabilities = fusion_tuned.probe.predict_proba(test_base)
                fusion_pred = np.argmax(fusion_probabilities, axis=1).astype(np.int64)
                fusion_metrics = classification_metrics(y_test, fusion_pred, dataset.label_space.n_classes)
                rows.append(_record(fold=outer.fold, proportion=proportion, method="probability_fusion", metrics=fusion_metrics, validation_macro_f1=fusion_tuned.validation_macro_f1, metadata={"probe_config": _config_dict(fusion_tuned.config), "stacking": "three-fold group-aware OOF"}))
                method_predictions["probability_fusion"] = fusion_pred
                method_probabilities["probability_fusion"] = fusion_probabilities

            if not options.skip_feature_fusion:
                normalized_train_embeddings = l2_normalize_rows(embeddings[train_indices])
                normalized_validation_embeddings = l2_normalize_rows(embeddings[validation_indices])
                normalized_test_embeddings = l2_normalize_rows(embeddings[test_indices])
                feature_candidates: list[tuple[float, TunedProbe]] = []
                for alpha in (0.25, 1.0, 4.0):
                    feature_tuned = tune_probe(
                        FeatureBlocks(train_x, normalized_train_embeddings, alpha),
                        _index_labels(dataset, train_indices),
                        FeatureBlocks(validation_x, normalized_validation_embeddings, alpha),
                        y_validation,
                        n_classes=dataset.label_space.n_classes,
                        candidates=[replace_seed(candidate, base_seed + int(alpha * 100)) for candidate in candidates],
                    )
                    feature_candidates.append((alpha, feature_tuned))
                alpha, feature_tuned = max(feature_candidates, key=lambda item: item[1].validation_macro_f1)
                feature_probabilities = feature_tuned.probe.predict_proba(FeatureBlocks(test_x, normalized_test_embeddings, alpha))
                feature_pred = np.argmax(feature_probabilities, axis=1).astype(np.int64)
                feature_metrics = classification_metrics(y_test, feature_pred, dataset.label_space.n_classes)
                rows.append(_record(fold=outer.fold, proportion=proportion, method="feature_fusion", metrics=feature_metrics, validation_macro_f1=feature_tuned.validation_macro_f1, metadata={"alpha": alpha, "probe_config": _config_dict(feature_tuned.config), "embedding_block": "row-wise L2 normalized"}))
                method_predictions["feature_fusion"] = feature_pred
                method_probabilities["feature_fusion"] = feature_probabilities

            for method, predictions in method_predictions.items():
                key = f"fold{outer.fold}_p{proportion}_{method}"
                prediction_store[key] = predictions
                probability_store[key] = method_probabilities[method]
                truth_store[key] = y_test
                groups_store[key] = test_groups

    result: dict[str, Any] = {
        "schema_version": 1,
        "seed": options.seed,
        "label_space": list(dataset.label_space.classes),
        "dataset_summary": group_summary(dataset.recipes),
        "split_plan": split_plan.to_dict(),
        "options": asdict(options),
        "rows": rows,
        "summary": _summary(rows),
    }
    primary_keys = [key for key in prediction_store if key.endswith("_p100_tfidf")]
    embedding_keys = [key for key in prediction_store if key.endswith("_p100_embedding")]
    if len(primary_keys) == len(embedding_keys) == len(split_plan.folds):
        primary_keys.sort()
        embedding_keys.sort()
        y = np.concatenate([truth_store[key] for key in primary_keys])
        tfidf_pred = np.concatenate([prediction_store[key] for key in primary_keys])
        embedding_pred = np.concatenate([prediction_store[key] for key in embedding_keys])
        groups = np.concatenate([groups_store[key] for key in primary_keys])
        fold_strata = np.concatenate(
            [np.full(len(truth_store[key]), key.split("_", 1)[0], dtype=object) for key in primary_keys]
        )
        result["primary_bootstrap_embedding_minus_tfidf"] = paired_group_bootstrap_difference(
            y,
            tfidf_pred,
            embedding_pred,
            groups,
            dataset.label_space.n_classes,
            n_bootstrap=options.bootstrap_replicates,
            seed=options.seed,
            strata=fold_strata,
        )

    with (destination / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    np.savez_compressed(
        destination / "predictions.npz",
        **{f"pred__{key}": value for key, value in prediction_store.items()},
        **{f"prob__{key}": value for key, value in probability_store.items()},
        **{f"truth__{key}": value for key, value in truth_store.items()},
        **{f"group__{key}": value.astype(str) for key, value in groups_store.items()},
    )
    return result


def replace_seed(config: ProbeConfig, seed: int) -> ProbeConfig:
    """Copy a candidate with a deterministic seed for a nested model fit."""

    return ProbeConfig(
        learning_rate=config.learning_rate,
        l2=config.l2,
        batch_size=config.batch_size,
        max_epochs=config.max_epochs,
        patience=config.patience,
        min_delta=config.min_delta,
        seed=seed,
    )
