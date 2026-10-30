"""Frozen, group-aware outer, validation, and learning-curve splits."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from .data import Recipe
from .labels import LabelSpace


DEFAULT_PROPORTIONS: tuple[int, ...] = (2, 5, 10, 20, 40, 60, 80, 100)


@dataclass(frozen=True)
class PreparedDataset:
    recipes: tuple[Recipe, ...]
    label_space: LabelSpace
    labels: np.ndarray
    groups: np.ndarray
    recipe_ids: np.ndarray
    _texts: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_texts", tuple(recipe.serialized_ingredients for recipe in self.recipes))

    @classmethod
    def from_recipes(cls, recipes: Sequence[Recipe]) -> "PreparedDataset":
        frozen = tuple(recipes)
        label_space = LabelSpace.from_labels(recipe.cuisine for recipe in frozen)
        labels = label_space.encode([recipe.cuisine for recipe in frozen])
        groups = np.asarray([recipe.group_key for recipe in frozen], dtype=object)
        recipe_ids = np.asarray([recipe.recipe_id for recipe in frozen], dtype=np.int64)
        return cls(frozen, label_space, labels, groups, recipe_ids)

    @property
    def n_samples(self) -> int:
        return len(self.recipes)

    @property
    def texts(self) -> tuple[str, ...]:
        return self._texts

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for recipe in self.recipes:
            digest.update(str(recipe.recipe_id).encode("ascii"))
            digest.update(b"\0")
            digest.update(recipe.cuisine.encode("utf-8"))
            digest.update(b"\0")
            digest.update(recipe.group_key.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()


@dataclass(frozen=True)
class OuterFold:
    fold: int
    train_pool: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    @property
    def development(self) -> np.ndarray:
        return np.sort(np.concatenate((self.train_pool, self.validation)))


@dataclass(frozen=True)
class SplitPlan:
    seed: int
    validation_fraction: float
    proportions: tuple[int, ...]
    folds: tuple[OuterFold, ...]
    dataset_fingerprint: str | None = None
    recipe_ids: tuple[int, ...] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "validation_fraction": self.validation_fraction,
            "proportions": list(self.proportions),
            "dataset_fingerprint": self.dataset_fingerprint,
            "recipe_ids": None if self.recipe_ids is None else list(self.recipe_ids),
            "folds": [
                {
                    "fold": fold.fold,
                    "train_pool": fold.train_pool.tolist(),
                    "validation": fold.validation.tolist(),
                    "test": fold.test.tolist(),
                }
                for fold in self.folds
            ],
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, path: str | Path, dataset: PreparedDataset | None = None) -> "SplitPlan":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        folds = tuple(
            OuterFold(
                int(item["fold"]),
                np.asarray(item["train_pool"], dtype=np.int64),
                np.asarray(item["validation"], dtype=np.int64),
                np.asarray(item["test"], dtype=np.int64),
            )
            for item in raw["folds"]
        )
        plan = cls(
            int(raw["seed"]),
            float(raw["validation_fraction"]),
            tuple(int(value) for value in raw["proportions"]),
            folds,
            raw.get("dataset_fingerprint"),
            None if raw.get("recipe_ids") is None else tuple(int(value) for value in raw["recipe_ids"]),
        )
        if dataset is not None:
            if plan.dataset_fingerprint is not None and plan.dataset_fingerprint != dataset.fingerprint:
                raise ValueError("split manifest dataset fingerprint does not match the dataset")
            if plan.recipe_ids is not None and plan.recipe_ids != tuple(int(value) for value in dataset.recipe_ids):
                raise ValueError("split manifest recipe order does not match the dataset")
            all_indices = np.arange(dataset.n_samples, dtype=np.int64)
            observed = np.concatenate([np.concatenate((f.train_pool, f.validation, f.test)) for f in plan.folds])
            if len(observed) != len(plan.folds) * dataset.n_samples:
                raise ValueError("split manifest does not assign each recipe exactly once per outer fold")
            if not np.array_equal(np.sort(np.unique(observed)), all_indices):
                raise ValueError("split manifest indices do not match the dataset")
            for fold in plan.folds:
                if fold.fold < 0 or fold.fold >= len(plan.folds):
                    raise ValueError("split manifest contains an invalid fold number")
                for name, indices in (("train_pool", fold.train_pool), ("validation", fold.validation), ("test", fold.test)):
                    if np.any(indices < 0) or np.any(indices >= dataset.n_samples):
                        raise ValueError(f"split manifest {name} indices are out of range")
                assert_group_integrity(
                    {"train_pool": fold.train_pool, "validation": fold.validation, "test": fold.test},
                    dataset.groups,
                )
        return plan


def _n_groups(indices: np.ndarray, groups: np.ndarray) -> int:
    return len(np.unique(groups[indices]))


def _stratified_group_test_folds(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[np.ndarray]:
    """Return a partition of ``indices`` without ever splitting a group."""

    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) < 2 or _n_groups(indices, groups) < 2:
        return [np.sort(indices)]
    n_splits = min(int(n_splits), _n_groups(indices, groups))
    n_splits = max(n_splits, 2)
    y = labels[indices]
    g = groups[indices]
    # StratifiedGroupKFold is the registered splitter for the full dataset.
    # A GroupKFold fallback keeps tiny smoke-test datasets usable when a class
    # has fewer groups than the requested number of folds.
    try:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        result = [indices[test] for _, test in splitter.split(np.zeros(len(indices)), y, g)]
    except (ValueError, RuntimeError):
        splitter = GroupKFold(n_splits=n_splits)
        result = [indices[test] for _, test in splitter.split(np.zeros(len(indices)), y, g)]
    return [np.sort(part) for part in result]


def make_outer_splits(
    dataset: PreparedDataset,
    *,
    seed: int = 345,
    n_splits: int = 3,
    validation_fraction: float = 0.15,
    proportions: Iterable[int] = DEFAULT_PROPORTIONS,
) -> SplitPlan:
    """Create the proposal's frozen three-way group-aware evaluation plan.

    The outer test folds are produced first.  A seven-way stratified group
    partition of each development set supplies an approximately 15% validation
    slice.  All indices are sorted before serialization so that the manifest is
    stable across runs and platforms.
    """

    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    proportions = tuple(int(value) for value in proportions)
    if not proportions or any(value <= 0 or value > 100 for value in proportions):
        raise ValueError("proportions must be non-empty percentages in 1..100")
    if tuple(sorted(set(proportions))) != tuple(proportions):
        raise ValueError("proportions must be sorted and unique")

    all_indices = np.arange(dataset.n_samples, dtype=np.int64)
    outer_parts = _stratified_group_test_folds(
        all_indices, dataset.labels, dataset.groups, n_splits, seed
    )
    if len(outer_parts) != n_splits:
        raise RuntimeError("could not create the requested number of outer folds")

    folds: list[OuterFold] = []
    val_splits = max(2, round(1.0 / validation_fraction))
    for fold_id, test in enumerate(outer_parts):
        test = np.sort(test)
        development = np.setdiff1d(all_indices, test, assume_unique=False)
        val_parts = _stratified_group_test_folds(
            development,
            dataset.labels,
            dataset.groups,
            val_splits,
            seed + 1009 * (fold_id + 1),
        )
        if len(val_parts) == 1:
            validation = np.asarray([], dtype=np.int64)
        else:
            validation = np.sort(
                min(
                    val_parts,
                    key=lambda part: (abs(len(part) / max(1, len(development)) - validation_fraction), int(np.min(part))),
                )
            )
        train_pool = np.setdiff1d(development, validation, assume_unique=False)
        if not len(validation) or not len(train_pool):
            raise RuntimeError(f"outer fold {fold_id} produced an empty split")
        if np.intersect1d(train_pool, validation).size or np.intersect1d(train_pool, test).size:
            raise RuntimeError(f"outer fold {fold_id} has overlapping splits")
        folds.append(OuterFold(fold_id, train_pool, validation, test))
    return SplitPlan(
        seed,
        validation_fraction,
        proportions,
        tuple(folds),
        dataset_fingerprint=dataset.fingerprint,
        recipe_ids=tuple(int(value) for value in dataset.recipe_ids),
    )


def make_nested_subsets(
    pool_indices: Sequence[int],
    labels: np.ndarray,
    groups: np.ndarray,
    proportions: Iterable[int] = DEFAULT_PROPORTIONS,
    *,
    seed: int = 345,
) -> dict[int, np.ndarray]:
    """Create nested, approximately stratified group-preserving subsets.

    Groups are shuffled inside their dominant cuisine and then greedily
    interleaved to track the overall cuisine proportions.  Accumulating that
    one order gives one shared nested sequence for every method and prevents a
    smaller learning-curve setting from borrowing vocabulary or examples from
    a larger setting.  This avoids the very expensive 100-way
    ``StratifiedGroupKFold`` call that a large recipe collection would require.
    """

    pool = np.sort(np.asarray(pool_indices, dtype=np.int64))
    requested = tuple(int(value) for value in proportions)
    if not requested or any(value <= 0 or value > 100 for value in requested):
        raise ValueError("proportions must be percentages in 1..100")
    unique_groups, group_inverse = np.unique(groups[pool], return_inverse=True)
    n_groups = len(unique_groups)
    if n_groups <= 1:
        return {value: pool.copy() for value in requested}
    rng = np.random.default_rng(seed)
    group_rows = [pool[np.flatnonzero(group_inverse == i)] for i in range(n_groups)]
    group_label_counts = np.zeros((n_groups, int(labels.max()) + 1), dtype=np.int64)
    for group_id, rows in enumerate(group_rows):
        group_label_counts[group_id] = np.bincount(labels[rows], minlength=group_label_counts.shape[1])
    dominant = np.argmax(group_label_counts, axis=1)
    queues: dict[int, list[int]] = {}
    for label in np.unique(dominant):
        members = np.flatnonzero(dominant == label)
        queues[int(label)] = [int(value) for value in rng.permutation(members)]
    target_rates = np.bincount(labels[pool], minlength=group_label_counts.shape[1]).astype(float)
    target_rates /= max(1.0, target_rates.sum())
    selected_counts = np.zeros_like(target_rates)
    ordered_groups: list[int] = []
    selected_total = 0
    while len(ordered_groups) < n_groups:
        available = [label for label, queue in queues.items() if queue]
        if not available:
            raise RuntimeError("stratified group ordering lost a group")
        if selected_total == 0:
            scores = target_rates[available]
        else:
            scores = target_rates[available] - selected_counts[available] / selected_total
        # A tiny deterministic jitter breaks ties without changing the
        # intended deficit ordering.
        chosen_label = int(available[int(np.argmax(scores + rng.random(len(available)) * 1e-9))])
        chosen_group = queues[chosen_label].pop()
        ordered_groups.append(chosen_group)
        selected_counts += group_label_counts[chosen_group]
        selected_total += int(group_label_counts[chosen_group].sum())
    result: dict[int, np.ndarray] = {}
    for proportion in requested:
        count = n_groups if proportion == 100 else max(1, int(np.ceil(n_groups * proportion / 100.0)))
        selected = np.concatenate([group_rows[group_id] for group_id in ordered_groups[:count]])
        result[proportion] = np.sort(selected)
    return result


def assert_group_integrity(
    parts: Mapping[str, Sequence[int]], groups: np.ndarray
) -> None:
    """Raise if a group appears in multiple named partitions."""

    seen: dict[object, str] = {}
    for name, indices in parts.items():
        for index in indices:
            group = groups[int(index)]
            owner = seen.setdefault(group, name)
            if owner != name:
                raise AssertionError(f"group {group!r} occurs in {owner} and {name}")
