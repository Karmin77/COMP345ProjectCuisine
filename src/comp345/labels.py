"""Fixed label-space handling shared by every evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class LabelSpace:
    """A stable class order, including classes absent from a small train subset."""

    classes: tuple[str, ...]

    @classmethod
    def from_labels(cls, labels: Iterable[str]) -> "LabelSpace":
        classes = tuple(sorted(set(labels)))
        if not classes:
            raise ValueError("cannot construct a label space from no labels")
        return cls(classes)

    def __post_init__(self) -> None:
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError("classes must be a non-empty sequence of unique labels")

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def encode(self, labels: Sequence[str] | Iterable[str]) -> np.ndarray:
        lookup = {label: i for i, label in enumerate(self.classes)}
        result = []
        for label in labels:
            if label not in lookup:
                raise ValueError(f"unknown cuisine label: {label!r}")
            result.append(lookup[label])
        return np.asarray(result, dtype=np.int64)

    def decode(self, indices: Sequence[int] | np.ndarray) -> list[str]:
        return [self.classes[int(index)] for index in indices]
