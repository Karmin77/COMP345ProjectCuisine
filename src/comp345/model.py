"""PyTorch multinomial logistic regression for sparse, dense, and fused blocks."""

from __future__ import annotations

from dataclasses import dataclass, replace
import copy
import os
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse
import torch
from torch import nn


@dataclass(frozen=True)
class ProbeConfig:
    learning_rate: float = 3e-3
    l2: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 60
    patience: int = 8
    min_delta: float = 1e-4
    seed: int = 345

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.l2 < 0 or self.batch_size <= 0:
            raise ValueError("learning_rate, batch_size, and l2 must be valid positive values")
        if self.max_epochs <= 0 or self.patience < 0 or self.min_delta < 0:
            raise ValueError("max_epochs must be positive and stopping values non-negative")


@dataclass(frozen=True)
class FeatureBlocks:
    """A sparse TF-IDF block plus an optional dense embedding block.

    Keeping the blocks separate avoids materializing a 40,000 x (vocabulary +
    768) dense or fully populated CSR matrix during feature fusion.
    """

    sparse_block: sparse.csr_matrix | None = None
    dense_block: np.ndarray | None = None
    dense_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.sparse_block is None and self.dense_block is None:
            raise ValueError("at least one feature block is required")
        n_rows: int | None = None
        if self.sparse_block is not None:
            if not sparse.isspmatrix_csr(self.sparse_block):
                raise TypeError("sparse_block must be a CSR matrix")
            n_rows = self.sparse_block.shape[0]
        if self.dense_block is not None:
            dense = np.asarray(self.dense_block)
            if dense.ndim != 2:
                raise ValueError("dense_block must be two-dimensional")
            if n_rows is not None and dense.shape[0] != n_rows:
                raise ValueError("feature blocks have different row counts")
            if not np.isfinite(dense).all():
                raise ValueError("dense_block contains NaN or infinite values")
            if self.dense_scale <= 0:
                raise ValueError("dense_scale must be positive")

    @property
    def n_rows(self) -> int:
        if self.sparse_block is not None:
            return int(self.sparse_block.shape[0])
        assert self.dense_block is not None
        return int(self.dense_block.shape[0])

    @property
    def n_features(self) -> int:
        sparse_dim = 0 if self.sparse_block is None else self.sparse_block.shape[1]
        dense_dim = 0 if self.dense_block is None else self.dense_block.shape[1]
        return int(sparse_dim + dense_dim)

    def batch_dense(self, indices: np.ndarray) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.sparse_block is not None:
            parts.append(self.sparse_block[indices].toarray().astype(np.float32, copy=False))
        if self.dense_block is not None:
            dense = np.asarray(self.dense_block[indices], dtype=np.float32)
            parts.append(dense * np.float32(self.dense_scale))
        if len(parts) == 1:
            return np.ascontiguousarray(parts[0], dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)

    def numpy_logits(self, indices: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """Compute a batch with SciPy/NumPy for fast validation and inference."""

        offset = 0
        logits: np.ndarray | None = None
        if self.sparse_block is not None:
            width = self.sparse_block.shape[1]
            logits = self.sparse_block[indices].dot(weights[:, :width].T).astype(np.float32, copy=False)
            offset = width
        if self.dense_block is not None:
            dense = np.asarray(self.dense_block[indices], dtype=np.float32) * np.float32(self.dense_scale)
            dense_logits = dense @ weights[:, offset:].T
            logits = dense_logits if logits is None else logits + dense_logits
        assert logits is not None
        return np.asarray(logits + bias, dtype=np.float32)

    def batch_tensors(self, indices: np.ndarray) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return sparse and dense batch tensors without densifying TF-IDF."""

        sparse_tensor: torch.Tensor | None = None
        dense_tensor: torch.Tensor | None = None
        if self.sparse_block is not None:
            coo = self.sparse_block[indices].tocoo()
            coordinates = np.vstack((coo.row, coo.col)).astype(np.int64, copy=False)
            sparse_tensor = torch.sparse_coo_tensor(
                torch.from_numpy(coordinates),
                torch.from_numpy(coo.data.astype(np.float32, copy=False)),
                size=coo.shape,
                dtype=torch.float32,
                check_invariants=False,
            ).coalesce()
        if self.dense_block is not None:
            dense = np.asarray(self.dense_block[indices], dtype=np.float32)
            dense_tensor = torch.from_numpy(np.ascontiguousarray(dense * np.float32(self.dense_scale)))
        return sparse_tensor, dense_tensor


def as_feature_blocks(features: object) -> FeatureBlocks:
    if isinstance(features, FeatureBlocks):
        return features
    if sparse.issparse(features):
        return FeatureBlocks(sparse_block=features.tocsr().astype(np.float32))
    array = np.asarray(features, dtype=np.float32)
    return FeatureBlocks(dense_block=array)


def l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.float32(1e-12))


class TorchLinearProbe:
    """A deterministic PyTorch linear softmax classifier."""

    def __init__(self, n_features: int, n_classes: int, config: ProbeConfig) -> None:
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.config = config
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.best_epoch: int | None = None
        self.history: list[dict[str, float]] = []

    def _module(self) -> nn.Linear:
        if self.weights is None or self.bias is None:
            raise RuntimeError("probe has not been fitted")
        module = nn.Linear(self.n_features, self.n_classes)
        with torch.no_grad():
            module.weight.copy_(torch.from_numpy(self.weights))
            module.bias.copy_(torch.from_numpy(self.bias))
        return module

    @staticmethod
    def _set_seed(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except (AttributeError, RuntimeError):
            pass
        # One CPU thread is slower for very large matrices but prevents tiny
        # smoke runs from varying with the host's thread scheduler.
        if os.environ.get("COMP345_DETERMINISTIC", "1") == "1":
            torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))

    def fit(
        self,
        features: object,
        labels: Sequence[int],
        *,
        validation_features: object | None = None,
        validation_labels: Sequence[int] | None = None,
    ) -> "TorchLinearProbe":
        blocks = as_feature_blocks(features)
        labels = np.asarray(labels, dtype=np.int64)
        if len(labels) != blocks.n_rows:
            raise ValueError("training feature and label lengths differ")
        if labels.size and (labels.min() < 0 or labels.max() >= self.n_classes):
            raise ValueError("training labels fall outside the fixed label space")
        val_blocks = None if validation_features is None else as_feature_blocks(validation_features)
        val_labels = None if validation_labels is None else np.asarray(validation_labels, dtype=np.int64)
        if (val_blocks is None) != (val_labels is None):
            raise ValueError("validation features and labels must be supplied together")
        if val_blocks is not None and len(val_labels) != val_blocks.n_rows:
            raise ValueError("validation feature and label lengths differ")
        self._set_seed(self.config.seed)
        module = nn.Linear(blocks.n_features, self.n_classes)
        optimizer = torch.optim.Adam(
            [
                {"params": [module.weight], "weight_decay": self.config.l2},
                {"params": [module.bias], "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
        )
        loss_fn = nn.CrossEntropyLoss()
        generator = torch.Generator().manual_seed(self.config.seed)
        best_state: dict[str, torch.Tensor] | None = None
        best_metric = float("inf")
        best_epoch = 0
        wait = 0
        self.history = []

        for epoch in range(self.config.max_epochs):
            module.train()
            permutation = torch.randperm(blocks.n_rows, generator=generator).numpy()
            train_loss = 0.0
            for start in range(0, blocks.n_rows, self.config.batch_size):
                batch_indices = permutation[start : start + self.config.batch_size]
                # Densify only this bounded minibatch.  Keeping the full
                # TF-IDF matrix sparse controls memory, while dense BLAS for a
                # 512-row batch is substantially faster than CPU sparse-mm on
                # common Windows PyTorch builds.
                x = torch.from_numpy(blocks.batch_dense(batch_indices))
                y = torch.from_numpy(labels[batch_indices])
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(module(x), y)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.detach()) * len(batch_indices)
            train_loss /= max(1, blocks.n_rows)

            if val_blocks is not None:
                current_weight = module.weight.detach().cpu().numpy()
                current_bias = module.bias.detach().cpu().numpy()
                val_loss = self._numpy_loss(val_blocks, val_labels, current_weight, current_bias, self.config.batch_size)
                metric = val_loss
            else:
                val_loss = float("nan")
                metric = train_loss
            self.history.append({"epoch": float(epoch + 1), "train_loss": train_loss, "validation_loss": val_loss})
            if metric < best_metric - self.config.min_delta:
                best_metric = metric
                best_epoch = epoch + 1
                best_state = {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if val_blocks is not None and wait > self.config.patience:
                break

        if best_state is None:
            best_state = {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
            best_epoch = len(self.history)
        module.load_state_dict(best_state)
        self.weights = module.weight.detach().cpu().numpy().astype(np.float32, copy=True)
        self.bias = module.bias.detach().cpu().numpy().astype(np.float32, copy=True)
        self.best_epoch = best_epoch
        return self

    @staticmethod
    def _numpy_loss(
        blocks: FeatureBlocks,
        labels: np.ndarray,
        weights: np.ndarray,
        bias: np.ndarray,
        batch_size: int,
    ) -> float:
        total = 0.0
        for start in range(0, blocks.n_rows, batch_size):
            indices = np.arange(start, min(start + batch_size, blocks.n_rows), dtype=np.int64)
            logits = blocks.numpy_logits(indices, weights, bias)
            logits -= logits.max(axis=1, keepdims=True)
            log_partition = np.log(np.exp(logits).sum(axis=1))
            total += float((-logits[np.arange(len(indices)), labels[indices]] + log_partition).sum())
        return total / max(1, blocks.n_rows)

    @staticmethod
    def _logits(module: nn.Linear, blocks: FeatureBlocks, indices: np.ndarray) -> torch.Tensor:
        sparse_batch, dense_batch = blocks.batch_tensors(indices)
        offset = 0
        logits: torch.Tensor | None = None
        if sparse_batch is not None:
            width = blocks.sparse_block.shape[1]  # type: ignore[union-attr]
            logits = torch.sparse.mm(sparse_batch, module.weight[:, :width].t())
            offset = width
        if dense_batch is not None:
            dense_width = dense_batch.shape[1]
            dense_logits = dense_batch @ module.weight[:, offset : offset + dense_width].t()
            logits = dense_logits if logits is None else logits + dense_logits
        assert logits is not None
        return logits + module.bias

    def predict_proba(self, features: object, *, batch_size: int | None = None) -> np.ndarray:
        blocks = as_feature_blocks(features)
        if blocks.n_features != self.n_features:
            raise ValueError("feature dimension differs from the fitted probe")
        outputs: list[np.ndarray] = []
        step = batch_size or self.config.batch_size
        assert self.weights is not None and self.bias is not None
        for start in range(0, blocks.n_rows, step):
            indices = np.arange(start, min(start + step, blocks.n_rows), dtype=np.int64)
            logits = blocks.numpy_logits(indices, self.weights, self.bias)
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            outputs.append(probabilities.astype(np.float32, copy=False))
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False) if outputs else np.empty((0, self.n_classes), dtype=np.float32)

    def predict(self, features: object) -> np.ndarray:
        return np.argmax(self.predict_proba(features), axis=1).astype(np.int64)


@dataclass(frozen=True)
class TunedProbe:
    probe: TorchLinearProbe
    config: ProbeConfig
    validation_macro_f1: float
    validation_loss: float


def _macro_f1(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> float:
    from sklearn.metrics import f1_score

    return float(
        f1_score(
            y_true,
            y_pred,
            labels=np.arange(n_classes),
            average="macro",
            zero_division=0,
        )
    )


def tune_probe(
    train_features: object,
    train_labels: Sequence[int],
    validation_features: object,
    validation_labels: Sequence[int],
    *,
    n_classes: int,
    candidates: Iterable[ProbeConfig],
) -> TunedProbe:
    """Select a probe configuration by validation macro-F1, then validation loss."""

    candidates = list(candidates)
    if not candidates:
        raise ValueError("at least one probe candidate is required")
    best: TunedProbe | None = None
    for candidate in candidates:
        probe = TorchLinearProbe(as_feature_blocks(train_features).n_features, n_classes, candidate)
        probe.fit(
            train_features,
            train_labels,
            validation_features=validation_features,
            validation_labels=validation_labels,
        )
        probabilities = probe.predict_proba(validation_features)
        predictions = np.argmax(probabilities, axis=1)
        score = _macro_f1(validation_labels, predictions, n_classes)
        # The final refit keeps the epoch at which validation loss was best.
        selected_epochs = max(1, int(probe.best_epoch or candidate.max_epochs))
        selected_config = replace(candidate, max_epochs=selected_epochs)
        validation_loss = float(min(item["validation_loss"] for item in probe.history))
        current = TunedProbe(probe, selected_config, score, validation_loss)
        if best is None or (score, -validation_loss) > (best.validation_macro_f1, -best.validation_loss):
            best = current
    assert best is not None
    final_probe = TorchLinearProbe(as_feature_blocks(train_features).n_features, n_classes, best.config)
    final_probe.fit(train_features, train_labels)
    return TunedProbe(final_probe, best.config, best.validation_macro_f1, best.validation_loss)
