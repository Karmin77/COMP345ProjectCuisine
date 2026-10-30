from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from comp345.baseline import run_tfidf_baseline
from comp345.data import Recipe, normalize_ingredients
from comp345.embeddings import generate_embeddings, load_embedding_cache
from comp345.metrics import classification_metrics, paired_group_bootstrap_difference
from comp345.model import FeatureBlocks, ProbeConfig, TorchLinearProbe
from comp345.splits import PreparedDataset, assert_group_integrity, make_nested_subsets, make_outer_splits
from comp345.tfidf import fit_transform_pair, ingredient_analyzer


def _toy_dataset() -> PreparedDataset:
    recipes = []
    labels = ("italian", "mexican", "thai")
    for label_id, label in enumerate(labels):
        for i in range(12):
            # Deliberate duplicate groups exercise the group boundary.
            ingredients = (f"ingredient {label_id}", "salt") if i < 2 else (f"ingredient {label_id} {i}", "salt")
            recipes.append(
                Recipe(
                    recipe_id=label_id * 100 + i,
                    cuisine=label,
                    ingredients=ingredients,
                    normalized_ingredients=normalize_ingredients(ingredients),
                )
            )
    return PreparedDataset.from_recipes(recipes)


def test_normalization_and_group_aware_nested_splits() -> None:
    assert normalize_ingredients(["  Garlic\tCloves ", "SALT"]) == ("garlic cloves", "salt")
    dataset = _toy_dataset()
    plan = make_outer_splits(dataset, seed=9, validation_fraction=0.2, proportions=(25, 50, 100))
    for fold in plan.folds:
        assert_group_integrity(
            {"train": fold.train_pool, "validation": fold.validation, "test": fold.test},
            dataset.groups,
        )
        subsets = make_nested_subsets(fold.train_pool, dataset.labels, dataset.groups, (25, 50, 100), seed=9)
        assert np.all(np.isin(subsets[25], subsets[50]))
        assert np.all(np.isin(subsets[50], subsets[100]))
        assert len(subsets[100]) == len(fold.train_pool)


def test_word_bigrams_do_not_cross_ingredient_boundaries() -> None:
    tokens = ingredient_analyzer("word_unigram_bigram")("red pepper\nolive oil")
    assert "red pepper" in tokens
    assert "pepper olive" not in tokens


def test_sparse_probe_and_feature_blocks() -> None:
    x = csr_matrix(np.asarray([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32))
    dense = np.asarray([[0.1, 0.2], [0.2, 0.1], [0.3, 0.4], [0.4, 0.3]], dtype=np.float32)
    blocks = FeatureBlocks(x, dense, 0.5)
    probe = TorchLinearProbe(4, 2, ProbeConfig(max_epochs=3, patience=1, batch_size=2, seed=4))
    probe.fit(blocks, [0, 1, 0, 1])
    probabilities = probe.predict_proba(blocks)
    assert probabilities.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_tfidf_and_metrics_keep_fixed_classes() -> None:
    vectorizer, train, (test,) = fit_transform_pair(
        ["salt\npepper", "rice\nsoy sauce"], ["salt\npepper"], option="whole_ingredient"
    )
    assert train.shape[1] == 4
    assert test.shape[1] == train.shape[1]
    metrics = classification_metrics([0, 0], [0, 0], 3)
    assert len(metrics["per_class_f1"]) == 3
    bootstrap = paired_group_bootstrap_difference(
        [0, 1, 0, 1], [0, 1, 0, 1], [1, 1, 0, 0], ["a", "a", "b", "b"], 2, n_bootstrap=20, seed=2
    )
    assert bootstrap["ci_95_low"] <= bootstrap["ci_95_high"]


def test_tfidf_baseline_runs_without_embeddings(tmp_path) -> None:
    dataset = _toy_dataset()
    plan = make_outer_splits(dataset, seed=12, validation_fraction=0.2, proportions=(100,))
    result = run_tfidf_baseline(dataset, plan, tmp_path, proportions=(100,), quick=True)
    assert len(result["rows"]) == 3
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "predictions.npz").exists()


def test_embedding_cache_can_be_filled_in_resumable_segments(tmp_path, monkeypatch) -> None:
    from types import ModuleType, SimpleNamespace

    monkeypatch.setattr("comp345.embeddings.time.sleep", lambda _seconds: None)
    dataset = _toy_dataset()
    calls = []

    class FakeClient:
        def __init__(self, api_key, http_options):
            assert api_key == "test-key"
            assert http_options.timeout == 20_000
            assert http_options.retry_options.attempts == 1
            self.models = self

        def embed_content(self, *, model, contents, config):
            calls.append((model, contents, config.task_type, config.output_dimensionality))
            value = float(len(calls))
            return SimpleNamespace(
                embeddings=[
                    SimpleNamespace(values=np.full(768, value, dtype=np.float32))
                    for _ in contents
                ]
            )

    genai = ModuleType("google.genai")
    genai.Client = FakeClient
    config_module = ModuleType("google.genai.types")

    class EmbedContentConfig:
        def __init__(self, *, task_type, output_dimensionality):
            self.task_type = task_type
            self.output_dimensionality = output_dimensionality

    class HttpRetryOptions:
        def __init__(self, *, attempts):
            self.attempts = attempts

    class HttpOptions:
        def __init__(self, *, timeout, retry_options):
            self.timeout = timeout
            self.retry_options = retry_options

    config_module.EmbedContentConfig = EmbedContentConfig
    config_module.HttpRetryOptions = HttpRetryOptions
    config_module.HttpOptions = HttpOptions
    google = ModuleType("google")
    google.genai = genai
    genai.types = config_module
    monkeypatch.setitem(__import__("sys").modules, "google", google)
    monkeypatch.setitem(__import__("sys").modules, "google.genai", genai)
    monkeypatch.setitem(__import__("sys").modules, "google.genai.types", config_module)

    output = tmp_path / "embeddings.npz"
    generate_embeddings(dataset.recipes, output, api_key="test-key", request_delay=0, batch_size=2, items_per_minute=100, start_index=0, end_index=5)
    assert not output.exists()
    assert sum(len(call[1]) for call in calls) == 5

    generate_embeddings(dataset.recipes, output, api_key="test-key", request_delay=0, batch_size=2, items_per_minute=100, start_index=5)
    assert output.exists()
    assert len(calls) == 19
    vectors = load_embedding_cache(output, dataset.recipes)
    assert vectors.shape == (dataset.n_samples, 768)
    assert all(call[0] == "gemini-embedding-001" and call[2:] == ("CLASSIFICATION", 768) for call in calls)
    assert all(1 <= len(call[1]) <= 2 for call in calls)


def test_embedding_generation_pauses_on_free_quota_limit(tmp_path, monkeypatch) -> None:
    from types import ModuleType

    dataset = _toy_dataset()

    class QuotaError(Exception):
        code = 429

    class FakeClient:
        def __init__(self, api_key, http_options):
            self.models = self

        def embed_content(self, **kwargs):
            raise QuotaError("quota exhausted")

    genai = ModuleType("google.genai")
    genai.Client = FakeClient
    config_module = ModuleType("google.genai.types")

    class HttpOptions:
        def __init__(self, **kwargs):
            pass

    class HttpRetryOptions:
        def __init__(self, **kwargs):
            pass

    class EmbedContentConfig:
        def __init__(self, **kwargs):
            pass

    config_module.HttpOptions = HttpOptions
    config_module.HttpRetryOptions = HttpRetryOptions
    config_module.EmbedContentConfig = EmbedContentConfig
    google = ModuleType("google")
    google.genai = genai
    genai.types = config_module
    monkeypatch.setitem(__import__("sys").modules, "google", google)
    monkeypatch.setitem(__import__("sys").modules, "google.genai", genai)
    monkeypatch.setitem(__import__("sys").modules, "google.genai.types", config_module)

    output = tmp_path / "embeddings.npz"
    with pytest.raises(RuntimeError, match="free-tier quota reached"):
        generate_embeddings(
            dataset.recipes,
            output,
            api_key="test-key",
            max_retries=4,
            batch_size=5,
            start_index=0,
            end_index=5,
        )

    done = np.load(output.with_suffix(".npz.done.npy"), allow_pickle=False)
    assert not done.any()
    assert output.with_suffix(".npz.partial.npy").exists()
    assert not output.exists()
