"""Resumable Gemini embedding generation and cache validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from .data import Recipe


MODEL_NAME = "gemini-embedding-001"
TASK_TYPE = "CLASSIFICATION"
OUTPUT_DIMENSIONALITY = 768
SCHEMA_VERSION = 1


def recipe_fingerprint(recipes: Sequence[Recipe]) -> str:
    digest = hashlib.sha256()
    for recipe in recipes:
        digest.update(str(recipe.recipe_id).encode("ascii"))
        digest.update(b"\0")
        digest.update(recipe.serialized_ingredients.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _metadata(recipes: Sequence[Recipe]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "task_type": TASK_TYPE,
        "output_dimensionality": OUTPUT_DIMENSIONALITY,
        "input_serialization": "sorted normalized ingredients joined by newline",
        "recipe_fingerprint": recipe_fingerprint(recipes),
        "recipe_count": len(recipes),
    }


def load_embedding_cache(path: str | Path, recipes: Sequence[Recipe]) -> np.ndarray:
    """Load a completed cache and ensure rows match the requested recipe order."""

    cache_path = Path(path)
    with np.load(cache_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        recipe_ids = archive["recipe_ids"].astype(np.int64)
        vectors = archive["vectors"].astype(np.float32)
    expected = _metadata(recipes)
    if metadata != expected:
        raise ValueError("embedding cache metadata does not match the requested dataset/configuration")
    expected_ids = np.asarray([recipe.recipe_id for recipe in recipes], dtype=np.int64)
    if not np.array_equal(recipe_ids, expected_ids):
        raise ValueError("embedding cache recipe order does not match the dataset")
    if vectors.shape != (len(recipes), OUTPUT_DIMENSIONALITY):
        raise ValueError(f"invalid embedding matrix shape: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("embedding cache contains NaN or infinite values")
    return vectors


def generate_embeddings(
    recipes: Sequence[Recipe],
    output: str | Path,
    *,
    api_key: str | None = None,
    request_delay: float = 0.0,
    max_retries: int = 2,
    batch_size: int = 100,
    items_per_minute: int = 80,
    start_index: int = 0,
    end_index: int | None = None,
) -> Path:
    """Generate an index range in bounded API batches and persist each result.

    The Google Gen AI SDK is imported only when this function runs, so the rest
    of the project remains usable without installing the API client or setting
    credentials. A partial cache is bound to the complete dataset fingerprint
    and can be filled over multiple bounded runs. ``end_index`` is exclusive.
    """

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Gemini embeddings require the `google-genai` package; install project requirements first."
        ) from exc

    credential = api_key or os.environ.get("GEMINI_API_KEY")
    if not credential:
        raise RuntimeError("Set GEMINI_API_KEY before generating Gemini embeddings.")
    if request_delay < 0 or max_retries < 0:
        raise ValueError("request_delay and max_retries must be non-negative")
    if not 1 <= batch_size <= 100:
        raise ValueError("batch_size must be between 1 and 100")
    if not 1 <= items_per_minute <= 3000:
        raise ValueError("items_per_minute must be between 1 and 3000")
    if end_index is None:
        end_index = len(recipes)
    if not 0 <= start_index < end_index <= len(recipes):
        raise ValueError("embedding range must satisfy 0 <= start_index < end_index <= recipe count")

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial_vectors = target.with_suffix(target.suffix + ".partial.npy")
    partial_done = target.with_suffix(target.suffix + ".done.npy")
    partial_meta = target.with_suffix(target.suffix + ".partial.json")
    expected_meta = _metadata(recipes)

    if target.exists():
        load_embedding_cache(target, recipes)
        return target

    if partial_vectors.exists() or partial_done.exists() or partial_meta.exists():
        if not (partial_vectors.exists() and partial_done.exists() and partial_meta.exists()):
            raise ValueError("incomplete partial cache files; remove them before restarting")
        with partial_meta.open("r", encoding="utf-8") as handle:
            if json.load(handle) != expected_meta:
                raise ValueError("partial embedding cache metadata does not match requested dataset/configuration")
        vectors = np.lib.format.open_memmap(partial_vectors, mode="r+")
        done = np.load(partial_done, allow_pickle=False)
        if vectors.shape != (len(recipes), OUTPUT_DIMENSIONALITY) or done.shape != (len(recipes),):
            raise ValueError("partial embedding cache has invalid dimensions")
    else:
        vectors = np.lib.format.open_memmap(
            partial_vectors,
            mode="w+",
            dtype=np.float32,
            shape=(len(recipes), OUTPUT_DIMENSIONALITY),
        )
        done = np.zeros(len(recipes), dtype=np.bool_)
        np.save(partial_done, done, allow_pickle=False)
        with partial_meta.open("w", encoding="utf-8") as handle:
            json.dump(expected_meta, handle, indent=2)

    client = genai.Client(
        api_key=credential,
        http_options=types.HttpOptions(
            timeout=20_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    ids = np.asarray([recipe.recipe_id for recipe in recipes], dtype=np.int64)
    pending = [index for index in range(start_index, end_index) if not done[index]]
    for offset in range(0, len(pending), batch_size):
        batch_indices = pending[offset : offset + batch_size]
        request_started = time.monotonic()
        response = None
        for attempt in range(max_retries + 1):
            try:
                response = client.models.embed_content(
                    model=MODEL_NAME,
                    contents=[recipes[index].serialized_ingredients for index in batch_indices],
                    config=types.EmbedContentConfig(
                        task_type=TASK_TYPE,
                        output_dimensionality=OUTPUT_DIMENSIONALITY,
                    ),
                )
                break
            except Exception as exc:
                status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if status_code == 429:
                    raise RuntimeError(
                        "Gemini free-tier quota reached (HTTP 429). The partial cache is safe; "
                        "wait for the quota window to reset and rerun the same segment."
                    ) from exc
                if attempt >= max_retries:
                    raise
                time.sleep(min(60.0, 2.0**attempt))
        returned = getattr(response, "embeddings", None) if response is not None else None
        if returned is None or len(returned) != len(batch_indices):
            raise RuntimeError(
                f"Gemini returned {0 if returned is None else len(returned)} vectors "
                f"for a batch of {len(batch_indices)} recipes"
            )
        for index, item in zip(batch_indices, returned):
            vector = np.asarray(item.values, dtype=np.float32)
            if vector.shape != (OUTPUT_DIMENSIONALITY,) or not np.isfinite(vector).all():
                raise RuntimeError(f"Gemini returned an invalid embedding for recipe id {recipes[index].recipe_id}")
            vectors[index] = vector
            vectors.flush()
            done[index] = True
            np.save(partial_done, done, allow_pickle=False)
        minimum_interval = 60.0 * len(batch_indices) / items_per_minute
        remaining_wait = max(request_delay, minimum_interval) - (time.monotonic() - request_started)
        if remaining_wait > 0:
            time.sleep(remaining_wait)

    vectors.flush()
    if not done.all():
        # Keep the full-sized, fingerprinted checkpoint for the next segment.
        return target

    final_vectors = np.asarray(vectors, dtype=np.float32)
    np.savez_compressed(
        target,
        vectors=final_vectors,
        recipe_ids=ids,
        metadata=np.asarray(json.dumps(expected_meta, sort_keys=True)),
    )
    load_embedding_cache(target, recipes)
    # Windows keeps a memmap file locked until its mapping and ndarray view close.
    del final_vectors
    vectors.flush()
    mapping = vectors._mmap
    del vectors
    mapping.close()
    partial_vectors.unlink(missing_ok=True)
    partial_done.unlink(missing_ok=True)
    partial_meta.unlink(missing_ok=True)
    return target
