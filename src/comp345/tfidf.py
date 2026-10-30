"""Leakage-safe TF-IDF transforms over normalized ingredient lists."""

from __future__ import annotations

import re
from typing import Callable, Iterable

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")
ANALYZER_OPTIONS = ("word_unigram", "word_unigram_bigram", "whole_ingredient")


def ingredient_analyzer(option: str) -> Callable[[str], list[str]]:
    if option not in ANALYZER_OPTIONS:
        raise ValueError(f"unknown TF-IDF analyzer {option!r}; expected one of {ANALYZER_OPTIONS}")

    def analyze(document: str) -> list[str]:
        tokens: list[str] = []
        for ingredient in document.split("\n"):
            if option == "whole_ingredient":
                tokens.append(ingredient)
                continue
            words = TOKEN_PATTERN.findall(ingredient)
            tokens.extend(words)
            if option == "word_unigram_bigram":
                tokens.extend(f"{left} {right}" for left, right in zip(words, words[1:]))
        return tokens

    return analyze


def make_vectorizer(option: str) -> TfidfVectorizer:
    """Build a row-wise L2-normalized vectorizer with no cross-ingredient n-grams."""

    return TfidfVectorizer(analyzer=ingredient_analyzer(option), norm="l2", dtype=np.float32)


def fit_transform_pair(
    train_documents: Iterable[str],
    *transform_documents: Iterable[str],
    option: str,
) -> tuple[TfidfVectorizer, csr_matrix, tuple[csr_matrix, ...]]:
    train_documents = list(train_documents)
    vectorizer = make_vectorizer(option)
    train = vectorizer.fit_transform(train_documents).tocsr().astype("float32")
    transformed = tuple(
        vectorizer.transform(list(documents)).tocsr().astype("float32")
        for documents in transform_documents
    )
    return vectorizer, train, transformed
