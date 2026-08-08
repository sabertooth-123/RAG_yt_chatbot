"""Deterministic, dependency-free embedder.

Purpose: make the *entire* test suite and CI run without downloading model
weights.  It is a hashing-trick bag-of-ngrams — functionally a real embedder
(same shapes, same normalization, same cache semantics) but semantically shallow.
Tests written against it assert plumbing, never semantic quality.

Uses blake2b rather than ``hash()`` because Python randomizes string hashing per
process, which would silently break the embedding cache across runs.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, Sequence

import numpy as np

from ytchat.embeddings.base import l2_normalize

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _features(text: str) -> Iterable[str]:
    toks = _tokens(text)
    yield from toks
    yield from (f"{a}_{b}" for a, b in zip(toks, toks[1:]))  # bigrams


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


class HashingEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def model_id(self) -> str:
        return f"hashing-{self._dim}-v1"

    @property
    def dim(self) -> int:
        return self._dim

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype="float32")
        for feature in _features(text):
            idx, sign = _bucket(feature, self._dim)
            vec[idx] += sign
        return vec

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype="float32")
        return l2_normalize(np.vstack([self._encode_one(t) for t in texts]))

    def encode_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._encode_one(text))