"""Embedder protocol.

``model_id`` is the embedding-cache key, so it must encode *everything* that
changes the output vectors — model name, normalization, and query/document
prefixing.  Two embedders that produce different vectors must never share an id.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization.  With normalized vectors, inner product == cosine,
    which is why every index in this project can use plain IP search."""
    arr = np.asarray(matrix, dtype="float32")
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 0 else arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms