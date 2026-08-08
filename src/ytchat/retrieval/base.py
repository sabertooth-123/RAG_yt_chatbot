"""Retriever protocol and score calibration.

Every retriever ranks by its own native ``score`` but also reports
``components["confidence"]`` in ``[0, 1]`` with consistent meaning across
implementations.  The refusal gate reads *only* the confidence, so one
``min_score`` threshold is valid no matter which retriever is active.
"""

from __future__ import annotations

import math
from typing import Protocol, Sequence, runtime_checkable

from ytchat.models import Chunk, ScoredChunk

CONFIDENCE_KEY = "confidence"


@runtime_checkable
class Retriever(Protocol):
    @property
    def name(self) -> str: ...

    def search(self, query: str, k: int) -> list[ScoredChunk]: ...


def confidence_of(hit: ScoredChunk) -> float:
    """Calibrated confidence for a hit, falling back to the raw score."""
    return float(hit.components.get(CONFIDENCE_KEY, hit.score))


def best_confidence(hits: Sequence[ScoredChunk]) -> float:
    return max((confidence_of(h) for h in hits), default=0.0)


def cosine_confidence(similarity: float) -> float:
    """Normalized embeddings give cosine in [-1, 1]; negatives mean 'unrelated',
    so clipping (rather than rescaling) keeps the threshold interpretable."""
    return max(0.0, min(1.0, float(similarity)))


def bm25_confidence(score: float, tau: float = 8.0) -> float:
    """Squash unbounded BM25 into [0, 1].

    ``1 - exp(-s/tau)`` is monotonic, hits ~0.63 at s=tau and ~0.86 at s=2*tau.
    ``tau`` is empirically set for typical 5-15 word questions and is exactly the
    knob to sweep in the refusal-threshold experiment.
    """
    if score <= 0:
        return 0.0
    return 1.0 - math.exp(-float(score) / tau)


def rerank(hits: Sequence[ScoredChunk], retriever: str) -> list[ScoredChunk]:
    """Sort by score and reassign dense 0-based ranks."""
    ordered = sorted(hits, key=lambda h: (-h.score, h.chunk.index))
    return [
        ScoredChunk(
            chunk=h.chunk, score=h.score, rank=i,
            retriever=retriever, components=h.components,
        )
        for i, h in enumerate(ordered)
    ]


def chunk_lookup(chunks: Sequence[Chunk]) -> dict[int, Chunk]:
    """Map ``chunk_id`` → chunk.  Falls back to ``index`` for unpersisted chunks."""
    return {(c.chunk_id if c.chunk_id is not None else c.index): c for c in chunks}