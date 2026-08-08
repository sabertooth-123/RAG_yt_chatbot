"""Cross-encoder reranking.

Bi-encoders (what ``DenseRetriever`` uses) embed query and chunk *separately*,
so the model never sees them together — fast, but it cannot reason about how the
two interact.  A cross-encoder scores the pair jointly, which is markedly more
accurate and far too slow to run over every chunk.

The standard resolution is retrieve-then-rerank: pull ``candidates`` cheaply,
then rescore that shortlist expensively.  Implemented here as a *decorator* over
any ``Retriever``, so it composes with dense, sparse, and hybrid alike and the
eval harness can measure the gain for each independently.
"""

from __future__ import annotations

import math
from typing import Sequence

from ytchat.errors import RetrievalError
from ytchat.models import ScoredChunk
from ytchat.retrieval.base import CONFIDENCE_KEY, Retriever

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _sigmoid(x: float) -> float:
    """Cross-encoders emit unbounded logits (roughly -11..+11 for ms-marco).

    Sigmoid maps them into [0, 1] so reranked hits honour the same calibrated
    confidence contract as every other retriever — otherwise switching on
    reranking would silently invalidate ``min_score``.  Computed branchwise to
    avoid ``exp`` overflow on large negative inputs.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


class CrossEncoderScorer:
    """Lazily-loaded sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    @property
    def model_id(self) -> str:
        return f"ce:{self.model_name}"

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RetrievalError(
                    "Reranking needs sentence-transformers: "
                    "pip install 'yt-chat[embeddings]' (or set enable_rerank=false)."
                ) from exc
            try:
                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as exc:
                raise RetrievalError(
                    f"Could not load reranker {self.model_name!r}: {exc}"
                ) from exc
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        model = self._load()
        raw = model.predict(
            [(query, t) for t in texts],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in raw]


class HashingScorer:
    """Deterministic offline stand-in: Jaccard token overlap.

    Not a real cross-encoder — it exists so the reranking *plumbing* is tested
    in CI without downloading model weights.  Quality claims come from the eval
    harness with a real model, never from this.
    """

    model_id = "hashing-scorer-v1"

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t for t in text.lower().split() if len(t) > 2}

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        q = self._tokens(query)
        scores: list[float] = []
        for text in texts:
            d = self._tokens(text)
            union = q | d
            # Logit-shaped output so the sigmoid calibration path is exercised.
            jaccard = len(q & d) / len(union) if union else 0.0
            scores.append(jaccard * 12.0 - 6.0)
        return scores


class RerankingRetriever:
    """Wraps any retriever: fetch ``candidates``, rescore, return top ``k``.

    ``components`` keeps the base retriever's original score and rank alongside
    the rerank logit, so the eval report can show how far the reranker moved
    each chunk — the difference between "it helped" and "it helped *because*".
    """

    def __init__(
        self,
        base: Retriever,
        scorer: CrossEncoderScorer | HashingScorer,
        candidates: int = 30,
    ) -> None:
        self.base = base
        self.scorer = scorer
        self.candidates = candidates

    @property
    def name(self) -> str:
        return f"{self.base.name}+rerank"

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        if not query.strip():
            return []
        pool = self.base.search(query, max(self.candidates, k))
        if not pool:
            return []

        scores = self.scorer.score(query, [h.chunk.text for h in pool])
        # Ties fall back to the base retriever's ordering rather than list order.
        order = sorted(range(len(pool)), key=lambda i: (-scores[i], pool[i].rank))

        out: list[ScoredChunk] = []
        for rank, i in enumerate(order[:k]):
            hit = pool[i]
            components = dict(hit.components)
            components["base_score"] = hit.score
            components["base_rank"] = float(hit.rank)
            components["rerank_logit"] = scores[i]
            components[CONFIDENCE_KEY] = _sigmoid(scores[i])
            out.append(
                ScoredChunk(
                    chunk=hit.chunk,
                    score=scores[i],
                    rank=rank,
                    retriever=self.name,
                    components=components,
                )
            )
        return out
