"""Hybrid retrieval: Reciprocal Rank Fusion (default) or weighted score fusion.

RRF is the default because it is scale-free — it consumes only ranks, so the
incomparability of cosine and BM25 magnitudes never arises and there is no
alpha to tune per video.  Weighted fusion is kept because comparing the two
(and testing whether a tuned alpha transfers across videos) is one of the
planned experiments, not because it is known to be better.
"""

from __future__ import annotations

from typing import Sequence

from ytchat.models import ScoredChunk
from ytchat.retrieval.base import CONFIDENCE_KEY, Retriever, confidence_of


def _minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 if hi > 0 else 0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


class HybridRetriever:
    def __init__(
        self,
        dense: Retriever,
        sparse: Retriever,
        fusion: str = "rrf",
        rrf_k: int = 60,
        alpha: float = 0.5,
        candidate_k: int = 30,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.alpha = alpha
        self.candidate_k = candidate_k

    @property
    def name(self) -> str:
        return f"hybrid-{self.fusion}"

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        depth = max(self.candidate_k, k)
        dense_hits = self.dense.search(query, depth)
        sparse_hits = self.sparse.search(query, depth)
        if not dense_hits and not sparse_hits:
            return []
        fused = (
            self._rrf(dense_hits, sparse_hits)
            if self.fusion == "rrf"
            else self._weighted(dense_hits, sparse_hits)
        )
        return fused[:k]

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _key(hit: ScoredChunk) -> int:
        return hit.chunk.chunk_id if hit.chunk.chunk_id is not None else hit.chunk.index

    def _collect(self, dense_hits, sparse_hits):
        """Union of both candidate lists, keyed by chunk."""
        table: dict[int, dict] = {}
        for source, hits in (("dense", dense_hits), ("sparse", sparse_hits)):
            for hit in hits:
                entry = table.setdefault(
                    self._key(hit), {"chunk": hit.chunk, "components": {}}
                )
                entry[f"{source}_rank"] = hit.rank
                entry[f"{source}_score"] = hit.score
                entry["components"].update(hit.components)
                entry["components"][f"{source}_score"] = hit.score
                entry["components"][f"{source}_conf"] = confidence_of(hit)
        return table

    def _finalise(self, table: dict[int, dict], scores: dict[int, float]) -> list[ScoredChunk]:
        ordered = sorted(
            table.items(), key=lambda kv: (-scores[kv[0]], kv[1]["chunk"].index)
        )
        out: list[ScoredChunk] = []
        for rank, (key, entry) in enumerate(ordered):
            comps = dict(entry["components"])
            # Dense-anchored rather than max(dense, sparse).  BM25 magnitude
            # reflects term rarity, not relevance, so a coincidental keyword
            # match on an unanswerable question can post a high sparse score and
            # defeat the refusal gate — measured on the benchmark, max() made
            # hybrid a *worse* refusal signal than dense alone.  Sparse agreement
            # can nudge confidence up but cannot manufacture it, and the result
            # stays >= dense so agreement is still rewarded.  Ordering is
            # unaffected: that is still the fusion score.
            dense_conf = comps.get("dense_conf", 0.0)
            sparse_conf = comps.get("sparse_conf", 0.0)
            comps[CONFIDENCE_KEY] = dense_conf + 0.15 * sparse_conf * (1.0 - dense_conf)
            comps["fusion"] = scores[key]
            out.append(
                ScoredChunk(chunk=entry["chunk"], score=scores[key], rank=rank,
                            retriever=self.name, components=comps)
            )
        return out

    def _rrf(self, dense_hits, sparse_hits) -> list[ScoredChunk]:
        table = self._collect(dense_hits, sparse_hits)
        scores = {
            key: sum(
                1.0 / (self.rrf_k + entry[f"{s}_rank"] + 1)
                for s in ("dense", "sparse") if f"{s}_rank" in entry
            )
            for key, entry in table.items()
        }
        return self._finalise(table, scores)

    def _weighted(self, dense_hits, sparse_hits) -> list[ScoredChunk]:
        table = self._collect(dense_hits, sparse_hits)
        keys = list(table)
        norm: dict[str, dict[int, float]] = {}
        for source in ("dense", "sparse"):
            present = [k for k in keys if f"{source}_score" in table[k]]
            values = _minmax([table[k][f"{source}_score"] for k in present])
            norm[source] = dict(zip(present, values))
        scores = {
            k: self.alpha * norm["dense"].get(k, 0.0)
            + (1 - self.alpha) * norm["sparse"].get(k, 0.0)
            for k in keys
        }
        return self._finalise(table, scores)