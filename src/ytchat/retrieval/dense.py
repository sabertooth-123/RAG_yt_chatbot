from __future__ import annotations

from typing import Sequence

from ytchat.database.vectorstore import VectorStore
from ytchat.embeddings.base import Embedder
from ytchat.models import Chunk, ScoredChunk
from ytchat.retrieval.base import CONFIDENCE_KEY, chunk_lookup, cosine_confidence


class DenseRetriever:
    """Exact cosine search over chunk embeddings."""

    def __init__(self, store: VectorStore, embedder: Embedder, chunks: Sequence[Chunk]) -> None:
        self.store = store
        self.embedder = embedder
        self._by_id = chunk_lookup(chunks)

    @property
    def name(self) -> str:
        return "dense"

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        if not query.strip():
            return []
        q = self.embedder.encode_query(query)
        hits = self.store.search(q, k)
        out: list[ScoredChunk] = []
        for rank, (chunk_id, sim) in enumerate(hits):
            chunk = self._by_id.get(chunk_id)
            if chunk is None:
                continue
            out.append(
                ScoredChunk(
                    chunk=chunk, score=float(sim), rank=rank, retriever="dense",
                    components={"cosine": float(sim), CONFIDENCE_KEY: cosine_confidence(sim)},
                )
            )
        return out