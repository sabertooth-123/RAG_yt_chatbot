from __future__ import annotations

from typing import Sequence

from ytchat.config import Settings
from ytchat.database.vectorstore import VectorStore
from ytchat.embeddings.base import Embedder
from ytchat.errors import ConfigurationError
from ytchat.models import Chunk
from ytchat.retrieval.base import Retriever
from ytchat.retrieval.dense import DenseRetriever
from ytchat.retrieval.hybrid import HybridRetriever
from ytchat.retrieval.sparse import SparseRetriever


def build_retriever(
    name: str,
    settings: Settings,
    chunks: Sequence[Chunk],
    store: VectorStore | None,
    embedder: Embedder | None,
) -> Retriever:
    def _dense() -> Retriever:
        if store is None or embedder is None:
            raise ConfigurationError(
                "Dense retrieval needs embeddings; none are available for this video."
            )
        return DenseRetriever(store, embedder, chunks)

    def _sparse() -> Retriever:
        return SparseRetriever(
            chunks, k1=settings.bm25_k1, b=settings.bm25_b, tau=8.0
        )

    if name == "dense":
        retriever: Retriever = _dense()
    elif name == "sparse":
        retriever = _sparse()
    elif name == "hybrid":
        retriever = HybridRetriever(
            dense=_dense(), sparse=_sparse(), fusion=settings.fusion,
            rrf_k=settings.rrf_k, alpha=settings.hybrid_alpha,
            candidate_k=settings.candidate_k,
        )
    else:
        raise ConfigurationError(
            f"Unknown retriever {name!r}. Choose from: dense, sparse, hybrid."
        )

    # Reranking decorates whichever base retriever was chosen, so the eval
    # harness can measure its contribution to each strategy independently.
    if settings.enable_rerank:
        from ytchat.retrieval.rerank import CrossEncoderScorer, RerankingRetriever

        retriever = RerankingRetriever(
            base=retriever,
            scorer=CrossEncoderScorer(
                model_name=settings.rerank_model,
                batch_size=settings.rerank_batch_size,
            ),
            candidates=settings.rerank_candidates,
        )
    return retriever