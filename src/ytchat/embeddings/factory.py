from __future__ import annotations

from ytchat.config import Settings
from ytchat.embeddings.base import Embedder
from ytchat.embeddings.hashing import HashingEmbedder
from ytchat.embeddings.sentence_tf import SentenceTransformerEmbedder
from ytchat.errors import ConfigurationError


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "hashing":
        return HashingEmbedder()
    if settings.embedder == "sentence-transformers":
        return SentenceTransformerEmbedder(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )
    raise ConfigurationError(f"Unknown embedder: {settings.embedder!r}")