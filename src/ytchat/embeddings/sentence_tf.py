"""Sentence-Transformers embedder.

Asymmetric models (BGE, E5, GTE) are trained with distinct query and passage
representations.  Skipping the query prefix on BGE costs several points of
Recall@5 — a silent, easy-to-miss bug, so the prefixes are auto-detected from
the model name and folded into ``model_id`` so a prefix change invalidates cache.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ytchat.embeddings.base import l2_normalize
from ytchat.errors import EmbeddingError

_QUERY_PREFIXES = {
    "bge": "Represent this sentence for searching relevant passages: ",
    "e5": "query: ",
    "gte": "",
}
_DOC_PREFIXES = {"bge": "", "e5": "passage: ", "gte": ""}


def _family(model_name: str) -> str | None:
    lowered = model_name.lower()
    for family in ("bge", "e5", "gte"):
        if family in lowered:
            return family
    return None


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        family = _family(model_name)
        self.query_prefix = _QUERY_PREFIXES.get(family or "", "")
        self.doc_prefix = _DOC_PREFIXES.get(family or "", "")
        self._model = None
        self._dim: int | None = None

    # -- lazy load: importing torch costs ~2s, so never pay it on a cache hit --
    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed. "
                    "Install with: pip install 'yt-chat[embeddings]' "
                    "(or use --embedder hashing for a dependency-free run)."
                ) from exc
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:
                raise EmbeddingError(
                    f"Could not load embedding model {self.model_name!r}: {exc}"
                ) from exc
            # Renamed in sentence-transformers 5.x; keep the old name as fallback
            # so both major versions work without a deprecation warning.
            get_dim = getattr(
                self._model,
                "get_embedding_dimension",
                getattr(self._model, "get_sentence_embedding_dimension", None),
            )
            self._dim = int(get_dim())
        return self._model

    @property
    def model_id(self) -> str:
        tag = "pfx" if (self.query_prefix or self.doc_prefix) else "raw"
        return f"st:{self.model_name}:{tag}:v1"

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        assert self._dim is not None
        return self._dim

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        payload = [self.doc_prefix + t for t in texts]
        vectors = model.encode(
            payload,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return l2_normalize(vectors)

    def encode_query(self, text: str) -> np.ndarray:
        model = self._load()
        vector = model.encode(
            [self.query_prefix + text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return l2_normalize(vector)