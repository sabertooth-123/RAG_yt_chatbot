"""Vector index over chunk embeddings.

Two interchangeable exact backends:

* ``faiss``  — IndexFlatIP over L2-normalized vectors (exact cosine).
* ``numpy``  — a single matmul.  Exact, zero dependencies, and genuinely faster
  than FAISS below roughly 10k vectors because it skips the wrapper overhead.

A video yields hundreds to a few thousand chunks, so an *approximate* index
(HNSW/IVF) would trade recall for a speedup we do not need.  Flat is the
correct choice here, and the README says so rather than reaching for HNSW
because it sounds more serious.

The on-disk index is a pure cache: SQLite holds the authoritative vectors, so a
missing or stale index file is rebuilt silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ytchat.database.repository import Repository
from ytchat.embeddings.base import l2_normalize
from ytchat.errors import RetrievalError


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:80]


@dataclass
class VectorStore:
    matrix: np.ndarray          # (n, dim), L2-normalized float32
    chunk_ids: list[int]
    backend: str = "numpy"
    _index: object | None = None

    # -- construction ------------------------------------------------------
    @classmethod
    def build(
        cls, matrix: np.ndarray, chunk_ids: Sequence[int], prefer_faiss: bool = True
    ) -> "VectorStore":
        arr = l2_normalize(np.asarray(matrix, dtype="float32"))
        if arr.ndim != 2 or arr.shape[0] != len(chunk_ids):
            raise RetrievalError(
                f"Vector matrix {arr.shape} does not match {len(chunk_ids)} chunk ids."
            )
        store = cls(matrix=arr, chunk_ids=list(chunk_ids))
        if prefer_faiss:
            store._try_faiss()
        return store

    def _try_faiss(self) -> None:
        try:
            import faiss  # type: ignore[import-not-found]
        except ImportError:
            return
        if self.matrix.shape[0] == 0:
            return
        index = faiss.IndexFlatIP(self.matrix.shape[1])
        index.add(self.matrix)
        self._index = index
        self.backend = "faiss"

    # -- search ------------------------------------------------------------
    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return ``(chunk_id, cosine_similarity)`` pairs, best first."""
        if self.matrix.shape[0] == 0 or k <= 0:
            return []
        q = l2_normalize(np.asarray(query_vector, dtype="float32")).reshape(1, -1)
        if q.shape[1] != self.matrix.shape[1]:
            raise RetrievalError(
                f"Query dim {q.shape[1]} != index dim {self.matrix.shape[1]}. "
                "The embedding model changed — clear the cache for this video."
            )
        k = min(k, self.matrix.shape[0])

        if self.backend == "faiss" and self._index is not None:
            scores, idx = self._index.search(q, k)  # type: ignore[attr-defined]
            return [
                (self.chunk_ids[int(i)], float(s))
                for i, s in zip(idx[0], scores[0]) if i >= 0
            ]

        sims = (self.matrix @ q[0]).astype("float32")
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.chunk_ids[int(i)], float(sims[int(i)])) for i in top]

    # -- persistence -------------------------------------------------------
    @staticmethod
    def index_path(index_dir: Path, video_id: str, chunk_set_id: int, model_id: str) -> Path:
        return Path(index_dir) / f"{_slug(video_id)}__cs{chunk_set_id}__{_slug(model_id)}.npz"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, matrix=self.matrix, chunk_ids=np.asarray(self.chunk_ids, dtype="int64"))

    @classmethod
    def load(cls, path: Path, prefer_faiss: bool = True) -> "VectorStore | None":
        if not path.exists():
            return None
        try:
            with np.load(path) as data:
                matrix = data["matrix"]
                chunk_ids = [int(i) for i in data["chunk_ids"]]
        except Exception:
            path.unlink(missing_ok=True)   # corrupt cache file: drop and rebuild
            return None
        return cls.build(matrix, chunk_ids, prefer_faiss=prefer_faiss)

    # -- the cache-aware entrypoint ---------------------------------------
    @classmethod
    def load_or_build(
        cls,
        repo: Repository,
        video_id: str,
        chunk_set_id: int,
        model_id: str,
        index_dir: Path,
        prefer_faiss: bool = True,
    ) -> "VectorStore | None":
        """Disk cache → SQLite rebuild → ``None`` (meaning: embeddings not computed yet)."""
        path = cls.index_path(index_dir, video_id, chunk_set_id, model_id)
        stored = repo.get_embeddings(chunk_set_id, model_id)
        if stored is None:
            path.unlink(missing_ok=True)
            return None

        db_ids, matrix = stored
        cached = cls.load(path, prefer_faiss=prefer_faiss)
        if cached is not None and cached.chunk_ids == db_ids:
            return cached

        store = cls.build(matrix, db_ids, prefer_faiss=prefer_faiss)
        store.save(path)
        return store