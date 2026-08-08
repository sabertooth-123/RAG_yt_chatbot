"""SQLite cache layer.

Guarantees
----------
* Second run of an already-processed video performs **zero** network calls and
  **zero** embedding computation.
* Vectors round-trip byte-exactly (float32 little-endian BLOBs).
* All writes for one stage happen in a single transaction, so an interrupted
  run never leaves a half-populated chunk set that would later look like a hit.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ytchat.errors import CacheError
from ytchat.models import (
    Chunk,
    ChunkerConfig,
    Citation,
    RawTranscript,
    TranscriptKind,
    TranscriptSegment,
    VideoMetadata,
)

SCHEMA_VERSION = 1
_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_blob(vec: np.ndarray) -> bytes:
    return np.ascontiguousarray(vec, dtype="<f4").tobytes()


def _from_blob(blob: bytes, dim: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype="<f4")
    if arr.size != dim:
        raise CacheError(f"Corrupt embedding blob: expected dim {dim}, got {arr.size}")
    return arr


class Repository:
    """Thin, explicit data-access layer.  No ORM on purpose — the queries are
    few, the schema is stable, and readability beats abstraction here."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self._migrate()

    # -- lifecycle ---------------------------------------------------------
    def _migrate(self) -> None:
        self.conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
        row = self.conn.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
            )
        elif row["version"] != SCHEMA_VERSION:
            raise CacheError(
                f"Cache at {self.db_path} is schema v{row['version']}, "
                f"this build expects v{SCHEMA_VERSION}. Delete it or run a migration."
            )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Repository":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- videos & segments -------------------------------------------------
    def has_video(self, video_id: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone() is not None

    def save_video(self, meta: VideoMetadata, transcript: RawTranscript) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO videos
                   (video_id, title, channel, duration_s, language, transcript_kind,
                    n_segments, extra_json, fetched_at)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(video_id) DO UPDATE SET
                     title=excluded.title, channel=excluded.channel,
                     duration_s=excluded.duration_s, language=excluded.language,
                     transcript_kind=excluded.transcript_kind,
                     n_segments=excluded.n_segments, extra_json=excluded.extra_json,
                     fetched_at=excluded.fetched_at""",
                (
                    meta.video_id,
                    meta.title,
                    meta.channel,
                    meta.duration_s if meta.duration_s is not None else transcript.duration_s,
                    transcript.language,
                    transcript.kind.value,
                    len(transcript.segments),
                    json.dumps(meta.extra),
                    _now(),
                ),
            )
            self.conn.execute("DELETE FROM segments WHERE video_id = ?", (meta.video_id,))
            self.conn.executemany(
                "INSERT INTO segments (video_id, idx, start_s, end_s, text) VALUES (?,?,?,?,?)",
                [(meta.video_id, *s.to_row()) for s in transcript.segments],
            )

    def get_video(self, video_id: str) -> VideoMetadata | None:
        row = self.conn.execute(
            "SELECT * FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        if row is None:
            return None
        return VideoMetadata(
            video_id=row["video_id"],
            title=row["title"],
            channel=row["channel"],
            duration_s=row["duration_s"],
            language=row["language"],
            transcript_kind=TranscriptKind(row["transcript_kind"]),
            extra=json.loads(row["extra_json"]),
        )

    def get_transcript(self, video_id: str) -> RawTranscript | None:
        video = self.conn.execute(
            "SELECT language, transcript_kind FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        if video is None:
            return None
        rows = self.conn.execute(
            "SELECT idx, start_s, end_s, text FROM segments WHERE video_id = ? ORDER BY idx",
            (video_id,),
        ).fetchall()
        segments = tuple(
            TranscriptSegment(
                index=r["idx"], text=r["text"],
                start_s=r["start_s"], duration_s=r["end_s"] - r["start_s"],
            )
            for r in rows
        )
        return RawTranscript(
            video_id=video_id,
            language=video["language"] or "en",
            kind=TranscriptKind(video["transcript_kind"]),
            segments=segments,
        )

    # -- chunk sets --------------------------------------------------------
    def get_chunk_set_id(self, video_id: str, fingerprint: str) -> int | None:
        row = self.conn.execute(
            "SELECT id FROM chunk_sets WHERE video_id = ? AND fingerprint = ?",
            (video_id, fingerprint),
        ).fetchone()
        return int(row["id"]) if row else None

    def save_chunks(
        self, video_id: str, config: ChunkerConfig, chunks: Sequence[Chunk]
    ) -> int:
        fp = config.fingerprint()
        with self.conn:
            cur = self.conn.execute(
                """INSERT INTO chunk_sets (video_id, fingerprint, config_json, n_chunks, created_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(video_id, fingerprint) DO UPDATE SET
                     n_chunks=excluded.n_chunks, created_at=excluded.created_at""",
                (video_id, fp, json.dumps(config.__dict__ if not hasattr(config, "__dataclass_fields__")
                                          else {k: getattr(config, k) for k in config.__dataclass_fields__}),
                 len(chunks), _now()),
            )
            chunk_set_id = self.get_chunk_set_id(video_id, fp)
            assert chunk_set_id is not None
            self.conn.execute("DELETE FROM chunks WHERE chunk_set_id = ?", (chunk_set_id,))
            self.conn.executemany(
                """INSERT INTO chunks
                   (chunk_set_id, idx, text, start_s, end_s, seg_start, seg_end, n_chars)
                   VALUES (?,?,?,?,?,?,?,?)""",
                [
                    (chunk_set_id, c.index, c.text, c.start_s, c.end_s,
                     c.seg_start, c.seg_end, c.n_chars)
                    for c in chunks
                ],
            )
            _ = cur
        return chunk_set_id

    def get_chunks(self, chunk_set_id: int) -> list[Chunk]:
        rows = self.conn.execute(
            """SELECT c.id, c.idx, c.text, c.start_s, c.end_s, c.seg_start, c.seg_end,
                      cs.video_id
               FROM chunks c JOIN chunk_sets cs ON cs.id = c.chunk_set_id
               WHERE c.chunk_set_id = ? ORDER BY c.idx""",
            (chunk_set_id,),
        ).fetchall()
        return [
            Chunk(
                index=r["idx"], text=r["text"], start_s=r["start_s"], end_s=r["end_s"],
                seg_start=r["seg_start"], seg_end=r["seg_end"],
                chunk_id=r["id"], video_id=r["video_id"],
            )
            for r in rows
        ]

    # -- embeddings --------------------------------------------------------
    def get_embedding_set_id(self, chunk_set_id: int, model_id: str) -> int | None:
        row = self.conn.execute(
            "SELECT id FROM embedding_sets WHERE chunk_set_id = ? AND model_id = ?",
            (chunk_set_id, model_id),
        ).fetchone()
        return int(row["id"]) if row else None

    def save_embeddings(
        self,
        chunk_set_id: int,
        model_id: str,
        chunk_ids: Sequence[int],
        matrix: np.ndarray,
        normalized: bool = True,
    ) -> int:
        if matrix.ndim != 2 or matrix.shape[0] != len(chunk_ids):
            raise CacheError(
                f"Embedding matrix {matrix.shape} does not match {len(chunk_ids)} chunks."
            )
        dim = int(matrix.shape[1])
        with self.conn:
            self.conn.execute(
                """INSERT INTO embedding_sets (chunk_set_id, model_id, dim, normalized, created_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(chunk_set_id, model_id) DO UPDATE SET
                     dim=excluded.dim, normalized=excluded.normalized,
                     created_at=excluded.created_at""",
                (chunk_set_id, model_id, dim, int(normalized), _now()),
            )
            es_id = self.get_embedding_set_id(chunk_set_id, model_id)
            assert es_id is not None
            self.conn.execute("DELETE FROM embeddings WHERE embedding_set_id = ?", (es_id,))
            self.conn.executemany(
                "INSERT INTO embeddings (embedding_set_id, chunk_id, vector) VALUES (?,?,?)",
                [(es_id, cid, _to_blob(matrix[i])) for i, cid in enumerate(chunk_ids)],
            )
        return es_id

    def get_embeddings(
        self, chunk_set_id: int, model_id: str
    ) -> tuple[list[int], np.ndarray] | None:
        row = self.conn.execute(
            "SELECT id, dim FROM embedding_sets WHERE chunk_set_id = ? AND model_id = ?",
            (chunk_set_id, model_id),
        ).fetchone()
        if row is None:
            return None
        dim = int(row["dim"])
        rows = self.conn.execute(
            """SELECT e.chunk_id, e.vector FROM embeddings e
               JOIN chunks c ON c.id = e.chunk_id
               WHERE e.embedding_set_id = ? ORDER BY c.idx""",
            (row["id"],),
        ).fetchall()
        if not rows:
            return None
        ids = [int(r["chunk_id"]) for r in rows]
        matrix = np.vstack([_from_blob(r["vector"], dim) for r in rows])
        return ids, matrix

    # -- conversations -----------------------------------------------------
    def create_conversation(self, video_id: str, retriever: str) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO conversations (video_id, retriever, created_at) VALUES (?,?,?)",
                (video_id, retriever, _now()),
            )
        return int(cur.lastrowid)

    def add_message(
        self, conversation_id: int, role: str, content: str,
        citations: Iterable[Citation] = (),
    ) -> None:
        payload = json.dumps([
            {"marker": c.marker, "chunk_id": c.chunk_id, "start_s": c.start_s,
             "end_s": c.end_s, "url": c.url, "timestamp": c.timestamp, "quote": c.quote}
            for c in citations
        ])
        with self.conn:
            self.conn.execute(
                """INSERT INTO messages (conversation_id, role, content, citations_json, created_at)
                   VALUES (?,?,?,?,?)""",
                (conversation_id, role, content, payload, _now()),
            )

    def get_messages(self, conversation_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content, citations_json FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        return [
            {"role": r["role"], "content": r["content"],
             "citations": json.loads(r["citations_json"])}
            for r in rows
        ]

    # -- maintenance -------------------------------------------------------
    def clear_video(self, video_id: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))

    def stats(self) -> dict[str, int]:
        q = lambda t: int(self.conn.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"])  # noqa: E731
        return {t: q(t) for t in
                ("videos", "segments", "chunk_sets", "chunks", "embedding_sets",
                 "embeddings", "conversations", "messages")}