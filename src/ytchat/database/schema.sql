-- yt-chat cache schema (SQLite).
-- Cache key chain:  video -> chunk_set(chunker fingerprint) -> embedding_set(model id)
-- Changing the chunker invalidates chunks + embeddings; changing the embedding
-- model reuses chunks and rebuilds only vectors.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
    video_id        TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    channel         TEXT,
    duration_s      REAL,
    language        TEXT,
    transcript_kind TEXT NOT NULL DEFAULT 'unknown',
    n_segments      INTEGER NOT NULL DEFAULT 0,
    extra_json      TEXT NOT NULL DEFAULT '{}',
    fetched_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS segments (
    video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    idx      INTEGER NOT NULL,
    start_s  REAL NOT NULL,
    end_s    REAL NOT NULL,
    text     TEXT NOT NULL,
    PRIMARY KEY (video_id, idx)
);

CREATE TABLE IF NOT EXISTS chunk_sets (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id     TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    fingerprint  TEXT NOT NULL,
    config_json  TEXT NOT NULL,
    n_chunks     INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    UNIQUE (video_id, fingerprint)
);

CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_set_id INTEGER NOT NULL REFERENCES chunk_sets(id) ON DELETE CASCADE,
    idx          INTEGER NOT NULL,
    text         TEXT NOT NULL,
    start_s      REAL NOT NULL,
    end_s        REAL NOT NULL,
    seg_start    INTEGER NOT NULL,
    seg_end      INTEGER NOT NULL,
    n_chars      INTEGER NOT NULL,
    UNIQUE (chunk_set_id, idx)
);

CREATE INDEX IF NOT EXISTS idx_chunks_set ON chunks(chunk_set_id, idx);

CREATE TABLE IF NOT EXISTS embedding_sets (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_set_id INTEGER NOT NULL REFERENCES chunk_sets(id) ON DELETE CASCADE,
    model_id     TEXT NOT NULL,
    dim          INTEGER NOT NULL,
    normalized   INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT NOT NULL,
    UNIQUE (chunk_set_id, model_id)
);

CREATE TABLE IF NOT EXISTS embeddings (
    embedding_set_id INTEGER NOT NULL REFERENCES embedding_sets(id) ON DELETE CASCADE,
    chunk_id         INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    vector           BLOB NOT NULL,
    PRIMARY KEY (embedding_set_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id   TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    retriever  TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    citations_json  TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);