import numpy as np
import pytest

from ytchat.models import ChunkerConfig
from ytchat.database.repository import Repository
from ytchat.preprocessing.chunking import TranscriptChunker
from ytchat.preprocessing.clean import clean_transcript


@pytest.fixture
def repo(tmp_db):
    with Repository(tmp_db) as r:
        yield r


def test_video_and_transcript_roundtrip(repo, metadata, punctuated_transcript) -> None:
    assert not repo.has_video(metadata.video_id)
    repo.save_video(metadata, punctuated_transcript)
    assert repo.has_video(metadata.video_id)

    got_meta = repo.get_video(metadata.video_id)
    assert got_meta is not None and got_meta.title == metadata.title

    got = repo.get_transcript(metadata.video_id)
    assert got is not None
    assert len(got.segments) == len(punctuated_transcript.segments)
    for a, b in zip(got.segments, punctuated_transcript.segments):
        assert a.text == b.text
        assert a.start_s == pytest.approx(b.start_s)
        assert a.end_s == pytest.approx(b.end_s)


def test_save_video_is_idempotent(repo, metadata, punctuated_transcript) -> None:
    repo.save_video(metadata, punctuated_transcript)
    repo.save_video(metadata, punctuated_transcript)
    assert repo.stats()["segments"] == len(punctuated_transcript.segments)


def test_chunk_cache_hit_and_miss(repo, metadata, punctuated_transcript) -> None:
    repo.save_video(metadata, punctuated_transcript)
    cfg = ChunkerConfig(max_chars=200, overlap_chars=50)
    chunks = TranscriptChunker(cfg).chunk(clean_transcript(punctuated_transcript).segments)

    assert repo.get_chunk_set_id(metadata.video_id, cfg.fingerprint()) is None  # miss
    cs_id = repo.save_chunks(metadata.video_id, cfg, chunks)
    assert repo.get_chunk_set_id(metadata.video_id, cfg.fingerprint()) == cs_id  # hit

    # A different chunker config is a different cache entry.
    other = ChunkerConfig(max_chars=400, overlap_chars=50)
    assert repo.get_chunk_set_id(metadata.video_id, other.fingerprint()) is None


def test_chunks_roundtrip_preserves_timestamps(repo, metadata, punctuated_transcript) -> None:
    repo.save_video(metadata, punctuated_transcript)
    cfg = ChunkerConfig(max_chars=200)
    chunks = TranscriptChunker(cfg).chunk(clean_transcript(punctuated_transcript).segments)
    cs_id = repo.save_chunks(metadata.video_id, cfg, chunks)

    loaded = repo.get_chunks(cs_id)
    assert len(loaded) == len(chunks)
    for a, b in zip(loaded, chunks):
        assert a.text == b.text
        assert a.start_s == pytest.approx(b.start_s)
        assert a.end_s == pytest.approx(b.end_s)
        assert a.seg_start == b.seg_start and a.seg_end == b.seg_end
        assert a.chunk_id is not None      # assigned on persist
        assert a.video_id == metadata.video_id


def test_embeddings_roundtrip_bit_exact(repo, metadata, punctuated_transcript) -> None:
    repo.save_video(metadata, punctuated_transcript)
    cfg = ChunkerConfig(max_chars=200)
    chunks = TranscriptChunker(cfg).chunk(clean_transcript(punctuated_transcript).segments)
    cs_id = repo.save_chunks(metadata.video_id, cfg, chunks)
    stored = repo.get_chunks(cs_id)

    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((len(stored), 16)).astype("float32")
    ids = [c.chunk_id for c in stored]

    assert repo.get_embeddings(cs_id, "model-a") is None
    repo.save_embeddings(cs_id, "model-a", ids, matrix)

    got = repo.get_embeddings(cs_id, "model-a")
    assert got is not None
    got_ids, got_matrix = got
    assert got_ids == ids
    np.testing.assert_array_equal(got_matrix, matrix)

    # Switching embedding model reuses chunks but misses the vector cache.
    assert repo.get_embeddings(cs_id, "model-b") is None


def test_cascade_delete_clears_everything(repo, metadata, punctuated_transcript) -> None:
    repo.save_video(metadata, punctuated_transcript)
    cfg = ChunkerConfig()
    chunks = TranscriptChunker(cfg).chunk(clean_transcript(punctuated_transcript).segments)
    cs_id = repo.save_chunks(metadata.video_id, cfg, chunks)
    stored = repo.get_chunks(cs_id)
    repo.save_embeddings(cs_id, "m", [c.chunk_id for c in stored],
                         np.zeros((len(stored), 4), dtype="float32"))

    repo.clear_video(metadata.video_id)
    s = repo.stats()
    assert s["videos"] == 0 and s["segments"] == 0
    assert s["chunks"] == 0 and s["embeddings"] == 0


def test_conversation_history_roundtrip(repo, metadata, punctuated_transcript) -> None:
    repo.save_video(metadata, punctuated_transcript)
    conv = repo.create_conversation(metadata.video_id, "hybrid")
    repo.add_message(conv, "user", "What is attention?")
    repo.add_message(conv, "assistant", "It focuses on relevant tokens.")
    msgs = repo.get_messages(conv)
    assert [m["role"] for m in msgs] == ["user", "assistant"]