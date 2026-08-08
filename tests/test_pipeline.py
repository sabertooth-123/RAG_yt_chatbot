import pytest

from ytchat.config import Settings
from ytchat.database.repository import Repository
from ytchat.embeddings.hashing import HashingEmbedder
from ytchat.pipeline import ensure_indexed


def _index(settings, db, providers, embedder, video_id):
    transcripts, metas = providers
    with Repository(db) as repo:
        return ensure_indexed(
            video_id, settings, repo,
            transcript_provider=transcripts, metadata_provider=metas, embedder=embedder,
        )


def test_first_run_processes_everything(settings, tmp_db, providers, embedder, metadata) -> None:
    idx = _index(settings, tmp_db, providers, embedder, metadata.video_id)
    assert not idx.stats.fully_cached
    assert idx.stats.n_chunks > 0
    assert idx.store is not None
    assert providers[0].calls == [metadata.video_id]


def test_second_run_is_a_full_cache_hit(settings, tmp_db, providers, embedder, metadata) -> None:
    _index(settings, tmp_db, providers, embedder, metadata.video_id)
    second = _index(settings, tmp_db, providers, embedder, metadata.video_id)

    assert second.stats.fully_cached
    assert providers[0].calls == [metadata.video_id], "cached run must not refetch the transcript"


def test_changing_chunker_reuses_transcript_but_rebuilds_chunks(
    settings, tmp_db, providers, embedder, metadata
) -> None:
    first = _index(settings, tmp_db, providers, embedder, metadata.video_id)

    coarser = settings.model_copy(update={"max_chars": 400})
    second = _index(coarser, tmp_db, providers, embedder, metadata.video_id)

    assert second.stats.transcript_cached, "the transcript should never be refetched"
    assert not second.stats.chunks_cached
    assert not second.stats.embeddings_cached
    assert second.chunk_set_id != first.chunk_set_id
    assert providers[0].calls == [metadata.video_id]


def test_changing_embedder_reuses_chunks(settings, tmp_db, providers, metadata) -> None:
    first = _index(settings, tmp_db, providers, HashingEmbedder(dim=128), metadata.video_id)
    second = _index(settings, tmp_db, providers, HashingEmbedder(dim=64), metadata.video_id)

    assert second.stats.chunks_cached, "a new embedding model must not re-chunk"
    assert not second.stats.embeddings_cached
    assert second.chunk_set_id == first.chunk_set_id


def test_both_embedding_sets_coexist(settings, tmp_db, providers, metadata) -> None:
    _index(settings, tmp_db, providers, HashingEmbedder(dim=128), metadata.video_id)
    _index(settings, tmp_db, providers, HashingEmbedder(dim=64), metadata.video_id)
    again = _index(settings, tmp_db, providers, HashingEmbedder(dim=128), metadata.video_id)
    assert again.stats.fully_cached, "the original embedding set should still be cached"


def test_force_reprocesses_from_scratch(settings, tmp_db, providers, embedder, metadata) -> None:
    _index(settings, tmp_db, providers, embedder, metadata.video_id)
    transcripts, metas = providers
    with Repository(tmp_db) as repo:
        forced = ensure_indexed(
            metadata.video_id, settings, repo, transcript_provider=transcripts,
            metadata_provider=metas, embedder=embedder, force=True,
        )
    assert not forced.stats.transcript_cached
    assert transcripts.calls == [metadata.video_id] * 2


def test_accepts_a_full_url_not_just_an_id(settings, tmp_db, providers, embedder, metadata) -> None:
    url = f"https://www.youtube.com/watch?v={metadata.video_id}&t=30s"
    idx = _index(settings, tmp_db, providers, embedder, url)
    assert idx.video_id == metadata.video_id


def test_chunk_ids_are_assigned_and_retrievable(settings, tmp_db, providers, embedder, metadata) -> None:
    idx = _index(settings, tmp_db, providers, embedder, metadata.video_id)
    assert all(c.chunk_id is not None for c in idx.chunks)
    assert idx.store.chunk_ids == [c.chunk_id for c in idx.chunks]