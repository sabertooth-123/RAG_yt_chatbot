import numpy as np
import pytest

from ytchat.database.vectorstore import VectorStore
from ytchat.errors import RetrievalError


@pytest.fixture
def store():
    matrix = np.eye(4, dtype="float32")
    return VectorStore.build(matrix, [10, 11, 12, 13])


def test_search_returns_ids_best_first(store) -> None:
    hits = store.search(np.array([1, 0, 0, 0], dtype="float32"), k=2)
    assert hits[0][0] == 10
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert hits[0][1] >= hits[1][1]


def test_k_is_clamped_to_index_size(store) -> None:
    assert len(store.search(np.ones(4, dtype="float32"), k=99)) == 4


def test_dimension_mismatch_is_a_clear_error(store) -> None:
    with pytest.raises(RetrievalError, match="dim"):
        store.search(np.ones(8, dtype="float32"), k=1)


def test_empty_index_returns_no_hits() -> None:
    empty = VectorStore.build(np.zeros((0, 4), dtype="float32"), [])
    assert empty.search(np.ones(4, dtype="float32"), k=5) == []


def test_save_load_roundtrip(store, tmp_path) -> None:
    path = tmp_path / "idx.npz"
    store.save(path)
    loaded = VectorStore.load(path)
    assert loaded is not None
    assert loaded.chunk_ids == store.chunk_ids
    np.testing.assert_array_equal(loaded.matrix, store.matrix)


def test_corrupt_index_file_is_discarded(tmp_path) -> None:
    path = tmp_path / "idx.npz"
    path.write_bytes(b"not an npz file")
    assert VectorStore.load(path) is None
    assert not path.exists(), "a corrupt cache file should be removed, not left to fail again"


def test_rebuilds_from_sqlite_when_index_file_is_deleted(index, settings) -> None:
    from ytchat.database.repository import Repository

    path = VectorStore.index_path(
        settings.index_dir, index.video_id, index.chunk_set_id, index.embedder.model_id
    )
    assert path.exists()
    path.unlink()

    with Repository(settings.db_path) as repo:
        rebuilt = VectorStore.load_or_build(
            repo, index.video_id, index.chunk_set_id,
            index.embedder.model_id, settings.index_dir,
        )
    assert rebuilt is not None
    assert rebuilt.chunk_ids == index.store.chunk_ids
    assert path.exists(), "the index file should be regenerated on rebuild"