import numpy as np
import pytest

from ytchat.embeddings.base import l2_normalize
from ytchat.embeddings.hashing import HashingEmbedder


def test_deterministic_across_instances() -> None:
    a = HashingEmbedder(dim=64).encode_documents(["attention is all you need"])
    b = HashingEmbedder(dim=64).encode_documents(["attention is all you need"])
    np.testing.assert_array_equal(a, b)


def test_output_is_normalized_and_correctly_shaped(embedder) -> None:
    m = embedder.encode_documents(["one two", "three four five", "six"])
    assert m.shape == (3, embedder.dim)
    assert m.dtype == np.dtype("float32")
    np.testing.assert_allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)


def test_query_encoding_is_a_unit_vector(embedder) -> None:
    q = embedder.encode_query("what is attention")
    assert q.shape == (embedder.dim,)
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-5)


def test_identical_text_is_maximally_similar(embedder) -> None:
    text = "self attention computes queries keys and values"
    doc = embedder.encode_documents([text])[0]
    assert float(doc @ embedder.encode_query(text)) == pytest.approx(1.0, abs=1e-5)


def test_overlapping_text_beats_disjoint_text(embedder) -> None:
    docs = embedder.encode_documents([
        "attention lets the model focus on relevant tokens",
        "bananas are a tropical fruit grown near the equator",
    ])
    q = embedder.encode_query("how does attention focus on tokens")
    assert float(docs[0] @ q) > float(docs[1] @ q)


def test_empty_input_returns_empty_matrix(embedder) -> None:
    assert embedder.encode_documents([]).shape == (0, embedder.dim)


def test_model_id_is_stable_and_dim_sensitive() -> None:
    assert HashingEmbedder(dim=64).model_id == HashingEmbedder(dim=64).model_id
    assert HashingEmbedder(dim=64).model_id != HashingEmbedder(dim=128).model_id


def test_l2_normalize_handles_zero_vectors() -> None:
    out = l2_normalize(np.zeros((2, 4), dtype="float32"))
    assert np.isfinite(out).all()