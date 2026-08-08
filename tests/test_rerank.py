import pytest

from ytchat.retrieval.base import CONFIDENCE_KEY, confidence_of
from ytchat.retrieval.rerank import HashingScorer, RerankingRetriever, _sigmoid


def _reranked(index, base="hybrid", candidates=10):
    return RerankingRetriever(index.retriever(base), HashingScorer(), candidates=candidates)


def test_sigmoid_is_bounded_and_monotonic() -> None:
    assert _sigmoid(-20) == pytest.approx(0.0, abs=1e-6)
    assert _sigmoid(0) == pytest.approx(0.5)
    assert _sigmoid(20) == pytest.approx(1.0, abs=1e-6)
    assert _sigmoid(-1) < _sigmoid(0) < _sigmoid(1)


def test_sigmoid_does_not_overflow_on_large_negatives() -> None:
    assert _sigmoid(-1000) == 0.0


def test_rerank_preserves_the_retriever_contract(index) -> None:
    hits = _reranked(index).search("what is multi head attention", k=3)
    assert hits and len(hits) <= 3
    assert [h.rank for h in hits] == list(range(len(hits)))
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    for h in hits:
        assert 0.0 <= confidence_of(h) <= 1.0
        assert h.chunk.end_s >= h.chunk.start_s      # the invariant survives reranking
        assert h.chunk.chunk_id is not None


def test_rerank_never_invents_chunks(index) -> None:
    query = "positional encodings"
    pool = {h.chunk.chunk_id for h in index.retriever("hybrid").search(query, 10)}
    reranked = {h.chunk.chunk_id for h in _reranked(index).search(query, 5)}
    assert reranked <= pool


def test_rerank_returns_no_duplicates(index) -> None:
    hits = _reranked(index).search("each head learns a different relationship", k=5)
    ids = [h.chunk.chunk_id for h in hits]
    assert len(ids) == len(set(ids))


def test_provenance_of_the_base_retriever_is_kept(index) -> None:
    hits = _reranked(index).search("attention", k=3)
    comps = hits[0].components
    assert "base_score" in comps and "base_rank" in comps
    assert "rerank_logit" in comps and CONFIDENCE_KEY in comps


def test_confidence_matches_the_sigmoid_of_the_logit(index) -> None:
    hit = _reranked(index).search("attention", k=1)[0]
    assert confidence_of(hit) == pytest.approx(_sigmoid(hit.components["rerank_logit"]))


def test_empty_query_returns_nothing(index) -> None:
    assert _reranked(index).search("   ", k=5) == []


@pytest.mark.parametrize("base", ["dense", "sparse", "hybrid"])
def test_rerank_composes_with_every_base_retriever(index, base) -> None:
    retriever = _reranked(index, base=base)
    assert retriever.name.endswith("+rerank")
    hits = retriever.search("positional encodings", k=3)
    assert all(0.0 <= confidence_of(h) <= 1.0 for h in hits)


def test_factory_wraps_when_enabled(index, settings, embedder) -> None:
    """The config flag must actually change what build_retriever returns."""
    from ytchat.retrieval.factory import build_retriever
    from ytchat.retrieval.rerank import RerankingRetriever as RR

    plain = build_retriever("hybrid", settings, index.chunks, index.store, embedder)
    assert not isinstance(plain, RR)

    on = settings.model_copy(update={"enable_rerank": True})
    wrapped = build_retriever("hybrid", on, index.chunks, index.store, embedder)
    assert isinstance(wrapped, RR)
    assert wrapped.candidates == on.rerank_candidates
