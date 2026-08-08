import pytest

from ytchat.retrieval.base import bm25_confidence, confidence_of, cosine_confidence
from ytchat.retrieval.sparse import BM25Index, tokenize


# ---------- tokenizer ------------------------------------------------------
def test_tokenizer_splits_and_keeps_compounds() -> None:
    toks = tokenize("Self-attention uses queries.")
    assert "self-attention" in toks and "self" in toks and "attention" in toks


def test_tokenizer_drops_stopwords() -> None:
    assert "the" not in tokenize("the model and the data")


# ---------- BM25 -----------------------------------------------------------
def test_bm25_ranks_the_matching_document_first() -> None:
    idx = BM25Index([
        "positional encodings inject order information",
        "multi head attention runs layers in parallel",
        "gradient descent updates the weights",
    ])
    scores = idx.score_all("what are positional encodings")
    assert scores[0] == max(scores) and scores[0] > 0


def test_bm25_returns_zero_when_nothing_matches() -> None:
    idx = BM25Index(["alpha beta", "gamma delta"])
    assert all(s == 0.0 for s in idx.score_all("kangaroo marsupial"))


def test_bm25_idf_is_never_negative() -> None:
    idx = BM25Index(["common word here"] * 5)
    assert all(v >= 0 for v in idx.idf.values())


# ---------- confidence calibration ----------------------------------------
def test_confidence_is_bounded_and_monotonic() -> None:
    assert cosine_confidence(-0.4) == 0.0
    assert cosine_confidence(1.5) == 1.0
    assert bm25_confidence(0.0) == 0.0
    assert 0.0 < bm25_confidence(4.0) < bm25_confidence(12.0) < 1.0


# ---------- retrievers -----------------------------------------------------
@pytest.mark.parametrize("name", ["dense", "sparse", "hybrid"])
def test_all_retrievers_share_the_same_contract(index, name) -> None:
    hits = index.retriever(name).search("what is multi head attention", k=3)
    assert hits, f"{name} returned nothing"
    assert len(hits) <= 3
    assert [h.rank for h in hits] == list(range(len(hits)))
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    for h in hits:
        assert 0.0 <= confidence_of(h) <= 1.0
        assert h.chunk.end_s >= h.chunk.start_s      # the invariant survives retrieval
        assert h.chunk.chunk_id is not None


@pytest.mark.parametrize("name", ["dense", "sparse", "hybrid"])
def test_retrievers_find_the_relevant_moment(index, name) -> None:
    hits = index.retriever(name).search("positional encodings", k=3)
    assert any("positional" in h.chunk.text.lower() for h in hits)


@pytest.mark.parametrize("name", ["dense", "sparse", "hybrid"])
def test_empty_query_returns_nothing(index, name) -> None:
    assert index.retriever(name).search("   ", k=5) == []


def test_offtopic_query_has_low_confidence(index) -> None:
    hits = index.retriever("sparse").search("recipe for sourdough bread", k=5)
    assert not hits or max(confidence_of(h) for h in hits) < 0.3


# ---------- hybrid ---------------------------------------------------------
def test_hybrid_unions_both_candidate_lists(index) -> None:
    query = "how do heads learn different relationships"
    # Hybrid searches its sub-retrievers at candidate_k depth, so the comparison
    # lists must be pulled at that same depth — otherwise chunks ranked 6-10 by
    # a sub-retriever look like the fusion invented them.
    depth = max(index.settings.candidate_k, 10)
    dense = {h.chunk.chunk_id for h in index.retriever("dense").search(query, depth)}
    sparse = {h.chunk.chunk_id for h in index.retriever("sparse").search(query, depth)}
    hybrid = {h.chunk.chunk_id for h in index.retriever("hybrid").search(query, 10)}
    assert hybrid & dense and hybrid & sparse
    assert hybrid <= (dense | sparse), "hybrid must not invent chunks"


def test_rrf_rewards_agreement_between_retrievers(index) -> None:
    query = "scaled dot product square root dimension"
    top_dense = index.retriever("dense").search(query, 3)[0].chunk.chunk_id
    top_sparse = index.retriever("sparse").search(query, 3)[0].chunk.chunk_id
    top_hybrid = index.retriever("hybrid").search(query, 3)[0].chunk.chunk_id
    if top_dense == top_sparse:
        assert top_hybrid == top_dense, "both retrievers agreed; fusion must not disagree"


def test_hybrid_records_component_scores(index) -> None:
    hits = index.retriever("hybrid").search("attention mechanism", k=3)
    comps = hits[0].components
    assert "fusion" in comps and "confidence" in comps
    assert "dense_score" in comps or "sparse_score" in comps


def test_weighted_fusion_alpha_shifts_the_ranking(index, settings) -> None:
    from ytchat.retrieval.hybrid import HybridRetriever

    dense = index.retriever("dense")
    sparse = index.retriever("sparse")
    query = "gradient stability during training"

    dense_only = HybridRetriever(dense, sparse, fusion="weighted", alpha=1.0).search(query, 3)
    sparse_only = HybridRetriever(dense, sparse, fusion="weighted", alpha=0.0).search(query, 3)
    assert dense_only[0].chunk.chunk_id == dense.search(query, 1)[0].chunk.chunk_id
    assert sparse_only[0].chunk.chunk_id == sparse.search(query, 1)[0].chunk.chunk_id