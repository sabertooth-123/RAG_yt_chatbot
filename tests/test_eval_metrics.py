import pytest

from ytchat.evaluation.dataset import Benchmark, EvalCase, TimeSpan, _to_seconds
from ytchat.evaluation.retrieval_metrics import (
    citation_precision,
    is_relevant,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    span_coverage,
    span_iou,
)
from ytchat.errors import ConfigurationError
from ytchat.models import Chunk, Citation, ScoredChunk


def _chunk(start, end, index=0):
    return Chunk(index=index, text="x", start_s=start, end_s=end,
                 seg_start=0, seg_end=0, chunk_id=index + 1)


def _hit(start, end, index=0, rank=0):
    return ScoredChunk(chunk=_chunk(start, end, index), score=1.0 - rank * 0.1,
                       rank=rank, retriever="test")


# ---------- time parsing ---------------------------------------------------
@pytest.mark.parametrize("value,expected", [
    (90, 90.0), ("90", 90.0), ("1:30", 90.0), ("1:02:07", 3727.0), ("0:07", 7.0),
])
def test_to_seconds(value, expected) -> None:
    assert _to_seconds(value) == expected


def test_timespan_parse_forms() -> None:
    assert TimeSpan.parse([10, 20]) == TimeSpan(10.0, 20.0)
    assert TimeSpan.parse("1:30-2:00") == TimeSpan(90.0, 120.0)
    assert TimeSpan.parse({"start": 5, "end": 9}) == TimeSpan(5.0, 9.0)


def test_inverted_span_rejected() -> None:
    with pytest.raises(ConfigurationError):
        TimeSpan(20.0, 10.0)


# ---------- relevance ------------------------------------------------------
def test_coverage_is_lenient_iou_is_strict() -> None:
    """A 5-minute chunk containing a 20-second answer fully covers it but has
    terrible IoU — this asymmetry is the whole point."""
    span = TimeSpan(100.0, 120.0)
    wide = _chunk(0.0, 300.0)
    assert span_coverage(wide, span) == pytest.approx(1.0)
    assert span_iou(0.0, 300.0, span) == pytest.approx(20 / 300)


def test_disjoint_spans_score_zero() -> None:
    span = TimeSpan(100.0, 120.0)
    assert span_coverage(_chunk(200.0, 260.0), span) == 0.0
    assert span_iou(200.0, 260.0, span) == 0.0


def test_is_relevant_uses_any_gold_span() -> None:
    spans = [TimeSpan(10, 20), TimeSpan(100, 120)]
    assert is_relevant(_chunk(105, 115), spans)
    assert not is_relevant(_chunk(50, 60), spans)


# ---------- metrics --------------------------------------------------------
def test_recall_is_span_level_not_document_level() -> None:
    """Two gold spans, only one retrieved → 0.5, not 1.0."""
    spans = [TimeSpan(10, 20), TimeSpan(100, 120)]
    hits = [_hit(10, 20, 0, 0)]
    assert recall_at_k(hits, spans, k=5) == pytest.approx(0.5)

    hits.append(_hit(100, 120, 1, 1))
    assert recall_at_k(hits, spans, k=5) == pytest.approx(1.0)


def test_recall_respects_k() -> None:
    spans = [TimeSpan(100, 120)]
    hits = [_hit(0, 5, 0, 0), _hit(6, 10, 1, 1), _hit(100, 120, 2, 2)]
    assert recall_at_k(hits, spans, k=2) == 0.0
    assert recall_at_k(hits, spans, k=3) == pytest.approx(1.0)


def test_precision_at_k() -> None:
    spans = [TimeSpan(100, 120)]
    hits = [_hit(100, 120, 0, 0), _hit(0, 5, 1, 1)]
    assert precision_at_k(hits, spans, k=2) == pytest.approx(0.5)


def test_mrr_rewards_early_hits() -> None:
    spans = [TimeSpan(100, 120)]
    assert reciprocal_rank([_hit(100, 120, 0, 0)], spans) == pytest.approx(1.0)
    assert reciprocal_rank([_hit(0, 5, 0, 0), _hit(100, 120, 1, 1)], spans) == pytest.approx(0.5)
    assert reciprocal_rank([_hit(0, 5, 0, 0)], spans) == 0.0


def test_ndcg_is_ordering_sensitive() -> None:
    spans = [TimeSpan(100, 120)]
    first = ndcg_at_k([_hit(100, 120, 0, 0), _hit(0, 5, 1, 1)], spans, k=2)
    second = ndcg_at_k([_hit(0, 5, 0, 0), _hit(100, 120, 1, 1)], spans, k=2)
    assert first > second


def test_ndcg_never_exceeds_one_with_overlapping_hits() -> None:
    """Several retrieved chunks can overlap one gold span.  Normalising by the
    span count instead of the relevant-hit count made this return 1.01 on the
    tokenizer benchmark."""
    spans = [TimeSpan(100, 200)]
    hits = [_hit(100, 130, 0, 0), _hit(130, 160, 1, 1), _hit(160, 200, 2, 2)]
    assert ndcg_at_k(hits, spans, k=5) == pytest.approx(1.0)


@pytest.mark.parametrize("n_spans,n_hits", [(1, 5), (2, 5), (3, 3), (5, 2)])
def test_ndcg_is_bounded(n_spans, n_hits) -> None:
    spans = [TimeSpan(i * 100, i * 100 + 50) for i in range(n_spans)]
    hits = [_hit(i * 100, i * 100 + 50, i, i) for i in range(n_hits)]
    assert 0.0 <= ndcg_at_k(hits, spans, k=5) <= 1.0


def test_empty_inputs_do_not_crash() -> None:
    assert recall_at_k([], [TimeSpan(1, 2)], k=5) == 0.0
    assert recall_at_k([_hit(1, 2)], [], k=5) == 0.0
    assert precision_at_k([], [TimeSpan(1, 2)], k=5) == 0.0


# ---------- citations ------------------------------------------------------
def test_citation_precision_rewards_tight_timestamps() -> None:
    spans = [TimeSpan(100, 120)]

    def cite(start, end):
        return Citation(marker=1, chunk_id=1, start_s=start, end_s=end,
                        url="u", timestamp="t", quote="q")

    tight = citation_precision([cite(100, 120)], spans)
    loose = citation_precision([cite(0, 300)], spans)
    assert tight == pytest.approx(1.0)
    assert loose < 0.2


# ---------- benchmark validation ------------------------------------------
def test_answerable_case_without_spans_is_invalid() -> None:
    bench = Benchmark(
        name="b", video_url="https://youtu.be/dQw4w9WgXcQ",
        cases=(EvalCase(id="a", question="q?", answerable=True),),
    )
    assert any("relevant_spans" in p for p in bench.validate())


def test_unanswerable_case_with_spans_is_invalid() -> None:
    bench = Benchmark(
        name="b", video_url="https://youtu.be/dQw4w9WgXcQ",
        cases=(EvalCase(id="a", question="q?", answerable=False,
                        relevant_spans=(TimeSpan(1, 2),)),),
    )
    assert any("must not have" in p for p in bench.validate())


def test_dangling_followup_is_invalid() -> None:
    bench = Benchmark(
        name="b", video_url="https://youtu.be/dQw4w9WgXcQ",
        cases=(EvalCase(id="a", question="q?", relevant_spans=(TimeSpan(1, 2),),
                        followup_to="ghost"),),
    )
    assert any("does not exist" in p for p in bench.validate())


def test_conversation_order_places_antecedents_first() -> None:
    bench = Benchmark(
        name="b", video_url="https://youtu.be/dQw4w9WgXcQ",
        cases=(
            EvalCase(id="b2", question="why?", relevant_spans=(TimeSpan(1, 2),),
                     followup_to="a1"),
            EvalCase(id="a1", question="what?", relevant_spans=(TimeSpan(1, 2),)),
        ),
    )
    assert [c.id for c in bench.conversation_order()] == ["a1", "b2"]