"""Retrieval metrics against time-range ground truth.

Two relevance notions, deliberately different:

* ``span_coverage`` — overlap / min(durations).  Lenient: "does this chunk
  contain the answer?"  Drives Recall/Precision/MRR/nDCG.
* ``span_iou`` — strict intersection-over-union.  Drives citation precision,
  and penalises a chunk that technically contains the answer but spans four
  minutes around it.

Recall therefore rises with chunk size while citation precision falls.  Reporting
only one of them is how RAG projects claim wins they did not earn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ytchat.evaluation.dataset import TimeSpan
from ytchat.models import Chunk, Citation, ScoredChunk

DEFAULT_THRESHOLD = 0.30


def overlap_seconds(a: TimeSpan, b_start: float, b_end: float) -> float:
    return max(0.0, min(a.end_s, b_end) - max(a.start_s, b_start))


def span_coverage(chunk: Chunk, span: TimeSpan) -> float:
    """Overlap normalised by the *shorter* of the two durations."""
    inter = overlap_seconds(span, chunk.start_s, chunk.end_s)
    if inter <= 0:
        return 0.0
    denom = min(chunk.span_s, span.duration_s)
    return inter / denom if denom > 0 else 0.0


def span_iou(chunk_start: float, chunk_end: float, span: TimeSpan) -> float:
    inter = overlap_seconds(span, chunk_start, chunk_end)
    if inter <= 0:
        return 0.0
    union = max(chunk_end, span.end_s) - min(chunk_start, span.start_s)
    return inter / union if union > 0 else 0.0


def is_relevant(
    chunk: Chunk, spans: Sequence[TimeSpan], threshold: float = DEFAULT_THRESHOLD
) -> bool:
    return any(span_coverage(chunk, s) >= threshold for s in spans)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def recall_at_k(
    hits: Sequence[ScoredChunk],
    spans: Sequence[TimeSpan],
    k: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    """Fraction of gold spans covered by at least one of the top-k chunks.

    Span-level rather than document-level: a question whose answer is split
    across two moments is only fully recalled when both are retrieved.
    """
    if not spans:
        return 0.0
    top = hits[:k]
    covered = sum(
        1 for s in spans if any(span_coverage(h.chunk, s) >= threshold for h in top)
    )
    return covered / len(spans)


def precision_at_k(
    hits: Sequence[ScoredChunk],
    spans: Sequence[TimeSpan],
    k: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    top = hits[:k]
    if not top:
        return 0.0
    return sum(1 for h in top if is_relevant(h.chunk, spans, threshold)) / len(top)


def reciprocal_rank(
    hits: Sequence[ScoredChunk],
    spans: Sequence[TimeSpan],
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    for i, hit in enumerate(hits, start=1):
        if is_relevant(hit.chunk, spans, threshold):
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    hits: Sequence[ScoredChunk],
    spans: Sequence[TimeSpan],
    k: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    """Binary-gain nDCG — rewards putting relevant chunks near the top.

    The ideal ranking is the *same retrieved set* reordered so every relevant
    chunk sits at the front.  Normalising by ``len(spans)`` instead is wrong and
    can produce nDCG > 1, because several retrieved chunks may overlap a single
    gold span, letting DCG accumulate more terms than IDCG has.  The tokenizer
    benchmark returned 1.01 under the old formula, which is what surfaced it.
    """
    top = hits[:k]
    gains = [1.0 if is_relevant(h.chunk, spans, threshold) else 0.0 for h in top]
    dcg = sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))
    n_relevant = int(sum(gains))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_relevant + 1))
    return dcg / idcg if idcg > 0 else 0.0


def citation_precision(citations: Sequence[Citation], spans: Sequence[TimeSpan]) -> float:
    """Mean best-IoU of emitted citations against gold spans.

    This is the metric almost nobody reports: not "was the answer right" but
    "does the timestamp actually take the viewer to the right moment".
    """
    if not citations or not spans:
        return 0.0
    return sum(
        max(span_iou(c.start_s, c.end_s, s) for s in spans) for c in citations
    ) / len(citations)


def citation_hit_rate(
    citations: Sequence[Citation], spans: Sequence[TimeSpan], threshold: float = 0.1
) -> float:
    """Fraction of citations that land on a gold span at all."""
    if not citations:
        return 0.0
    return sum(
        1 for c in citations
        if any(span_iou(c.start_s, c.end_s, s) >= threshold for s in spans)
    ) / len(citations)


@dataclass
class RetrievalScores:
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    k: int = 5

    @classmethod
    def compute(
        cls,
        hits: Sequence[ScoredChunk],
        spans: Sequence[TimeSpan],
        k: int = 5,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> "RetrievalScores":
        return cls(
            recall_at_k=recall_at_k(hits, spans, k, threshold),
            precision_at_k=precision_at_k(hits, spans, k, threshold),
            mrr=reciprocal_rank(hits, spans, threshold),
            ndcg_at_k=ndcg_at_k(hits, spans, k, threshold),
            k=k,
        )


def mean_scores(scores: Sequence[RetrievalScores]) -> RetrievalScores:
    if not scores:
        return RetrievalScores()
    n = len(scores)
    return RetrievalScores(
        recall_at_k=sum(s.recall_at_k for s in scores) / n,
        precision_at_k=sum(s.precision_at_k for s in scores) / n,
        mrr=sum(s.mrr for s in scores) / n,
        ndcg_at_k=sum(s.ndcg_at_k for s in scores) / n,
        k=scores[0].k,
    )