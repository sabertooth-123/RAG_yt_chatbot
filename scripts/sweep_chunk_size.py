"""Chunk-size sweep: recall vs citation precision.

Free to run -- retrieval metrics only, no LLM calls.  Each configuration
re-chunks and re-embeds (the transcript is never re-downloaded, thanks to the
staged cache fingerprints), then scores retrieval against the gold time spans.

The hypothesis under test: larger chunks raise Recall@k because a passage is
more likely to contain the answer, while lowering timestamp IoU because the
citation points at more than the answer.  If both move together instead, the
chunking design needs rethinking.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ytchat.config import Settings  # noqa: E402
from ytchat.database.repository import Repository  # noqa: E402
from ytchat.evaluation.dataset import Benchmark  # noqa: E402
from ytchat.evaluation.retrieval_metrics import (  # noqa: E402
    RetrievalScores,
    citation_precision,
    mean_scores,
    span_iou,
)
from ytchat.pipeline import ensure_indexed  # noqa: E402

BENCHMARKS = ["karpathy_llm_intro", "karpathy_tokenizer", "karpathy_deepdive"]
CHUNK_SIZES = [400, 700, 1000, 1500]
RETRIEVERS = ["dense", "hybrid"]
OUT = ROOT / "results" / "chunk_sweep.csv"


def chunk_iou_proxy(hits, spans) -> float:
    """Best IoU between each retrieved chunk and any gold span.

    A stand-in for citation precision that needs no generation: a citation
    inherits its chunk's time span, so this is the ceiling the citation system
    can reach for a given chunk size.
    """
    if not hits or not spans:
        return 0.0
    return sum(
        max(span_iou(h.chunk.start_s, h.chunk.end_s, s) for s in spans) for h in hits
    ) / len(hits)


def main() -> int:
    rows = []
    for size in CHUNK_SIZES:
        overlap = int(size * 0.17)          # hold overlap ratio constant
        for bench_name in BENCHMARKS:
            bench = Benchmark.from_yaml(ROOT / "benchmarks" / f"{bench_name}.yaml")
            settings = Settings(max_chars=size, overlap_chars=overlap,
                                min_chars=min(250, size // 3))
            with Repository(settings.db_path) as repo:
                index = ensure_indexed(bench.video_url, settings, repo,
                                       progress=lambda m: None)
                for retriever_name in RETRIEVERS:
                    retriever = index.retriever(retriever_name)
                    scores, ious = [], []
                    for case in bench.answerable:
                        hits = retriever.search(case.question, settings.top_k)
                        scores.append(
                            RetrievalScores.compute(hits, case.relevant_spans,
                                                    k=settings.top_k)
                        )
                        ious.append(chunk_iou_proxy(hits, case.relevant_spans))
                    agg = mean_scores(scores)
                    rows.append({
                        "max_chars": size,
                        "benchmark": bench_name,
                        "retriever": retriever_name,
                        "n_chunks": index.stats.n_chunks,
                        "mean_chunk_s": round(
                            sum(c.span_s for c in index.chunks) / len(index.chunks), 1
                        ),
                        "recall_at_k": round(agg.recall_at_k, 4),
                        "precision_at_k": round(agg.precision_at_k, 4),
                        "mrr": round(agg.mrr, 4),
                        "ndcg_at_k": round(agg.ndcg_at_k, 4),
                        "citation_iou": round(sum(ious) / len(ious), 4),
                    })
                    print(f"  {size:>5} {bench_name:<20} {retriever_name:<7} "
                          f"chunks={index.stats.n_chunks:>4} "
                          f"recall={agg.recall_at_k:.2f} "
                          f"iou={sum(ious)/len(ious):.3f}", flush=True)

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} rows -> {OUT}\n")
    print("Pooled across videos:")
    print(f"{'max_chars':>10}{'chunk_s':>9}{'recall':>9}{'cite_iou':>10}")
    for size in CHUNK_SIZES:
        sub = [r for r in rows if r["max_chars"] == size and r["retriever"] == "dense"]
        n = len(sub)
        print(f"{size:>10}{sum(r['mean_chunk_s'] for r in sub)/n:>9.1f}"
              f"{sum(r['recall_at_k'] for r in sub)/n:>9.2f}"
              f"{sum(r['citation_iou'] for r in sub)/n:>10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
