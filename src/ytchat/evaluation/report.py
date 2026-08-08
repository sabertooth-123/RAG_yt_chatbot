"""Report rendering: markdown for the README, CSV for plotting."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ytchat.evaluation.dataset import Benchmark
from ytchat.evaluation.runner import RetrieverResult, ThresholdPoint, best_threshold


def _row(values: Sequence[str]) -> str:
    return "| " + " | ".join(values) + " |"


def comparison_table(results: Sequence[RetrieverResult]) -> str:
    headers = [
        "Retriever", "Recall@k", "Precision@k", "MRR", "nDCG@k",
        "CitePrec", "Faithful", "AnsRel", "FalseRefuse", "CorrRefuse",
    ]
    lines = [_row(headers), _row(["---"] * len(headers))]
    for r in results:
        ret, gen = r.retrieval, r.generation
        lines.append(_row([
            r.retriever,
            f"{ret.recall_at_k:.2f}",
            f"{ret.precision_at_k:.2f}",
            f"{ret.mrr:.2f}",
            f"{ret.ndcg_at_k:.2f}",
            f"{r.citation_precision:.2f}",
            f"{gen.faithfulness:.2f}",
            f"{gen.answer_relevance:.2f}",
            f"{r.false_refusal_rate:.2f}",
            f"{r.correct_refusal_rate:.2f}",
        ]))
    return "\n".join(lines)


def threshold_table(points: Sequence[ThresholdPoint]) -> str:
    lines = [
        _row(["min_score", "Correct refusal", "False refusal", "Youden J"]),
        _row(["---"] * 4),
    ]
    for p in points:
        lines.append(_row([
            f"{p.threshold:.2f}",
            f"{p.correct_refusal_rate:.2f}",
            f"{p.false_refusal_rate:.2f}",
            f"{p.youden_j:+.2f}",
        ]))
    return "\n".join(lines)


def failure_digest(results: Sequence[RetrieverResult], limit: int = 8) -> str:
    """The cases worth reading by hand — where the numbers won't tell you why."""
    lines: list[str] = []
    for r in results:
        bad = [
            c for c in r.cases
            if (c.case.answerable and (c.refused or c.retrieval.recall_at_k == 0.0))
            or (not c.case.answerable and not c.refused)
        ][:limit]
        if not bad:
            continue
        lines.append(f"\n### {r.retriever}\n")
        for c in bad:
            kind = (
                "FALSE REFUSAL" if c.case.answerable and c.refused
                else "RETRIEVAL MISS" if c.case.answerable
                else "MISSED REFUSAL"
            )
            lines.append(
                f"- **{kind}** `{c.case.id}` — {c.case.question}\n"
                f"  - top confidence: {c.top_confidence:.3f}, "
                f"recall@k: {c.retrieval.recall_at_k:.2f}"
            )
    return "\n".join(lines) if lines else "\nNo failures.\n"


@dataclass
class Report:
    benchmark: Benchmark
    results: list[RetrieverResult]
    thresholds: list[ThresholdPoint] | None = None
    settings_summary: dict[str, object] | None = None

    def to_markdown(self) -> str:
        parts = [
            f"# Evaluation — {self.benchmark.name}",
            "",
            f"**Video:** {self.benchmark.video_url}  ",
            f"**Cases:** {len(self.benchmark.cases)} "
            f"({len(self.benchmark.answerable)} answerable, "
            f"{len(self.benchmark.unanswerable)} unanswerable)",
            "",
        ]
        if self.settings_summary:
            parts.append("**Configuration:** " + ", ".join(
                f"`{k}={v}`" for k, v in self.settings_summary.items()
            ))
            parts.append("")

        parts += ["## Retriever comparison", "", comparison_table(self.results), ""]

        if self.thresholds:
            best = best_threshold(self.thresholds)
            parts += [
                "## Refusal threshold calibration", "",
                threshold_table(self.thresholds), "",
            ]
            if best:
                parts.append(
                    f"**Recommended `min_score`: {best.threshold:.2f}** "
                    f"(correct refusal {best.correct_refusal_rate:.0%}, "
                    f"false refusal {best.false_refusal_rate:.0%})"
                )
                parts.append("")

        parts += ["## Failure cases", failure_digest(self.results)]

        timings = ", ".join(f"{r.retriever} {r.elapsed_s:.1f}s" for r in self.results)
        parts += ["", f"_Wall clock: {timings}_"]
        return "\n".join(parts)

    def to_csv(self, path: str | Path) -> None:
        """Per-case rows — the input for every plot in the research section."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "retriever", "case_id", "answerable", "tags", "recall_at_k",
                "precision_at_k", "mrr", "ndcg_at_k", "citation_precision",
                "citation_hit_rate", "top_confidence", "refused",
                "faithfulness", "answer_relevance", "context_relevance",
                "correctness", "latency_ms",
            ])
            for r in self.results:
                for c in r.cases:
                    g = c.generation
                    writer.writerow([
                        r.retriever, c.case.id, int(c.case.answerable),
                        "|".join(c.case.tags),
                        f"{c.retrieval.recall_at_k:.4f}",
                        f"{c.retrieval.precision_at_k:.4f}",
                        f"{c.retrieval.mrr:.4f}",
                        f"{c.retrieval.ndcg_at_k:.4f}",
                        f"{c.citation_precision:.4f}",
                        f"{c.citation_hit_rate:.4f}",
                        f"{c.top_confidence:.4f}",
                        int(c.refused),
                        f"{g.faithfulness:.4f}" if g else "",
                        f"{g.answer_relevance:.4f}" if g else "",
                        f"{g.context_relevance:.4f}" if g else "",
                        f"{g.correctness:.4f}" if g and g.correctness is not None else "",
                        f"{c.latency_ms:.1f}",
                    ])