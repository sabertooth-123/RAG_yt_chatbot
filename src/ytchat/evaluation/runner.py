"""Evaluation orchestration.

Two entry points:

* ``run`` — full retrieval + generation evaluation across retrievers.
* ``calibrate_threshold`` — sweeps ``min_score`` using *only* retrieval
  confidences.  Layer A is a pure retrieval gate, so the whole refusal ROC costs
  zero LLM calls.  That's what makes it practical on a free tier.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from ytchat.config import Settings
from ytchat.database.repository import Repository
from ytchat.evaluation.dataset import Benchmark, EvalCase
from ytchat.evaluation.generation_metrics import (
    GenerationScores,
    LLMJudge,
    mean_generation,
)
from ytchat.evaluation.retrieval_metrics import (
    RetrievalScores,
    citation_hit_rate,
    citation_precision,
    mean_scores,
)
from ytchat.generation.llm import LLM, build_llm
from ytchat.models import Answer
from ytchat.pipeline import ChatSession, VideoIndex, ensure_indexed
from ytchat.retrieval.base import best_confidence


@dataclass
class CaseResult:
    case: EvalCase
    retrieval: RetrievalScores
    generation: GenerationScores | None = None
    answer: Answer | None = None
    top_confidence: float = 0.0
    citation_precision: float = 0.0
    citation_hit_rate: float = 0.0
    refused: bool = False
    latency_ms: float = 0.0


@dataclass
class RetrieverResult:
    retriever: str
    cases: list[CaseResult] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def retrieval(self) -> RetrievalScores:
        return mean_scores([c.retrieval for c in self.cases if c.case.answerable])

    @property
    def generation(self) -> GenerationScores:
        return mean_generation(
            [c.generation for c in self.cases if c.generation is not None]
        )

    @property
    def citation_precision(self) -> float:
        scored = [c.citation_precision for c in self.cases if c.case.answerable and c.answer]
        return sum(scored) / len(scored) if scored else 0.0

    @property
    def correct_refusal_rate(self) -> float:
        """Of the unanswerable cases, how many were refused."""
        cases = [c for c in self.cases if not c.case.answerable and c.answer]
        return sum(1 for c in cases if c.refused) / len(cases) if cases else 0.0

    @property
    def false_refusal_rate(self) -> float:
        """Of the answerable cases, how many were wrongly refused."""
        cases = [c for c in self.cases if c.case.answerable and c.answer]
        return sum(1 for c in cases if c.refused) / len(cases) if cases else 0.0


@dataclass
class ThresholdPoint:
    threshold: float
    correct_refusal_rate: float
    false_refusal_rate: float

    @property
    def youden_j(self) -> float:
        """Correct-refusal minus false-refusal; maximised at the best operating point."""
        return self.correct_refusal_rate - self.false_refusal_rate


class EvalRunner:
    def __init__(
        self,
        benchmark: Benchmark,
        settings: Settings,
        repo: Repository,
        llm: LLM | None = None,
        judge: LLM | None = None,
        progress=lambda _m: None,
    ) -> None:
        self.benchmark = benchmark
        self.settings = settings
        self.repo = repo
        self.progress = progress
        self._llm = llm
        self._judge_llm = judge
        self._index: VideoIndex | None = None

    # -- shared index: built once, reused by every retriever ---------------
    def index(self) -> VideoIndex:
        if self._index is None:
            self._index = ensure_indexed(
                self.benchmark.video_url, self.settings, self.repo, progress=self.progress
            )
        return self._index

    # -- retrieval-only (no LLM calls) -------------------------------------
    def evaluate_retrieval(self, retriever_name: str, k: int | None = None) -> RetrieverResult:
        started = time.perf_counter()
        index = self.index()
        retriever = index.retriever(retriever_name)
        k = k or self.settings.top_k
        result = RetrieverResult(retriever=retriever_name)

        for case in self.benchmark.cases:
            # Must match production exactly: the Answerer's Layer A gate sees the
            # top-k hits, and best_confidence maxes over whatever list it is
            # given.  Searching to candidate depth here inflates top_confidence
            # and makes these numbers disagree with the calibration sweep.
            hits = retriever.search(case.question, k)
            result.cases.append(
                CaseResult(
                    case=case,
                    retrieval=RetrievalScores.compute(hits, case.relevant_spans, k=k),
                    top_confidence=best_confidence(hits),
                )
            )
        result.elapsed_s = time.perf_counter() - started
        return result

    # -- full pipeline including generation --------------------------------
    def evaluate_full(
        self, retriever_name: str, k: int | None = None, judge: bool = True
    ) -> RetrieverResult:
        started = time.perf_counter()
        index = self.index()
        k = k or self.settings.top_k
        llm = self._llm or build_llm(self.settings)
        judger = LLMJudge(self._judge_llm or llm) if judge else None

        result = RetrieverResult(retriever=retriever_name)
        session = ChatSession(index, llm=llm, retriever_name=retriever_name)
        by_id: dict[str, CaseResult] = {}

        for case in self.benchmark.conversation_order():
            # Follow-ups need the antecedent in history; standalone cases must
            # not inherit unrelated context, so history is reset between chains.
            if not case.followup_to:
                session.clear_history()
            self.progress(f"[{retriever_name}] {case.id}: {case.question[:60]}")

            answer = session.ask(case.question, k=k)
            retrieval = RetrievalScores.compute(
                list(answer.retrieved), case.relevant_spans, k=k
            )
            case_result = CaseResult(
                case=case,
                retrieval=retrieval,
                answer=answer,
                refused=answer.refused,
                top_confidence=best_confidence(answer.retrieved),
                citation_precision=citation_precision(answer.citations, case.relevant_spans),
                citation_hit_rate=citation_hit_rate(answer.citations, case.relevant_spans),
                latency_ms=answer.latency_ms,
            )
            if judger is not None:
                case_result.generation = judger.score(
                    case.question, answer, answer.retrieved, case.expected_answer
                )
            result.cases.append(case_result)
            by_id[case.id] = case_result

        result.elapsed_s = time.perf_counter() - started
        return result

    def compare(
        self,
        retrievers: Sequence[str] = ("dense", "sparse", "hybrid"),
        full: bool = True,
        k: int | None = None,
        judge: bool = True,
    ) -> list[RetrieverResult]:
        return [
            self.evaluate_full(name, k=k, judge=judge) if full
            else self.evaluate_retrieval(name, k=k)
            for name in retrievers
        ]

    # -- refusal calibration (free: no LLM calls) --------------------------
    def calibrate_threshold(
        self,
        retriever_name: str | None = None,
        thresholds: Sequence[float] | None = None,
        k: int | None = None,
    ) -> list[ThresholdPoint]:
        """Sweep ``min_score`` over the Layer A gate.

        Layer A depends only on retrieval confidence, so this measures the full
        refusal trade-off without generating a single token.
        """
        index = self.index()
        retriever = index.retriever(retriever_name or self.settings.retriever)
        k = k or self.settings.top_k
        thresholds = thresholds or [i / 20 for i in range(21)]

        # Follow-ups are excluded: this sweep calls the retriever directly, so no
        # query rewriting happens and "Why does that work?" retrieves almost
        # nothing.  Including them would inflate the false-refusal rate with a
        # measurement artefact rather than a real gate failure.
        confidences: list[tuple[EvalCase, float]] = [
            (case, best_confidence(retriever.search(case.question, k)))
            for case in self.benchmark.cases
            if not case.followup_to
        ]
        answerable = [c for c in confidences if c[0].answerable]
        unanswerable = [c for c in confidences if not c[0].answerable]

        points: list[ThresholdPoint] = []
        for t in thresholds:
            points.append(
                ThresholdPoint(
                    threshold=t,
                    correct_refusal_rate=(
                        sum(1 for _, conf in unanswerable if conf < t) / len(unanswerable)
                        if unanswerable else 0.0
                    ),
                    false_refusal_rate=(
                        sum(1 for _, conf in answerable if conf < t) / len(answerable)
                        if answerable else 0.0
                    ),
                )
            )
        return points


def best_threshold(
    points: Sequence[ThresholdPoint], max_false_refusal: float = 0.15
) -> ThresholdPoint | None:
    """Highest correct-refusal rate subject to a false-refusal budget.

    Deliberately *not* Youden J.  That objective assumes a missed refusal and a
    false refusal cost the same, and in this architecture they do not: Layer A is
    a cost optimisation, while Layers B and C are the actual safety net.  A
    missed refusal costs one API call and is usually caught downstream; a false
    refusal denies the user an answer the system had already retrieved.

    Measured on the benchmark, the Youden-J optimum (0.65) refused questions
    whose recall@k was 1.00 -- the right passage was found and then discarded.

    Falls back to the Youden-J point only if no threshold meets the budget.
    """
    if not points:
        return None
    affordable = [p for p in points if p.false_refusal_rate <= max_false_refusal]
    if not affordable:
        return max(points, key=lambda p: p.youden_j)
    return max(affordable, key=lambda p: (p.correct_refusal_rate, -p.threshold))