"""Generation quality: a built-in LLM judge, plus optional RAGAS/DeepEval.

The built-in judge is the default because it works with any provider you already
have configured — including free Groq and local Ollama.  RAGAS and DeepEval
default to OpenAI and need LangChain wrapper plumbing to point elsewhere, so they
are opt-in cross-checks rather than the primary metric.

Using three judges is not redundancy for its own sake: measuring how much they
agree (experiment #10) tells you how much to trust any of them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Sequence

from ytchat.generation.llm import LLM
from ytchat.models import Answer, ScoredChunk

_JSON_BLOCK = re.compile(r"\{.*?\}", re.DOTALL)
_BARE_NUMBER = re.compile(r"(?:score\D{0,10})?([01](?:\.\d+)?)")

JUDGE_SYSTEM = """\
You are a strict evaluator of retrieval-augmented answers. You output only JSON.
Be harsh: partial support is not support. When uncertain, score lower.
Respond with exactly: {"score": <float 0.0-1.0>, "reason": "<one sentence>"}
"""

FAITHFULNESS_PROMPT = """\
Judge whether EVERY claim in the ANSWER is directly supported by the CONTEXT.

Score 1.0 only if every claim is explicitly supported. Score 0.0 if the answer
introduces facts absent from the context. Outside knowledge, plausible
elaboration, and unstated inference all count as unsupported.

CONTEXT:
{context}

ANSWER:
{answer}
"""

ANSWER_RELEVANCE_PROMPT = """\
Judge whether the ANSWER actually addresses the QUESTION.

Score 1.0 if it directly and completely answers what was asked. Penalise
evasion, topic drift, and answers that restate the question without resolving it.
Do not judge factual accuracy here — only relevance.

QUESTION: {question}

ANSWER:
{answer}
"""

CONTEXT_RELEVANCE_PROMPT = """\
Estimate what proportion of the CONTEXT was actually needed to answer the QUESTION.

Score 1.0 if nearly all of it was necessary, 0.0 if almost none was. This
measures retrieval precision as experienced by the generator.

QUESTION: {question}

CONTEXT:
{context}
"""

CORRECTNESS_PROMPT = """\
Compare the ANSWER against the REFERENCE answer.

Score 1.0 if they convey the same information, 0.5 for partial agreement, 0.0 if
they contradict or the answer misses the point. Wording differences do not matter.

QUESTION: {question}

REFERENCE:
{reference}

ANSWER:
{answer}
"""


def _parse_score(raw: str) -> tuple[float, str]:
    """Tolerant parsing — a judge that wraps JSON in prose shouldn't zero a run."""
    match = _JSON_BLOCK.search(raw)
    if match:
        try:
            payload = json.loads(match.group(0))
            score = float(payload.get("score", 0.0))
            return max(0.0, min(1.0, score)), str(payload.get("reason", ""))
        except (ValueError, TypeError):
            pass
    fallback = _BARE_NUMBER.search(raw)
    if fallback:
        try:
            return max(0.0, min(1.0, float(fallback.group(1)))), raw.strip()[:160]
        except ValueError:
            pass
    return 0.0, f"unparseable judge output: {raw.strip()[:120]}"


@dataclass
class GenerationScores:
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_relevance: float = 0.0
    correctness: float | None = None
    reasons: dict[str, str] = field(default_factory=dict)


COMBINED_PROMPT = """\
Score the ANSWER on four independent criteria. Judge each one separately; a low
score on one must not drag down the others.

faithfulness      - is EVERY claim in the answer directly supported by the
                    CONTEXT? 1.0 only if all are. Outside knowledge, plausible
                    elaboration and unstated inference all count as unsupported.
answer_relevance  - does the answer actually address the QUESTION? Ignore
                    factual accuracy here; penalise evasion and topic drift.
context_relevance - what proportion of the CONTEXT was needed to answer? 1.0 if
                    nearly all of it, 0.0 if almost none.
correctness       - does the answer convey the same information as the
                    REFERENCE? 0.5 for partial, 0.0 if it contradicts or misses
                    the point. Wording differences do not matter. Use null if no
                    reference is given.

QUESTION: {question}

CONTEXT:
{context}

ANSWER:
{answer}

REFERENCE:
{reference}

Respond with exactly this JSON and nothing else:
{{"faithfulness": <float>, "answer_relevance": <float>,
  "context_relevance": <float>, "correctness": <float or null>,
  "reason": "<one sentence covering the lowest score>"}}
"""

_METRIC_KEYS = ("faithfulness", "answer_relevance", "context_relevance", "correctness")


class LLMJudge:
    """Provider-agnostic judge. Works with Groq, Gemini, or local Ollama.

    Defaults to a single combined call scoring all four criteria at once.  The
    per-metric prompts each re-send the full context, so scoring one case cost
    four copies of it -- which exhausted a 100k tokens/day free tier partway
    through a three-video run.  Combined scoring cuts judge tokens roughly 4x
    and is what makes evaluation affordable on a free plan at all.

    ``combined=False`` restores the per-metric calls, which isolate each
    criterion and are the more careful choice when tokens are not the binding
    constraint -- useful for checking whether combining biases the scores.
    """

    def __init__(
        self, llm: LLM, judge_temperature: float = 0.0, combined: bool = True
    ) -> None:
        self.llm = llm
        self.temperature = judge_temperature
        self.combined = combined

    def _judge(self, prompt: str) -> tuple[float, str]:
        try:
            raw = self.llm.complete(JUDGE_SYSTEM, prompt, temperature=self.temperature)
        except Exception as exc:
            return 0.0, f"judge call failed: {exc}"
        return _parse_score(raw)

    def _score_combined(
        self, question: str, answer: Answer, context: str, reference: str | None
    ) -> GenerationScores:
        scores = GenerationScores()
        try:
            raw = self.llm.complete(
                JUDGE_SYSTEM,
                COMBINED_PROMPT.format(
                    question=question, context=context, answer=answer.text,
                    reference=reference or "(none given)",
                ),
                temperature=self.temperature,
            )
        except Exception as exc:
            scores.reasons["error"] = f"judge call failed: {exc}"
            return scores

        match = _JSON_BLOCK.search(raw)
        payload: dict = {}
        if match:
            try:
                payload = json.loads(match.group(0))
            except (ValueError, TypeError):
                payload = {}
        if not payload:
            scores.reasons["error"] = f"unparseable judge output: {raw.strip()[:120]}"
            return scores

        def _get(key: str) -> float | None:
            value = payload.get(key)
            if value is None:
                return None
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                return None

        scores.faithfulness = _get("faithfulness") or 0.0
        scores.answer_relevance = _get("answer_relevance") or 0.0
        scores.context_relevance = _get("context_relevance") or 0.0
        scores.correctness = _get("correctness") if reference else None
        reason = str(payload.get("reason", ""))
        for key in _METRIC_KEYS:
            scores.reasons[key] = reason

        # Same carve-out as the per-metric path: a refusal asserts nothing, so
        # it cannot be unfaithful.  Scoring it 0.0 would punish correct behaviour.
        if answer.refused:
            scores.faithfulness = 1.0
            scores.reasons["faithfulness"] = "refusal: no claims to verify"
        return scores

    def score(
        self,
        question: str,
        answer: Answer,
        retrieved: Sequence[ScoredChunk],
        reference: str | None = None,
    ) -> GenerationScores:
        context = "\n\n".join(h.chunk.text for h in retrieved) or "(no context)"
        if self.combined:
            return self._score_combined(question, answer, context, reference)

        scores = GenerationScores()

        # A refusal has nothing to be unfaithful about; scoring it as 0.0
        # faithfulness would punish the system for behaving correctly.
        if answer.refused:
            scores.faithfulness = 1.0
            scores.reasons["faithfulness"] = "refusal: no claims to verify"
        else:
            scores.faithfulness, scores.reasons["faithfulness"] = self._judge(
                FAITHFULNESS_PROMPT.format(context=context, answer=answer.text)
            )

        scores.answer_relevance, scores.reasons["answer_relevance"] = self._judge(
            ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer.text)
        )
        scores.context_relevance, scores.reasons["context_relevance"] = self._judge(
            CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context)
        )
        if reference:
            scores.correctness, scores.reasons["correctness"] = self._judge(
                CORRECTNESS_PROMPT.format(
                    question=question, reference=reference, answer=answer.text
                )
            )
        return scores


def mean_generation(scores: Sequence[GenerationScores]) -> GenerationScores:
    if not scores:
        return GenerationScores()
    n = len(scores)
    with_correctness = [s.correctness for s in scores if s.correctness is not None]
    return GenerationScores(
        faithfulness=sum(s.faithfulness for s in scores) / n,
        answer_relevance=sum(s.answer_relevance for s in scores) / n,
        context_relevance=sum(s.context_relevance for s in scores) / n,
        correctness=(sum(with_correctness) / len(with_correctness))
        if with_correctness else None,
    )


# ---------------------------------------------------------------------------
# Optional third-party judges (cross-checks for the judge-agreement experiment)
# ---------------------------------------------------------------------------


def ragas_scores(records: Sequence[dict]) -> dict[str, float] | None:
    """RAGAS faithfulness / answer_relevancy / context_precision.

    ``records`` need keys: question, answer, contexts (list[str]), ground_truth.
    Returns ``None`` when RAGAS is unavailable or misconfigured — the run
    continues on the built-in judge rather than dying.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError:
        return None
    try:
        dataset = Dataset.from_list(list(records))
        result = evaluate(
            dataset, metrics=[faithfulness, answer_relevancy, context_precision]
        )
        return {k: float(v) for k, v in dict(result).items()}
    except Exception as exc:  # RAGAS defaults to OpenAI; misconfig is expected
        return {"error": str(exc)}  # type: ignore[dict-item]


def deepeval_scores(records: Sequence[dict], threshold: float = 0.7) -> dict[str, float] | None:
    """DeepEval faithfulness + answer relevancy, averaged over cases."""
    try:
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
    except ImportError:
        return None
    try:
        faith = FaithfulnessMetric(threshold=threshold)
        relev = AnswerRelevancyMetric(threshold=threshold)
        totals = {"faithfulness": 0.0, "answer_relevancy": 0.0}
        for rec in records:
            case = LLMTestCase(
                input=rec["question"],
                actual_output=rec["answer"],
                retrieval_context=list(rec["contexts"]),
            )
            faith.measure(case)
            relev.measure(case)
            totals["faithfulness"] += float(faith.score or 0.0)
            totals["answer_relevancy"] += float(relev.score or 0.0)
        n = max(1, len(records))
        return {k: v / n for k, v in totals.items()}
    except Exception as exc:
        return {"error": str(exc)}  # type: ignore[dict-item]