"""Token-budget behaviour: combined judge scoring, and daily-cap detection."""

import pytest

from ytchat.evaluation.generation_metrics import LLMJudge
from ytchat.generation.llm import ScriptedLLM, _is_transient
from ytchat.models import Answer

GOOD = (
    '{"faithfulness": 0.9, "answer_relevance": 0.8, "context_relevance": 0.6, '
    '"correctness": 0.7, "reason": "mostly supported"}'
)


def _answer(text="Attention focuses on tokens [1].", refused=False) -> Answer:
    return Answer(text=text, citations=(), refused=refused)


def test_combined_judge_uses_one_call_not_four() -> None:
    """Four per-metric calls each re-send the full context, which is what
    exhausted a 100k tokens/day free tier mid-run."""
    llm = ScriptedLLM([GOOD])
    LLMJudge(llm).score("What is attention?", _answer(), [], reference="ref")
    assert len(llm.calls) == 1


def test_combined_judge_parses_all_four_metrics() -> None:
    scores = LLMJudge(ScriptedLLM([GOOD])).score("q?", _answer(), [], reference="ref")
    assert scores.faithfulness == pytest.approx(0.9)
    assert scores.answer_relevance == pytest.approx(0.8)
    assert scores.context_relevance == pytest.approx(0.6)
    assert scores.correctness == pytest.approx(0.7)


def test_correctness_is_none_without_a_reference() -> None:
    scores = LLMJudge(ScriptedLLM([GOOD])).score("q?", _answer(), [])
    assert scores.correctness is None


def test_combined_judge_clamps_out_of_range_scores() -> None:
    raw = ('{"faithfulness": 7, "answer_relevance": -2, "context_relevance": 0.5, '
           '"correctness": null, "reason": "x"}')
    scores = LLMJudge(ScriptedLLM([raw])).score("q?", _answer(), [])
    assert scores.faithfulness == 1.0
    assert scores.answer_relevance == 0.0


def test_refusal_still_scores_faithful_in_combined_mode() -> None:
    raw = ('{"faithfulness": 0.0, "answer_relevance": 0.1, "context_relevance": 0.1, '
           '"correctness": null, "reason": "refused"}')
    scores = LLMJudge(ScriptedLLM([raw])).score("q?", _answer(refused=True), [])
    assert scores.faithfulness == 1.0, "a refusal asserts nothing, so it cannot be unfaithful"


def test_unparseable_output_does_not_crash_the_run() -> None:
    scores = LLMJudge(ScriptedLLM(["total nonsense"])).score("q?", _answer(), [])
    assert scores.faithfulness == 0.0
    assert "error" in scores.reasons


def test_per_metric_mode_still_available_for_ablation() -> None:
    llm = ScriptedLLM(['{"score": 0.8, "reason": "ok"}'] * 4)
    LLMJudge(llm, combined=False).score("q?", _answer(), [], reference="ref")
    assert len(llm.calls) == 4


# ---- daily-cap detection --------------------------------------------------
def test_daily_token_cap_is_not_retried() -> None:
    exc = Exception(
        "Error code: 429 - Rate limit reached for model on tokens per day (TPD): "
        "Limit 100000, Used 99288. Please try again in 7m42s."
    )
    assert not _is_transient(exc), "a daily cap cannot clear inside a retry window"


def test_zero_allowance_is_not_retried() -> None:
    exc = Exception("429 Quota exceeded for metric generate_content_free_tier, limit: 0")
    assert not _is_transient(exc)


def test_ordinary_rate_limit_is_still_retried() -> None:
    assert _is_transient(Exception("429 rate limit exceeded, retry in 2s"))
    assert _is_transient(Exception("503 Service overloaded"))


def test_real_errors_are_not_retried() -> None:
    assert not _is_transient(ValueError("bad model name"))
