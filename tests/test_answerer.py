import pytest

from ytchat.generation.answerer import Answerer
from ytchat.generation.llm import ScriptedLLM
from ytchat.generation.prompts import REFUSAL_MESSAGE, REFUSAL_SENTINEL


def _hits(index, query="attention", k=3):
    return index.retriever("hybrid").search(query, k)


def test_grounded_answer_is_returned_with_citations(index, settings) -> None:
    llm = ScriptedLLM(["Attention lets the model focus on relevant tokens [1]."])
    answer = Answerer(llm, settings).answer("What is attention?", _hits(index), index.video_id)

    assert not answer.refused
    assert "[1]" in answer.text
    assert len(answer.citations) == 1
    assert answer.citations[0].url.startswith("https://www.youtube.com/watch?v=")


def test_layer_a_refuses_before_calling_the_llm(index, settings) -> None:
    """A low-confidence retrieval must not cost an API call — this matters on a
    free tier, and it is the cheapest hallucination guard available."""
    strict = settings.model_copy(update={"min_score": 0.99})
    llm = ScriptedLLM(["This should never be produced."])
    answerer = Answerer(llm, strict)

    answer = answerer.answer("What is the capital of Peru?", _hits(index), index.video_id)
    assert answer.refused
    assert answer.text == REFUSAL_MESSAGE
    assert llm.calls == [], "Layer A must short-circuit before generation"
    assert answerer.last_trace.layer == "A:retrieval"


def test_layer_a_refuses_on_empty_retrieval(index, settings) -> None:
    llm = ScriptedLLM(["never"])
    answer = Answerer(llm, settings).answer("anything", [], index.video_id)
    assert answer.refused and llm.calls == []


def test_layer_b_honours_the_sentinel(index, settings) -> None:
    llm = ScriptedLLM([REFUSAL_SENTINEL])
    answerer = Answerer(llm, settings)
    answer = answerer.answer("What year was this invented?", _hits(index), index.video_id)

    assert answer.refused
    assert answer.text == REFUSAL_MESSAGE
    assert answerer.last_trace.layer == "B:sentinel"


def test_layer_b_catches_an_empty_response(index, settings) -> None:
    answer = Answerer(ScriptedLLM(["   "]), settings).answer("q", _hits(index), index.video_id)
    assert answer.refused


def test_layer_c_refuses_an_uncited_answer(index, settings) -> None:
    """Fluent, plausible, no citation — the classic hallucination shape."""
    llm = ScriptedLLM(["The algorithm was invented in 1997 by Hochreiter and Schmidhuber."])
    answerer = Answerer(llm, settings)
    answer = answerer.answer("What year was it invented?", _hits(index), index.video_id)

    assert answer.refused
    assert answerer.last_trace.layer == "C:citations"


def test_layer_c_refuses_when_every_citation_is_invented(index, settings) -> None:
    llm = ScriptedLLM(["It was invented in 1997 [9]."])
    answerer = Answerer(llm, settings)
    answer = answerer.answer("What year?", _hits(index, k=3), index.video_id)

    assert answer.refused
    assert "invalid" in answerer.last_trace.detail


def test_partially_valid_citations_survive_with_a_warning(index, settings) -> None:
    llm = ScriptedLLM(["Supported claim [1]. Unsupported claim [9]."])
    answerer = Answerer(llm, settings)
    answer = answerer.answer("q", _hits(index, k=3), index.video_id)

    assert not answer.refused
    assert "[9]" not in answer.text
    assert [c.marker for c in answer.citations] == [1]
    assert answerer.last_trace.layer == "C:partial"


def test_disabling_layer_c_lets_uncited_answers_through(index, settings) -> None:
    """The layers are independently toggleable so each one's contribution can be
    measured in the ablation."""
    lenient = settings.model_copy(update={"require_valid_citations": False})
    llm = ScriptedLLM(["An answer with no citations at all."])
    answer = Answerer(llm, lenient).answer("q", _hits(index), index.video_id)
    assert not answer.refused and answer.citations == ()


def test_prompt_contains_the_numbered_excerpts(index, settings) -> None:
    llm = ScriptedLLM(["Answer [1]."])
    Answerer(llm, settings).answer("What is attention?", _hits(index, k=2), index.video_id)
    prompt = llm.calls[0]["prompt"]
    assert "[1] (" in prompt and "[2] (" in prompt
    assert "What is attention?" in prompt
    assert REFUSAL_SENTINEL in llm.calls[0]["system"]