from ytchat.generation.llm import ScriptedLLM
from ytchat.generation.rewriter import QueryRewriter, needs_rewrite
from ytchat.models import Turn

HISTORY = (
    Turn(role="user", content="What is attention?"),
    Turn(role="assistant", content="Attention lets the model focus on relevant tokens."),
)


def test_no_history_means_no_rewrite() -> None:
    assert not needs_rewrite("Why is that useful?", ())


def test_referential_and_short_questions_need_rewriting() -> None:
    assert needs_rewrite("Why is that useful?", HISTORY)
    assert needs_rewrite("Why?", HISTORY)
    assert needs_rewrite("And how does it scale?", HISTORY)


def test_standalone_questions_skip_the_llm_call() -> None:
    llm = ScriptedLLM(["should not be used"])
    query, rewritten = QueryRewriter(llm).rewrite(
        "What are positional encodings used for?", HISTORY
    )
    assert query == "What are positional encodings used for?"
    assert not rewritten
    assert llm.calls == [], "the heuristic gate must avoid a wasted API call"


def test_follow_up_is_rewritten_to_standalone() -> None:
    llm = ScriptedLLM(["Why is self-attention useful in transformers?"])
    query, rewritten = QueryRewriter(llm).rewrite("Why is that useful?", HISTORY)
    assert rewritten
    assert "self-attention" in query
    assert "What is attention?" in llm.calls[0]["prompt"], "history must reach the prompt"


def test_rewriting_can_be_disabled() -> None:
    llm = ScriptedLLM(["rewritten"])
    query, rewritten = QueryRewriter(llm, enabled=False).rewrite("Why is that useful?", HISTORY)
    assert query == "Why is that useful?" and not rewritten and llm.calls == []


def test_rewriter_failure_falls_back_to_the_original() -> None:
    class Broken:
        model_id = "broken"

        def complete(self, system, prompt, temperature=0.0):
            raise RuntimeError("network down")

    query, rewritten = QueryRewriter(Broken()).rewrite("Why is that useful?", HISTORY)
    assert query == "Why is that useful?" and not rewritten


def test_runaway_rewrite_is_rejected() -> None:
    llm = ScriptedLLM(["x" * 500])
    query, rewritten = QueryRewriter(llm).rewrite("Why is that useful?", HISTORY)
    assert query == "Why is that useful?" and not rewritten