import pytest

from ytchat.database.repository import Repository
from ytchat.errors import ConfigurationError
from ytchat.generation.llm import ScriptedLLM
from ytchat.pipeline import ChatSession


def test_history_accumulates_both_turns(index) -> None:
    session = ChatSession(index, llm=ScriptedLLM(["Attention focuses on tokens [1]."]))
    session.ask("What is attention?")
    assert [t.role for t in session.history] == ["user", "assistant"]
    assert session.history[1].citations


def test_follow_up_retrieves_with_the_rewritten_query(index) -> None:
    llm = ScriptedLLM([
        "Attention focuses on relevant tokens [1].",           # answer 1
        "Why is self-attention useful in transformers?",       # rewrite
        "It keeps gradients stable [1].",                      # answer 2
    ])
    session = ChatSession(index, llm=llm)
    session.ask("What is attention?")
    answer = session.ask("Why is that useful?")

    assert answer.rewritten_query == "Why is self-attention useful in transformers?"
    assert len(llm.calls) == 3


def test_first_question_never_triggers_a_rewrite(index) -> None:
    llm = ScriptedLLM(["Answer [1]."])
    ChatSession(index, llm=llm).ask("What is attention?")
    assert len(llm.calls) == 1


def test_change_retriever_switches_and_validates(index) -> None:
    session = ChatSession(index, llm=ScriptedLLM(), retriever_name="dense")
    session.change_retriever("sparse")
    assert session.retriever_name == "sparse"

    with pytest.raises(ConfigurationError):
        session.change_retriever("magic")
    assert session.retriever_name == "sparse", "a failed switch must not corrupt state"


def test_clear_history_resets_the_session(index) -> None:
    session = ChatSession(index, llm=ScriptedLLM(["Answer [1]."]))
    session.ask("What is attention?")
    session.clear_history()
    assert session.history == [] and session.last_answer is None


def test_conversation_is_persisted(index, settings) -> None:
    with Repository(settings.db_path) as repo:
        session = ChatSession(index, llm=ScriptedLLM(["Answer [1]."]), repo=repo)
        session.ask("What is attention?")
        messages = repo.get_messages(session.conversation_id)

    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["citations"], "citations must survive the round-trip"


def test_answer_carries_retrieval_debug_info(index) -> None:
    session = ChatSession(index, llm=ScriptedLLM(["Answer [1]."]), retriever_name="hybrid")
    answer = session.ask("What is attention?")
    assert answer.retriever == "hybrid"
    assert answer.retrieved and answer.latency_ms >= 0