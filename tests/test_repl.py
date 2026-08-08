from rich.console import Console

from ytchat.cli.repl import Repl
from ytchat.generation.llm import ScriptedLLM
from ytchat.pipeline import ChatSession


def _repl(index, responses=None):
    console = Console(record=True, width=100, force_terminal=False)
    session = ChatSession(index, llm=ScriptedLLM(responses or ["Answer [1]."]))
    return Repl(session, console), console


def test_bare_text_is_treated_as_a_question(index) -> None:
    repl, console = _repl(index)
    assert repl.handle("What is attention?") is True
    assert len(repl.session.history) == 2
    assert "Answer" in console.export_text()


def test_exit_stops_the_loop(index) -> None:
    repl, _ = _repl(index)
    assert repl.handle("/exit") is False


def test_sources_before_asking_is_handled(index) -> None:
    repl, console = _repl(index)
    repl.handle("/sources")
    assert "No question asked yet" in console.export_text()


def test_change_retriever_command(index) -> None:
    repl, console = _repl(index)
    repl.handle("/change-retriever sparse")
    assert repl.session.retriever_name == "sparse"
    repl.handle("/change-retriever nonsense")
    assert "Unknown retriever" in console.export_text()


def test_unknown_command_does_not_crash(index) -> None:
    repl, console = _repl(index)
    assert repl.handle("/banana") is True
    assert "Unknown command" in console.export_text()


def test_history_and_debug_render(index) -> None:
    repl, console = _repl(index)
    repl.handle("What is attention?")
    repl.handle("/history")
    repl.handle("/debug")
    out = console.export_text()
    assert "You:" in out and "retriever" in out