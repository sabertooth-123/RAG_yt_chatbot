"""Typer entrypoint.

``yt-chat <url>`` is the documented usage, but Click cannot have both a
group-level argument and subcommands without ambiguity, so ``main()`` inserts
the implicit ``chat`` subcommand when argv[0] looks like a URL or video ID.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from ytchat.config import Settings, load_settings
from ytchat.database.repository import Repository
from ytchat.errors import YtChatError
from ytchat.pipeline import ChatSession, ensure_indexed

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Ask questions about any YouTube video, with timestamped citations.")
console = Console()

SUBCOMMANDS = {"chat", "ask", "cache", "eval", "--help", "-h", "--version"}

# Shared options
_Retriever = typer.Option(None, "--retriever", "-r", help="dense | sparse | hybrid")
_TopK = typer.Option(None, "--top-k", "-k", help="Chunks to retrieve")
_LLM = typer.Option(None, "--llm", help="gemini | groq | openrouter | ollama")
_Model = typer.Option(None, "--model", "-m", help="Model name override")
_Embedder = typer.Option(None, "--embedder", help="sentence-transformers | hashing")
_Force = typer.Option(False, "--force", "-f", help="Reprocess, ignoring the cache")
_Quiet = typer.Option(False, "--quiet", "-q", help="Suppress progress output")


def _settings(**kwargs) -> Settings:
    return load_settings(**kwargs)


def _progress(quiet: bool):
    return (lambda _m: None) if quiet else (lambda m: console.print(f"[dim]· {m}[/dim]"))


@app.command()
def chat(
    url: str = typer.Argument(..., help="YouTube URL, video ID, or local media file"),
    retriever: Optional[str] = _Retriever,
    top_k: Optional[int] = _TopK,
    llm: Optional[str] = _LLM,
    model: Optional[str] = _Model,
    embedder: Optional[str] = _Embedder,
    force: bool = _Force,
    quiet: bool = _Quiet,
) -> None:
    """Start an interactive session about a video."""
    from ytchat.cli.repl import Repl

    settings = _settings(retriever=retriever, top_k=top_k, llm_provider=llm,
                         llm_model=model, embedder=embedder)
    with Repository(settings.db_path) as repo:
        index = ensure_indexed(url, settings, repo, progress=_progress(quiet), force=force)
        session = ChatSession(index, repo=repo)
        Repl(session, console).run()


@app.command()
def ask(
    url: str = typer.Argument(...),
    question: str = typer.Option(..., "--question", "-Q", help="The question to ask"),
    retriever: Optional[str] = _Retriever,
    top_k: Optional[int] = _TopK,
    llm: Optional[str] = _LLM,
    model: Optional[str] = _Model,
    embedder: Optional[str] = _Embedder,
    quiet: bool = typer.Option(True, "--quiet/--verbose"),
) -> None:
    """One-shot question — useful for scripting and piping."""
    from ytchat.generation.citations import render_sources

    settings = _settings(retriever=retriever, top_k=top_k, llm_provider=llm,
                         llm_model=model, embedder=embedder)
    with Repository(settings.db_path) as repo:
        index = ensure_indexed(url, settings, repo, progress=_progress(quiet))
        answer = ChatSession(index, repo=repo).ask(question)

    console.print(answer.text)
    if answer.citations:
        console.print()
        console.print(render_sources(answer.citations))
    raise typer.Exit(code=1 if answer.refused else 0)


@app.command()
def cache(
    action: str = typer.Argument("stats", help="stats | clear"),
    video: Optional[str] = typer.Option(None, "--video", "-v", help="Video URL or ID"),
) -> None:
    """Inspect or clear the processed-video cache."""
    from ytchat.ingestion.url import parse_video_id

    settings = _settings()
    with Repository(settings.db_path) as repo:
        if action == "stats":
            console.print(f"[bold]Cache[/bold] {settings.db_path}")
            for table, count in repo.stats().items():
                console.print(f"  {table:<16} {count}")
        elif action == "clear":
            if not video:
                console.print("[red]Pass --video <url|id> (refusing to wipe everything).[/red]")
                raise typer.Exit(code=2)
            repo.clear_video(parse_video_id(video))
            console.print("[green]Cleared.[/green]")
        else:
            console.print(f"[red]Unknown action {action!r}. Use stats or clear.[/red]")
            raise typer.Exit(code=2)


eval_app = typer.Typer(help="Measure retrieval and generation quality.")
app.add_typer(eval_app, name="eval")


@eval_app.command("run")
def eval_run(
    benchmark: Path = typer.Argument(..., help="Path to a benchmark YAML"),
    retrievers: str = typer.Option("dense,sparse,hybrid", "--retrievers"),
    top_k: Optional[int] = _TopK,
    llm: Optional[str] = _LLM,
    judge_llm: Optional[str] = typer.Option(None, "--judge-llm",
                                            help="Provider for the judge (default: same as --llm)"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Retrieval metrics only — no LLM calls"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write markdown report here"),
    csv_out: Optional[Path] = typer.Option(None, "--csv", help="Write per-case CSV here"),
    calibrate: bool = typer.Option(True, "--calibrate/--no-calibrate"),
) -> None:
    """Compare retrieval strategies on a benchmark."""
    from ytchat.evaluation.dataset import Benchmark
    from ytchat.evaluation.report import Report
    from ytchat.evaluation.runner import EvalRunner
    from ytchat.generation.llm import build_llm

    bench = Benchmark.from_yaml(benchmark)
    settings = _settings(top_k=top_k, llm_provider=llm)
    names = [n.strip() for n in retrievers.split(",") if n.strip()]

    judge = None
    if judge_llm:
        judge = build_llm(_settings(llm_provider=judge_llm))

    with Repository(settings.db_path) as repo:
        runner = EvalRunner(bench, settings, repo, judge=judge,
                            progress=lambda m: console.print(f"[dim]· {m}[/dim]"))
        results = runner.compare(names, full=not no_judge, judge=not no_judge)
        thresholds = runner.calibrate_threshold() if calibrate else None

    report = Report(
        benchmark=bench, results=results, thresholds=thresholds,
        settings_summary={
            "retriever_top_k": settings.top_k,
            "max_chars": settings.max_chars,
            "embedding_model": settings.embedding_model,
            "llm": settings.resolved_model if not no_judge else "n/a",
        },
    )
    markdown = report.to_markdown()
    console.print(markdown)
    if out:
        Path(out).write_text(markdown, encoding="utf-8")
        console.print(f"\n[green]Report → {out}[/green]")
    if csv_out:
        report.to_csv(csv_out)
        console.print(f"[green]Per-case CSV → {csv_out}[/green]")


@eval_app.command("calibrate")
def eval_calibrate(
    benchmark: Path = typer.Argument(...),
    retriever: str = typer.Option("hybrid", "--retriever", "-r"),
) -> None:
    """Sweep the Layer A refusal threshold. Costs zero LLM calls."""
    from ytchat.evaluation.dataset import Benchmark
    from ytchat.evaluation.report import threshold_table
    from ytchat.evaluation.runner import EvalRunner, best_threshold

    bench = Benchmark.from_yaml(benchmark)
    settings = _settings()
    with Repository(settings.db_path) as repo:
        runner = EvalRunner(bench, settings, repo,
                            progress=lambda m: console.print(f"[dim]· {m}[/dim]"))
        points = runner.calibrate_threshold(retriever)

    console.print(threshold_table(points))
    best = best_threshold(points)
    if best:
        console.print(
            f"\n[green]Recommended YTCHAT_MIN_SCORE={best.threshold:.2f}[/green] "
            f"(correct refusal {best.correct_refusal_rate:.0%}, "
            f"false refusal {best.false_refusal_rate:.0%})"
        )


@eval_app.command("draft")
def eval_draft(
    url: str = typer.Argument(...),
    question: str = typer.Option(..., "--question", "-Q"),
    k: int = typer.Option(3, "--k"),
) -> None:
    """Propose gold time spans for a question, for you to verify by hand.

    Never trust these blindly — retrieval proposing its own ground truth is
    circular. Watch the moments, then keep or correct the spans.
    """
    from ytchat.models import format_timestamp

    settings = _settings()
    with Repository(settings.db_path) as repo:
        index = ensure_indexed(url, settings, repo, progress=_progress(True))
        hits = index.retriever("hybrid").search(question, k)

    console.print(f"  - id: TODO\n    question: {question!r}")
    console.print("    expected_answer: TODO")
    console.print("    relevant_spans:")
    for h in hits:
        console.print(
            f"      - [{format_timestamp(h.chunk.start_s)}, "
            f"{format_timestamp(h.chunk.end_s)}]   "
            f"# conf={h.components.get('confidence', 0):.2f} — VERIFY: {h.chunk.text[:70]}…"
        )


def main() -> None:
    argv = sys.argv[1:]
    if argv and not argv[0].startswith("-") and argv[0] not in SUBCOMMANDS:
        sys.argv.insert(1, "chat")     # yt-chat <url>  →  yt-chat chat <url>
    try:
        app()
    except YtChatError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(exc.exit_code)


if __name__ == "__main__":
    main()