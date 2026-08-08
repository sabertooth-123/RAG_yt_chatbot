"""Interactive REPL."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ytchat.errors import YtChatError
from ytchat.models import Answer, format_timestamp
from ytchat.pipeline import ChatSession, VideoIndex
from ytchat.retrieval.base import confidence_of

HELP = """\
[bold]Commands[/bold]
  [cyan]/ask <question>[/cyan]        Ask about the video (or just type your question)
  [cyan]/sources[/cyan]               Show sources for the last answer
  [cyan]/history[/cyan]               Show the conversation so far
  [cyan]/change-retriever <n>[/cyan]  Switch retriever: dense | sparse | hybrid
  [cyan]/compare <question>[/cyan]    Run all three retrievers side by side
  [cyan]/video[/cyan]                 Show video and index details
  [cyan]/debug[/cyan]                 Explain the last answer (scores, refusal layer)
  [cyan]/clear[/cyan]                 Forget the conversation history
  [cyan]/help[/cyan]                  Show this
  [cyan]/exit[/cyan]                  Quit
"""


class Repl:
    def __init__(self, session: ChatSession, console: Console | None = None) -> None:
        self.session = session
        self.console = console or Console()

    # -- rendering ----------------------------------------------------------
    def banner(self) -> None:
        index: VideoIndex = self.session.index
        meta = index.metadata
        self.console.print(
            Panel(
                Text.from_markup(
                    f"[bold]{meta.title}[/bold]\n"
                    f"[dim]{meta.channel or 'unknown channel'} · "
                    f"{format_timestamp(meta.duration_s or 0)} · "
                    f"{index.stats.n_chunks} chunks · "
                    f"{meta.transcript_kind.value} captions[/dim]\n"
                    f"[dim]retriever: {self.session.retriever_name} · "
                    f"llm: {self.session.llm.model_id}[/dim]"
                ),
                title="yt-chat", border_style="cyan",
            )
        )
        self.console.print("[dim]Type a question, or /help for commands.[/dim]\n")

    def show_answer(self, answer: Answer) -> None:
        style = "yellow" if answer.refused else "green"
        body = Text(answer.text)
        self.console.print(Panel(body, border_style=style, title="Answer"))

        if answer.citations:
            self.console.print()
            for c in answer.citations:
                self.console.print(
                    f"  [bold cyan][{c.marker}][/bold cyan] "
                    f"[link={c.url}][bold]{c.timestamp}[/bold][/link]  [dim]{c.url}[/dim]"
                )
        if answer.rewritten_query:
            self.console.print(f"\n[dim]↳ retrieved using: “{answer.rewritten_query}”[/dim]")
        self.console.print()

    def show_sources(self) -> None:
        answer = self.session.last_answer
        if answer is None:
            self.console.print("[yellow]No question asked yet.[/yellow]")
            return
        if not answer.citations:
            self.console.print("[yellow]The last answer had no sources.[/yellow]")
            return
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("#", justify="right")
        table.add_column("Time")
        table.add_column("Excerpt", overflow="fold")
        for c in answer.citations:
            table.add_row(
                str(c.marker), f"[link={c.url}]{c.timestamp}[/link]", f"“{c.quote}”"
            )
        self.console.print(table)
        self.console.print()

    def show_history(self) -> None:
        if not self.session.history:
            self.console.print("[yellow]No history yet.[/yellow]")
            return
        for turn in self.session.history:
            if turn.role == "user":
                self.console.print(f"[bold blue]You:[/bold blue] {turn.content}")
            else:
                marks = " ".join(f"[{c.timestamp}]" for c in turn.citations)
                self.console.print(f"[bold green]yt-chat:[/bold green] {turn.content}")
                if marks:
                    self.console.print(f"[dim]         {marks}[/dim]")
        self.console.print()

    def show_debug(self) -> None:
        answer = self.session.last_answer
        if answer is None:
            self.console.print("[yellow]No question asked yet.[/yellow]")
            return
        trace = self.session.refusal_trace
        self.console.print(
            f"[bold]retriever:[/bold] {answer.retriever}   "
            f"[bold]latency:[/bold] {answer.latency_ms:.0f} ms   "
            f"[bold]refused:[/bold] {answer.refused}"
        )
        if trace.layer:
            self.console.print(f"[bold]gate:[/bold] {trace.layer} — {trace.detail}")
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        for col in ("#", "time", "score", "conf", "text"):
            table.add_column(col, overflow="fold" if col == "text" else "ellipsis")
        for i, hit in enumerate(answer.retrieved, start=1):
            table.add_row(
                str(i), format_timestamp(hit.chunk.start_s), f"{hit.score:.4f}",
                f"{confidence_of(hit):.2f}", hit.chunk.text[:80] + "…",
            )
        self.console.print(table)
        self.console.print()

    def show_compare(self, question: str) -> None:
        from ytchat.pipeline import compare_retrievers

        results = compare_retrievers(self.session.index, question, k=self.session.settings.top_k)
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("retriever")
        table.add_column("rank")
        table.add_column("time")
        table.add_column("conf")
        table.add_column("text", overflow="fold")
        for name, hits in results.items():
            for hit in hits:
                table.add_row(
                    name if hit.rank == 0 else "", str(hit.rank + 1),
                    format_timestamp(hit.chunk.start_s), f"{confidence_of(hit):.2f}",
                    hit.chunk.text[:70] + "…",
                )
        self.console.print(table)
        self.console.print()

    def show_video(self) -> None:
        index = self.session.index
        meta = index.metadata
        self.console.print(
            f"[bold]{meta.title}[/bold]\n"
            f"  url        {meta.url}\n"
            f"  channel    {meta.channel or '—'}\n"
            f"  duration   {format_timestamp(meta.duration_s or 0)}\n"
            f"  captions   {meta.transcript_kind.value} ({meta.language or '—'})\n"
            f"  segments   {index.stats.n_segments}\n"
            f"  chunks     {index.stats.n_chunks}\n"
            f"  embedder   {index.embedder.model_id if index.embedder else '—'}\n"
            f"  index      {index.store.backend if index.store else '—'}\n"
        )

    # -- loop ---------------------------------------------------------------
    def handle(self, line: str) -> bool:
        """Returns False to exit the loop."""
        text = line.strip()
        if not text:
            return True

        if not text.startswith("/"):
            self.show_answer(self.session.ask(text))
            return True

        command, _, rest = text.partition(" ")
        command, rest = command.lower(), rest.strip()

        if command in ("/exit", "/quit", "/q"):
            return False
        if command == "/help":
            self.console.print(HELP)
        elif command == "/ask":
            if not rest:
                self.console.print("[yellow]Usage: /ask <question>[/yellow]")
            else:
                self.show_answer(self.session.ask(rest))
        elif command == "/sources":
            self.show_sources()
        elif command == "/history":
            self.show_history()
        elif command in ("/change-retriever", "/retriever"):
            try:
                self.session.change_retriever(rest)
                self.console.print(f"[green]Retriever → {rest}[/green]\n")
            except YtChatError as exc:
                self.console.print(f"[red]{exc}[/red]\n")
        elif command == "/compare":
            if not rest:
                self.console.print("[yellow]Usage: /compare <question>[/yellow]")
            else:
                self.show_compare(rest)
        elif command == "/video":
            self.show_video()
        elif command == "/debug":
            self.show_debug()
        elif command == "/clear":
            self.session.clear_history()
            self.console.print("[green]History cleared.[/green]\n")
        else:
            self.console.print(f"[yellow]Unknown command {command}. Try /help[/yellow]\n")
        return True

    def run(self) -> None:
        self.banner()
        while True:
            try:
                line = self.console.input("[bold blue]›[/bold blue] ")
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]Bye.[/dim]")
                return
            try:
                if not self.handle(line):
                    self.console.print("[dim]Bye.[/dim]")
                    return
            except YtChatError as exc:
                self.console.print(f"[red]{exc}[/red]\n")
            except Exception as exc:  # keep the session alive on unexpected errors
                self.console.print(f"[red]Unexpected error: {exc}[/red]\n")