"""Citation extraction, validation, and rendering.

Layer C of the anti-hallucination design lives here: a model can emit a
confident-looking answer citing excerpt [7] when only five were supplied.  Those
markers are stripped, and an answer left with no valid citations is downgraded
to a refusal by the Answerer.
"""

from __future__ import annotations

import re
from typing import Sequence

from ytchat.models import Citation, ScoredChunk, format_timestamp, timestamp_url

_MARKER = re.compile(r"\[(\d{1,2})\]")
_ORPHAN_SPACE = re.compile(r"\s+([.,;:!?])")
_MULTISPACE = re.compile(r"[ \t]{2,}")

QUOTE_CHARS = 160


def extract_markers(text: str) -> list[int]:
    """Marker numbers in order of first appearance, deduplicated."""
    seen: list[int] = []
    for match in _MARKER.finditer(text):
        n = int(match.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def _quote(text: str, limit: int = QUOTE_CHARS) -> str:
    snippet = " ".join(text.split())
    return snippet if len(snippet) <= limit else snippet[: limit - 1].rstrip() + "…"


def build_citations(
    answer_text: str, hits: Sequence[ScoredChunk], video_id: str
) -> tuple[str, tuple[Citation, ...], list[int]]:
    """Validate markers against the excerpts that were actually supplied.

    Returns ``(cleaned_text, citations, invalid_markers)``.  Markers outside
    ``1..len(hits)`` are removed from the text and reported.
    """
    valid_range = range(1, len(hits) + 1)
    invalid = [n for n in extract_markers(answer_text) if n not in valid_range]

    cleaned = answer_text
    if invalid:
        cleaned = _MARKER.sub(
            lambda m: "" if int(m.group(1)) not in valid_range else m.group(0), cleaned
        )
        cleaned = _ORPHAN_SPACE.sub(r"\1", _MULTISPACE.sub(" ", cleaned)).strip()

    citations = tuple(
        Citation(
            marker=n,
            chunk_id=hits[n - 1].chunk.chunk_id,
            start_s=hits[n - 1].chunk.start_s,
            end_s=hits[n - 1].chunk.end_s,
            url=timestamp_url(video_id, hits[n - 1].chunk.start_s),
            timestamp=format_timestamp(hits[n - 1].chunk.start_s),
            quote=_quote(hits[n - 1].chunk.text),
        )
        for n in extract_markers(cleaned)
        if n in valid_range
    )
    return cleaned, citations, invalid


def build_context(hits: Sequence[ScoredChunk], max_chars: int) -> tuple[str, list[ScoredChunk]]:
    """Numbered excerpt blocks within a character budget.

    Excerpts are numbered by rank, so the ``[n]`` the model emits maps directly
    onto ``hits[n-1]`` — no fuzzy matching between answer text and source, which
    is where most citation systems quietly go wrong.
    """
    from ytchat.generation.prompts import CONTEXT_BLOCK

    blocks: list[str] = []
    used: list[ScoredChunk] = []
    total = 0
    for hit in hits:
        block = CONTEXT_BLOCK.format(
            n=len(used) + 1,
            start=format_timestamp(hit.chunk.start_s),
            end=format_timestamp(hit.chunk.end_s),
            text=hit.chunk.text,
        )
        if used and total + len(block) > max_chars:
            break
        blocks.append(block)
        used.append(hit)
        total += len(block)
    return "\n\n".join(blocks), used


def terminal_link(url: str, label: str) -> str:
    """OSC-8 hyperlink: renders as clickable label in Windows Terminal, iTerm2,
    GNOME Terminal, VS Code.  Degrades to plain text elsewhere."""
    return f"\033]8;;{url}\033\\{label}\033]8;;\033\\"


def render_sources(citations: Sequence[Citation], plain: bool = False) -> str:
    if not citations:
        return "No sources — the answer was not grounded in the video."
    lines = ["Sources:"]
    for c in citations:
        label = f"[{c.timestamp}]"
        link = f"{label} {c.url}" if plain else f"{terminal_link(c.url, label)} {c.url}"
        lines.append(f"  {c.marker}. {link}\n     “{c.quote}”")
    return "\n".join(lines)