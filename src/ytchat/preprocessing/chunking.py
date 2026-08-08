"""Timestamp-preserving chunking.

Pipeline: cleaned segments → Timeline → *units* → chunks.

Unit selection is adaptive.  Human-written captions are punctuated, so units are
sentences.  ASR captions frequently contain no terminal punctuation at all; for
those, splitting on '.' yields one enormous unit, so we fall back to fixed
word-windows.  The switch is made by measuring punctuation density.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from ytchat.models import Chunk, ChunkerConfig, TranscriptSegment
from ytchat.preprocessing.timeline import Timeline

_SENTENCE_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*(?:\s+|$)")
_WORD = re.compile(r"\S+")
_TERMINAL_PUNCT = re.compile(r"[.!?]")


@dataclass(frozen=True, slots=True)
class Unit:
    """An atomic, indivisible piece of text with character offsets."""

    char_start: int
    char_end: int

    @property
    def n_chars(self) -> int:
        return self.char_end - self.char_start


def punctuation_density(text: str) -> float:
    if not text:
        return 0.0
    return len(_TERMINAL_PUNCT.findall(text)) / len(text)


def sentence_units(text: str) -> list[Unit]:
    units: list[Unit] = []
    start = 0
    for m in _SENTENCE_BOUNDARY.finditer(text):
        end = m.end() - (len(m.group()) - len(m.group().rstrip()))
        if end > start:
            units.append(Unit(start, end))
        start = m.end()
    if start < len(text):
        units.append(Unit(start, len(text)))
    return units


def word_window_units(text: str, words_per_window: int) -> list[Unit]:
    words = list(_WORD.finditer(text))
    units: list[Unit] = []
    for i in range(0, len(words), words_per_window):
        window = words[i: i + words_per_window]
        if window:
            units.append(Unit(window[0].start(), window[-1].end()))
    return units


def _split_oversized(unit: Unit, text: str, max_chars: int) -> list[Unit]:
    """A single unit longer than ``max_chars`` (rambling ASR run-on) is split on
    word boundaries so no chunk can exceed the budget."""
    if unit.n_chars <= max_chars:
        return [unit]
    out: list[Unit] = []
    words = list(_WORD.finditer(text, unit.char_start, unit.char_end))
    cur_start = unit.char_start
    cur_end = unit.char_start
    for w in words:
        if w.end() - cur_start > max_chars and cur_end > cur_start:
            out.append(Unit(cur_start, cur_end))
            cur_start = w.start()
        cur_end = w.end()
    if cur_end > cur_start:
        out.append(Unit(cur_start, cur_end))
    return out or [unit]


def build_units(text: str, config: ChunkerConfig) -> list[Unit]:
    if punctuation_density(text) >= config.punctuation_density_threshold:
        units = sentence_units(text)
    else:
        units = word_window_units(text, config.words_per_window)
    flat: list[Unit] = []
    for u in units:
        flat.extend(_split_oversized(u, text, config.max_chars))
    return [u for u in flat if u.n_chars > 0]


def _overlap_tail(units: Sequence[Unit], overlap_chars: int) -> list[Unit]:
    """Trailing units whose cumulative length fits inside the overlap budget.

    Overlap is expressed in characters but applied at unit granularity, so a
    sentence is never cut in half by the overlap window.
    """
    if overlap_chars <= 0:
        return []
    tail: list[Unit] = []
    total = 0
    for u in reversed(units):
        if total + u.n_chars > overlap_chars:
            break
        tail.insert(0, u)
        total += u.n_chars
    # Always carry at least one unit if overlap was requested and there's room.
    if not tail and units and units[-1].n_chars <= overlap_chars * 2:
        tail = [units[-1]]
    return tail


class TranscriptChunker:
    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    def chunk(self, segments: Sequence[TranscriptSegment]) -> list[Chunk]:
        cfg = self.config
        timeline = Timeline.build(segments)
        text = timeline.text
        if not text:
            return []

        units = build_units(text, cfg)
        if not units:
            return []

        groups: list[list[Unit]] = []
        current: list[Unit] = []
        current_chars = 0

        for unit in units:
            projected = current_chars + unit.n_chars + (1 if current else 0)
            if current and projected > cfg.max_chars:
                groups.append(current)
                carry = _overlap_tail(current, cfg.overlap_chars)
                current = list(carry)
                current_chars = sum(u.n_chars for u in carry) + max(0, len(carry) - 1)
            current.append(unit)
            current_chars += unit.n_chars + (1 if len(current) > 1 else 0)

        if current:
            groups.append(current)

        # Merge a runt tail chunk into its predecessor rather than emitting a
        # near-empty chunk that would pollute BM25 statistics.
        if len(groups) > 1:
            tail_chars = sum(u.n_chars for u in groups[-1])
            if tail_chars < cfg.min_chars:
                merged = groups[-2] + [u for u in groups[-1] if u not in groups[-2]]
                groups = groups[:-2] + [merged]

        return [self._materialise(i, g, timeline, text) for i, g in enumerate(groups)]

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _materialise(index: int, group: list[Unit], timeline: Timeline, text: str) -> Chunk:
        char_start = group[0].char_start
        char_end = group[-1].char_end
        body = " ".join(text[u.char_start: u.char_end].strip() for u in group).strip()
        seg_start, seg_end = timeline.segments_covering(char_start, char_end)
        start_s = timeline.time_at(char_start)
        end_s = max(start_s, timeline.time_at(char_end))
        return Chunk(
            index=index,
            text=body,
            start_s=round(start_s, 3),
            end_s=round(end_s, 3),
            seg_start=seg_start,
            seg_end=seg_end,
        )