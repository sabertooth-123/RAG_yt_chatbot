"""Character-offset ↔ wall-clock timeline.

The cleaned transcript is concatenated into a single string.  Each segment owns
a half-open character range ``[char_start, char_end)`` mapped to its time span.
``time_at`` then returns a timestamp for *any* character offset by linear
interpolation inside the owning segment — which is how a chunk boundary that
falls mid-cue still gets an accurate timestamp.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Sequence

from ytchat.models import TranscriptSegment

JOINER = " "


@dataclass(frozen=True, slots=True)
class SegmentSpan:
    seg_index: int
    char_start: int
    char_end: int
    t_start: float
    t_end: float


class Timeline:
    def __init__(self, text: str, spans: Sequence[SegmentSpan]) -> None:
        self.text = text
        self.spans = list(spans)
        self._starts = [s.char_start for s in self.spans]

    # -- construction ------------------------------------------------------
    @classmethod
    def build(cls, segments: Sequence[TranscriptSegment]) -> "Timeline":
        parts: list[str] = []
        spans: list[SegmentSpan] = []
        cursor = 0
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            if parts:
                parts.append(JOINER)
                cursor += len(JOINER)
            start = cursor
            parts.append(text)
            cursor += len(text)
            # Guard against zero/negative durations in malformed caption tracks.
            t_end = seg.end_s if seg.end_s > seg.start_s else seg.start_s + 0.001
            spans.append(
                SegmentSpan(seg.index, start, cursor, seg.start_s, t_end)
            )
        return cls("".join(parts), spans)

    # -- queries -----------------------------------------------------------
    def _span_index(self, char_pos: int) -> int:
        """Index of the span owning ``char_pos`` (or the nearest preceding one)."""
        i = bisect.bisect_right(self._starts, char_pos) - 1
        return max(0, min(i, len(self.spans) - 1))

    def time_at(self, char_pos: int) -> float:
        """Interpolated timestamp for a character offset.  Monotonic in ``char_pos``."""
        if not self.spans:
            return 0.0
        char_pos = max(0, min(char_pos, len(self.text)))
        i = self._span_index(char_pos)
        span = self.spans[i]
        if char_pos <= span.char_start:
            return span.t_start
        if char_pos >= span.char_end:
            # In the joiner gap: clamp to this span's end (never overshoot into
            # the next segment, so a chunk end never claims audio it doesn't cover).
            return span.t_end
        width = span.char_end - span.char_start
        frac = (char_pos - span.char_start) / width
        return span.t_start + frac * (span.t_end - span.t_start)

    def segments_covering(self, char_start: int, char_end: int) -> tuple[int, int]:
        """Inclusive range of source segment indices touched by a character span."""
        if not self.spans:
            return (0, 0)
        first = self.spans[self._span_index(char_start)]
        last = self.spans[self._span_index(max(char_start, char_end - 1))]
        return (first.seg_index, last.seg_index)

    @property
    def duration_s(self) -> float:
        return self.spans[-1].t_end if self.spans else 0.0