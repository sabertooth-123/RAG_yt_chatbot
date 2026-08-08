"""Domain models.

Design invariant
----------------
Every text-bearing object in the pipeline carries an absolute time span
(``start_s``/``end_s``) in seconds from the start of the video, plus enough
provenance to trace back to the original caption segments.  Nothing downstream
is allowed to drop that information.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Iterable, Sequence


class TranscriptKind(str, Enum):
    MANUAL = "manual"          # human-written captions
    GENERATED = "generated"    # ASR / auto-generated
    TRANSLATED = "translated"  # machine-translated from another track
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One raw caption cue, exactly as YouTube serves it."""

    index: int
    text: str
    start_s: float
    duration_s: float

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    def to_row(self) -> tuple[int, float, float, str]:
        return (self.index, self.start_s, self.end_s, self.text)


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    video_id: str
    title: str
    channel: str | None = None
    duration_s: float | None = None
    language: str | None = None
    transcript_kind: TranscriptKind = TranscriptKind.UNKNOWN
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


@dataclass(frozen=True, slots=True)
class RawTranscript:
    video_id: str
    language: str
    kind: TranscriptKind
    segments: tuple[TranscriptSegment, ...]

    @property
    def duration_s(self) -> float:
        return self.segments[-1].end_s if self.segments else 0.0

    @property
    def n_chars(self) -> int:
        return sum(len(s.text) for s in self.segments)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrieval unit.  ``chunk_id`` is assigned by the database on persist."""

    index: int
    text: str
    start_s: float
    end_s: float
    seg_start: int          # first source segment index (inclusive)
    seg_end: int            # last source segment index (inclusive)
    chunk_id: int | None = None
    video_id: str | None = None

    @property
    def n_chars(self) -> int:
        return len(self.text)

    @property
    def span_s(self) -> float:
        return self.end_s - self.start_s

    def overlaps(self, start_s: float, end_s: float) -> float:
        """Seconds of overlap with ``[start_s, end_s]`` (0.0 if disjoint)."""
        return max(0.0, min(self.end_s, end_s) - max(self.start_s, start_s))

    def iou(self, start_s: float, end_s: float) -> float:
        """Intersection-over-union with a time span — used by the eval harness."""
        inter = self.overlaps(start_s, end_s)
        if inter <= 0.0:
            return 0.0
        union = max(self.end_s, end_s) - min(self.start_s, start_s)
        return inter / union if union > 0 else 0.0


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    """Everything that changes chunk boundaries.  Hashed into the cache key."""

    max_chars: int = 900
    overlap_chars: int = 150
    min_chars: int = 250
    words_per_window: int = 28          # fallback unit size for unpunctuated ASR
    punctuation_density_threshold: float = 0.004
    version: int = 1                    # bump when the algorithm itself changes

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A retrieval hit.  ``components`` records per-retriever scores for hybrid."""

    chunk: Chunk
    score: float
    rank: int
    retriever: str
    components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Citation:
    marker: int              # the [n] used in the answer
    chunk_id: int | None
    start_s: float
    end_s: float
    url: str
    timestamp: str           # "12:43"
    quote: str               # short excerpt for /sources


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    refused: bool
    rewritten_query: str | None = None
    retrieved: tuple[ScoredChunk, ...] = ()
    retriever: str = ""
    latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class Turn:
    role: str                # "user" | "assistant"
    content: str
    citations: tuple[Citation, ...] = ()


def format_timestamp(seconds: float) -> str:
    """12:43 for <1h, 1:02:07 for >=1h.  Always floors — a citation must never
    point *past* the moment it references."""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def timestamp_url(video_id: str, seconds: float) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={max(0, int(seconds))}s"


def dedupe_preserve_order(items: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def chunks_time_coverage(chunks: Sequence[Chunk]) -> float:
    """Union of chunk time spans, in seconds.  Used as a chunking sanity check."""
    if not chunks:
        return 0.0
    spans = sorted((c.start_s, c.end_s) for c in chunks)
    total, cur_start, cur_end = 0.0, *spans[0]
    for s, e in spans[1:]:
        if s > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = s, e
        else:
            cur_end = max(cur_end, e)
    return total + (cur_end - cur_start)