"""Caption cleaning.

Auto-generated YouTube captions have two artifacts that wreck naive pipelines:

1. Non-speech annotations: ``[Music]``, ``[Applause]``, ``(laughter)``.
2. Rolling duplication: consecutive cues repeat the previous cue's tail, e.g.
   ``"and so the model"`` → ``"and so the model learns to"``.  Left in place,
   this inflates BM25 term frequencies and produces near-duplicate chunks.
"""

from __future__ import annotations

import re
import unicodedata

from ytchat.models import RawTranscript, TranscriptSegment

_ANNOTATION = re.compile(r"[\[\(](?:music|applause|laughter|inaudible|silence|"
                         r"foreign|blank_audio|crosstalk)[^\])]*[\]\)]", re.IGNORECASE)
_SPEAKER = re.compile(r"^\s*(?:>>+|-\s|[A-Z][A-Z .'-]{1,24}:)\s*")
_TAGS = re.compile(r"</?[^>]{1,20}>")
_WS = re.compile(r"\s+")


def _strip_speaker_prefixes(text: str) -> str:
    """Repeatedly strip leading speaker markers.

    Captions stack them: ``'>> JOHN SMITH: hello there'``.  ``_SPEAKER`` is
    anchored with ``^``, so a single sub removes only the outermost marker and
    never rescans from the new start \u2014 hence the loop.
    """
    for _ in range(3):
        stripped = _SPEAKER.sub("", text, count=1)
        if stripped == text:
            return text
        text = stripped
    return text


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\xa0", " ")
    text = _TAGS.sub(" ", text)
    text = _ANNOTATION.sub(" ", text)
    text = _strip_speaker_prefixes(text)
    text = text.replace("\n", " ")
    return _WS.sub(" ", text).strip()


def _strip_rolling_overlap(prev: str, curr: str, max_overlap_words: int = 12) -> str:
    """Remove the leading portion of ``curr`` that merely repeats the tail of ``prev``."""
    if not prev or not curr:
        return curr
    p_words = prev.split()
    c_words = curr.split()
    limit = min(max_overlap_words, len(p_words), len(c_words))
    for n in range(limit, 0, -1):
        if [w.lower() for w in p_words[-n:]] == [w.lower() for w in c_words[:n]]:
            return " ".join(c_words[n:])
    return curr


def clean_transcript(
    transcript: RawTranscript, *, drop_rolling_duplicates: bool = True
) -> RawTranscript:
    """Clean each cue, drop empties, and reindex.

    Timing is never altered — only text.  A cue whose text is fully consumed by
    de-duplication is dropped, and its time span is absorbed by its neighbours
    through the timeline interpolation downstream.
    """
    cleaned: list[TranscriptSegment] = []
    prev_text = ""
    for seg in transcript.segments:
        text = clean_text(seg.text)
        if drop_rolling_duplicates:
            text = _strip_rolling_overlap(prev_text, text).strip()
        if not text:
            continue
        cleaned.append(
            TranscriptSegment(
                index=len(cleaned),
                text=text,
                start_s=seg.start_s,
                duration_s=max(0.0, seg.duration_s),
            )
        )
        prev_text = text

    return RawTranscript(
        video_id=transcript.video_id,
        language=transcript.language,
        kind=transcript.kind,
        segments=tuple(cleaned),
    )