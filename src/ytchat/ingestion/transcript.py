"""Transcript acquisition.

The Protocol is the seam that makes the whole suite runnable offline: tests
inject ``StaticTranscriptProvider`` and never touch the network.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ytchat.errors import TranscriptUnavailableError
from ytchat.models import RawTranscript, TranscriptKind, TranscriptSegment


@runtime_checkable
class TranscriptProvider(Protocol):
    def fetch(self, video_id: str, languages: Sequence[str]) -> RawTranscript: ...


def _to_segments(rows: Sequence[dict]) -> tuple[TranscriptSegment, ...]:
    segs: list[TranscriptSegment] = []
    for i, row in enumerate(rows):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        segs.append(
            TranscriptSegment(
                index=len(segs),
                text=text,
                start_s=float(row.get("start", 0.0)),
                duration_s=float(row.get("duration", 0.0)),
            )
        )
    return tuple(segs)


class YouTubeTranscriptProvider:
    """Wraps ``youtube-transcript-api``.

    Preference order: manual captions in a requested language > generated
    captions in a requested language > translation of any available track.
    """

    def __init__(self, prefer_manual: bool = True) -> None:
        self.prefer_manual = prefer_manual

    def fetch(self, video_id: str, languages: Sequence[str]) -> RawTranscript:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api._errors import (  # type: ignore[attr-defined]
                CouldNotRetrieveTranscript,
            )
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise TranscriptUnavailableError(
                "youtube-transcript-api is not installed. Run: pip install youtube-transcript-api"
            ) from exc

        try:
            listing = self._list_transcripts(YouTubeTranscriptApi, video_id)
            track, kind = self._choose_track(listing, languages)
            rows = [
                {"text": s.text, "start": s.start, "duration": s.duration}
                if hasattr(s, "text") else dict(s)
                for s in track.fetch()
            ]
        except CouldNotRetrieveTranscript as exc:
            raise TranscriptUnavailableError(
                f"No usable captions for video {video_id}: {exc.__class__.__name__}. "
                "The video may have captions disabled, be private, or be region-locked."
            ) from exc
        except TranscriptUnavailableError:
            raise
        except Exception as exc:  # network / API surface changes
            raise TranscriptUnavailableError(
                f"Failed to fetch captions for {video_id}: {exc}"
            ) from exc

        segments = _to_segments(rows)
        if not segments:
            raise TranscriptUnavailableError(
                f"Video {video_id} returned an empty transcript."
            )
        return RawTranscript(
            video_id=video_id,
            language=getattr(track, "language_code", languages[0] if languages else "en"),
            kind=kind,
            segments=segments,
        )

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _list_transcripts(api_cls, video_id: str):
        """Support both the 1.x instance API and the legacy 0.6 classmethod API."""
        if hasattr(api_cls, "list"):          # >= 1.0
            return api_cls().list(video_id)
        return api_cls.list_transcripts(video_id)  # <= 0.6

    def _choose_track(self, listing, languages: Sequence[str]):
        langs = list(languages) or ["en"]
        attempts = (
            [(listing.find_manually_created_transcript, TranscriptKind.MANUAL),
             (listing.find_generated_transcript, TranscriptKind.GENERATED)]
            if self.prefer_manual else
            [(listing.find_generated_transcript, TranscriptKind.GENERATED),
             (listing.find_manually_created_transcript, TranscriptKind.MANUAL)]
        )
        for finder, kind in attempts:
            try:
                return finder(langs), kind
            except Exception:
                continue
        try:  # last resort: translate whatever exists
            for track in listing:
                if track.is_translatable:
                    return track.translate(langs[0]), TranscriptKind.TRANSLATED
        except Exception:
            pass
        raise TranscriptUnavailableError(
            f"No transcript available in {langs} and none translatable."
        )


class StaticTranscriptProvider:
    """Offline provider backed by an in-memory dict — the test double."""

    def __init__(self, transcripts: dict[str, RawTranscript]) -> None:
        self._transcripts = transcripts
        self.calls: list[str] = []

    def fetch(self, video_id: str, languages: Sequence[str]) -> RawTranscript:
        self.calls.append(video_id)
        try:
            return self._transcripts[video_id]
        except KeyError as exc:
            raise TranscriptUnavailableError(
                f"No fixture transcript for {video_id}"
            ) from exc