"""Provider chaining: try each transcript source in order, first success wins.

This is the seam that makes "any random YouTube video" work.  Captions cover
roughly 80% of videos; the rest need local ASR.  Because every provider returns
the same ``RawTranscript`` shape, nothing downstream in the pipeline knows or
cares which source produced the text.
"""

from __future__ import annotations

from typing import Callable, Sequence

from ytchat.errors import TranscriptUnavailableError
from ytchat.ingestion.transcript import TranscriptProvider
from ytchat.models import RawTranscript


class FallbackTranscriptProvider:
    """Try providers in order until one succeeds.

    Only ``TranscriptUnavailableError`` is treated as "try the next source" —
    any other exception propagates, so a genuine bug in a provider surfaces
    immediately instead of being silently swallowed as a fallback.
    """

    def __init__(
        self,
        providers: Sequence[TranscriptProvider],
        progress: Callable[[str], None] | None = None,
    ) -> None:
        if not providers:
            raise ValueError("FallbackTranscriptProvider needs at least one provider.")
        self.providers = list(providers)
        self.progress = progress or (lambda _msg: None)
        self.attempts: list[tuple[str, str]] = []  # (provider, outcome) — surfaced by /debug

    def fetch(self, video_id: str, languages: Sequence[str]) -> RawTranscript:
        failures: list[str] = []
        for provider in self.providers:
            name = type(provider).__name__
            try:
                transcript = provider.fetch(video_id, languages)
                self.attempts.append((name, "ok"))
                return transcript
            except TranscriptUnavailableError as exc:
                self.attempts.append((name, str(exc)))
                failures.append(f"  • {name}: {exc}")
                self.progress(f"{name} failed, trying next source…")

        raise TranscriptUnavailableError(
            f"Could not obtain a transcript for {video_id} from any source:\n"
            + "\n".join(failures)
            + "\n\nThe video may be private, age-restricted, or have no spoken audio."
        )
