"""Video metadata: yt-dlp when installed, oEmbed as a keyless fallback."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ytchat.models import TranscriptKind, VideoMetadata


@runtime_checkable
class MetadataProvider(Protocol):
    def fetch(self, video_id: str) -> VideoMetadata: ...


class BestEffortMetadataProvider:
    """Never fails: degrades from yt-dlp → oEmbed → a placeholder title.

    Metadata is cosmetic; a missing title must not block answering questions.
    """

    def __init__(self, timeout: float = 8.0) -> None:
        self.timeout = timeout

    def fetch(self, video_id: str) -> VideoMetadata:
        return (
            self._via_ytdlp(video_id)
            or self._via_oembed(video_id)
            or VideoMetadata(video_id=video_id, title=f"YouTube video {video_id}")
        )

    def _via_ytdlp(self, video_id: str) -> VideoMetadata | None:
        try:
            import yt_dlp  # type: ignore[import-not-found]
        except ImportError:
            return None
        opts = {"quiet": True, "no_warnings": True, "skip_download": True,
                "socket_timeout": self.timeout, "extract_flat": False}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}", download=False
                )
        except Exception:
            return None
        if not info:
            return None
        return VideoMetadata(
            video_id=video_id,
            title=info.get("title") or f"YouTube video {video_id}",
            channel=info.get("uploader") or info.get("channel"),
            duration_s=float(info["duration"]) if info.get("duration") else None,
            language=info.get("language"),
            extra={
                "upload_date": info.get("upload_date"),
                "view_count": info.get("view_count"),
            },
        )

    def _via_oembed(self, video_id: str) -> VideoMetadata | None:
        try:
            import httpx
        except ImportError:
            return None
        url = "https://www.youtube.com/oembed"
        try:
            resp = httpx.get(
                url,
                params={"url": f"https://www.youtube.com/watch?v={video_id}",
                        "format": "json"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return None
        return VideoMetadata(
            video_id=video_id,
            title=data.get("title") or f"YouTube video {video_id}",
            channel=data.get("author_name"),
        )


class StaticMetadataProvider:
    """Test double."""

    def __init__(self, metadata: dict[str, VideoMetadata] | None = None) -> None:
        self._metadata = metadata or {}

    def fetch(self, video_id: str) -> VideoMetadata:
        return self._metadata.get(
            video_id,
            VideoMetadata(
                video_id=video_id,
                title=f"Fixture video {video_id}",
                channel="Fixture Channel",
                transcript_kind=TranscriptKind.MANUAL,
            ),
        )