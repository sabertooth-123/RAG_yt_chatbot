"""YouTube URL / ID parsing.  Pure, offline, exhaustively tested."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from ytchat.errors import InvalidVideoURLError

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

_PATH_PATTERNS = (
    re.compile(r"^/embed/(?P<id>[A-Za-z0-9_-]{11})"),
    re.compile(r"^/shorts/(?P<id>[A-Za-z0-9_-]{11})"),
    re.compile(r"^/live/(?P<id>[A-Za-z0-9_-]{11})"),
    re.compile(r"^/v/(?P<id>[A-Za-z0-9_-]{11})"),
)

_YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "music.youtube.com", "youtube-nocookie.com", "www.youtube-nocookie.com",
}
_SHORT_HOSTS = {"youtu.be", "www.youtu.be"}


def parse_video_id(value: str) -> str:
    """Accept a full URL in any common shape, or a bare 11-character video ID."""
    if not value or not value.strip():
        raise InvalidVideoURLError("No URL or video ID provided.")

    raw = value.strip()

    if _VIDEO_ID_RE.match(raw):
        return raw

    candidate = raw if "://" in raw else f"https://{raw}"
    parsed = urlparse(candidate)
    host = (parsed.netloc or "").lower()

    if host in _SHORT_HOSTS:
        vid = parsed.path.lstrip("/").split("/")[0]
        if _VIDEO_ID_RE.match(vid):
            return vid
        raise InvalidVideoURLError(f"Could not extract a video ID from {raw!r}.")

    if host in _YOUTUBE_HOSTS:
        qs = parse_qs(parsed.query)
        if "v" in qs and qs["v"]:
            vid = qs["v"][0]
            if _VIDEO_ID_RE.match(vid):
                return vid
        for pattern in _PATH_PATTERNS:
            m = pattern.match(parsed.path)
            if m:
                return m.group("id")

    raise InvalidVideoURLError(
        f"{raw!r} is not a recognisable YouTube URL or video ID."
    )


def parse_start_seconds(value: str) -> int | None:
    """Extract ``t=``/``start=`` from a URL.  Supports '763', '763s', '12m43s'."""
    parsed = urlparse(value if "://" in value else f"https://{value}")
    qs = parse_qs(parsed.query)
    raw = (qs.get("t") or qs.get("start") or [None])[0]
    if raw is None:
        return None
    if raw.isdigit():
        return int(raw)
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", raw)
    if not m or not any(m.groups()):
        return None
    h, mnt, s = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mnt * 60 + s