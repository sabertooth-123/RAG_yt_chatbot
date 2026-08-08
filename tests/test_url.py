import pytest

from ytchat.errors import InvalidVideoURLError
from ytchat.ingestion.url import parse_start_seconds, parse_video_id

VID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "value",
    [
        VID,
        f"https://www.youtube.com/watch?v={VID}",
        f"http://youtube.com/watch?v={VID}",
        f"https://m.youtube.com/watch?v={VID}&feature=share",
        f"https://www.youtube.com/watch?app=desktop&v={VID}&t=763s",
        f"https://youtu.be/{VID}",
        f"https://youtu.be/{VID}?t=100",
        f"https://www.youtube.com/embed/{VID}",
        f"https://www.youtube.com/shorts/{VID}",
        f"https://www.youtube.com/live/{VID}",
        f"www.youtube.com/watch?v={VID}",
        f"https://music.youtube.com/watch?v={VID}",
    ],
)
def test_parses_all_url_shapes(value: str) -> None:
    assert parse_video_id(value) == VID


@pytest.mark.parametrize(
    "value", ["", "   ", "not a url", "https://vimeo.com/12345",
              "https://www.youtube.com/watch?v=tooshort", "https://youtube.com/"],
)
def test_rejects_invalid(value: str) -> None:
    with pytest.raises(InvalidVideoURLError):
        parse_video_id(value)


def test_parse_start_seconds() -> None:
    assert parse_start_seconds(f"https://youtu.be/{VID}?t=763") == 763
    assert parse_start_seconds(f"https://youtu.be/{VID}?t=763s") == 763
    assert parse_start_seconds(f"https://youtu.be/{VID}?t=12m43s") == 763
    assert parse_start_seconds(f"https://youtu.be/{VID}?t=1h2m7s") == 3727
    assert parse_start_seconds(f"https://youtu.be/{VID}") is None