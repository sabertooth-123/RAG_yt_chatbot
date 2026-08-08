import pytest

from ytchat.models import Chunk, format_timestamp, timestamp_url


@pytest.mark.parametrize(
    "seconds,expected",
    [(0, "0:00"), (7, "0:07"), (63, "1:03"), (763, "12:43"),
     (3600, "1:00:00"), (3727, "1:02:07"), (763.9, "12:43")],
)
def test_format_timestamp(seconds, expected) -> None:
    assert format_timestamp(seconds) == expected


def test_timestamp_url_floors_and_never_goes_negative() -> None:
    assert timestamp_url("abc", 763.9) == "https://www.youtube.com/watch?v=abc&t=763s"
    assert timestamp_url("abc", -5) == "https://www.youtube.com/watch?v=abc&t=0s"


def test_chunk_iou() -> None:
    c = Chunk(index=0, text="x", start_s=10.0, end_s=20.0, seg_start=0, seg_end=1)
    assert c.iou(10.0, 20.0) == pytest.approx(1.0)
    assert c.iou(15.0, 25.0) == pytest.approx(5 / 15)
    assert c.iou(30.0, 40.0) == 0.0