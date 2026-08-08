import pytest

from ytchat.models import ChunkerConfig, chunks_time_coverage
from ytchat.preprocessing.chunking import (
    TranscriptChunker,
    build_units,
    punctuation_density,
)
from ytchat.preprocessing.clean import clean_transcript
from ytchat.preprocessing.timeline import Timeline


# ---------- timeline -------------------------------------------------------
def test_timeline_interpolates_monotonically(punctuated_transcript) -> None:
    tl = Timeline.build(punctuated_transcript.segments)
    times = [tl.time_at(i) for i in range(0, len(tl.text), 7)]
    assert all(b >= a - 1e-9 for a, b in zip(times, times[1:])), "time_at must be monotonic"
    assert times[0] == pytest.approx(0.0)
    assert times[-1] <= punctuated_transcript.duration_s + 1e-6


def test_timeline_maps_boundaries_exactly(punctuated_transcript) -> None:
    tl = Timeline.build(punctuated_transcript.segments)
    for span in tl.spans:
        assert tl.time_at(span.char_start) == pytest.approx(span.t_start)
        assert tl.time_at(span.char_end) == pytest.approx(span.t_end)


def test_timeline_handles_zero_duration_segments() -> None:
    from ytchat.models import TranscriptSegment

    segs = (
        TranscriptSegment(0, "one", 0.0, 0.0),
        TranscriptSegment(1, "two", 1.0, 0.0),
    )
    tl = Timeline.build(segs)
    assert tl.time_at(0) == pytest.approx(0.0)
    assert tl.time_at(len(tl.text)) >= 1.0


# ---------- units ----------------------------------------------------------
def test_unit_strategy_switches_on_punctuation(punctuated_transcript, asr_transcript) -> None:
    cfg = ChunkerConfig()
    punct_text = Timeline.build(punctuated_transcript.segments).text
    asr_text = Timeline.build(clean_transcript(asr_transcript).segments).text

    assert punctuation_density(punct_text) >= cfg.punctuation_density_threshold
    assert punctuation_density(asr_text) < cfg.punctuation_density_threshold

    # Unpunctuated text must still produce many units, not one giant blob.
    assert len(build_units(asr_text, cfg)) > 1


def test_no_unit_exceeds_max_chars() -> None:
    cfg = ChunkerConfig(max_chars=80)
    text = "word " * 200
    assert all(u.n_chars <= cfg.max_chars for u in build_units(text, cfg))


# ---------- chunks: the invariant -----------------------------------------
@pytest.fixture
def chunks(punctuated_transcript):
    cfg = ChunkerConfig(max_chars=140, overlap_chars=40, min_chars=40)
    return TranscriptChunker(cfg).chunk(clean_transcript(punctuated_transcript).segments)


def test_chunks_are_produced(chunks) -> None:
    assert len(chunks) >= 3


def test_every_chunk_has_a_valid_time_span(chunks, punctuated_transcript) -> None:
    duration = punctuated_transcript.duration_s
    for c in chunks:
        assert c.start_s >= 0.0
        assert c.end_s >= c.start_s, f"chunk {c.index} has an inverted span"
        assert c.end_s <= duration + 1e-6, f"chunk {c.index} runs past the video"
        assert c.text.strip(), f"chunk {c.index} is empty"


def test_chunk_start_times_are_non_decreasing(chunks) -> None:
    starts = [c.start_s for c in chunks]
    assert starts == sorted(starts)


def test_chunks_cover_essentially_the_whole_video(chunks, punctuated_transcript) -> None:
    coverage = chunks_time_coverage(chunks)
    assert coverage >= 0.9 * punctuated_transcript.duration_s


def test_segment_provenance_is_recorded(chunks, punctuated_transcript) -> None:
    n = len(punctuated_transcript.segments)
    for c in chunks:
        assert 0 <= c.seg_start <= c.seg_end < n


def test_overlap_produces_shared_content(punctuated_transcript) -> None:
    cfg = ChunkerConfig(max_chars=140, overlap_chars=60, min_chars=40)
    segs = clean_transcript(punctuated_transcript).segments
    with_overlap = TranscriptChunker(cfg).chunk(segs)
    without = TranscriptChunker(
        ChunkerConfig(max_chars=140, overlap_chars=0, min_chars=40)
    ).chunk(segs)
    assert len(with_overlap) >= len(without)
    # Consecutive overlapping chunks must share at least one word.
    shared = [
        bool(set(a.text.lower().split()) & set(b.text.lower().split()))
        for a, b in zip(with_overlap, with_overlap[1:])
    ]
    assert any(shared)


def test_chunking_is_deterministic(punctuated_transcript) -> None:
    segs = clean_transcript(punctuated_transcript).segments
    cfg = ChunkerConfig(max_chars=200, overlap_chars=50)
    a = TranscriptChunker(cfg).chunk(segs)
    b = TranscriptChunker(cfg).chunk(segs)
    assert [(c.text, c.start_s, c.end_s) for c in a] == [(c.text, c.start_s, c.end_s) for c in b]


def test_asr_transcript_chunks_keep_timestamps(asr_transcript) -> None:
    segs = clean_transcript(asr_transcript).segments
    out = TranscriptChunker(ChunkerConfig(max_chars=100, overlap_chars=20, min_chars=30)).chunk(segs)
    assert out
    assert all(c.end_s >= c.start_s for c in out)
    assert out[0].start_s == pytest.approx(segs[0].start_s, abs=0.01)


def test_empty_transcript_yields_no_chunks() -> None:
    assert TranscriptChunker().chunk(()) == []


def test_fingerprint_changes_with_config() -> None:
    a = ChunkerConfig(max_chars=900)
    b = ChunkerConfig(max_chars=1000)
    assert a.fingerprint() != b.fingerprint()
    assert a.fingerprint() == ChunkerConfig(max_chars=900).fingerprint()