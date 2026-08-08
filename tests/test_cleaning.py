from ytchat.preprocessing.clean import clean_text, clean_transcript


def test_strips_annotations_and_speakers() -> None:
    assert clean_text("[Music] hello world") == "hello world"
    assert clean_text(">> SPEAKER: welcome back") == "welcome back"
    assert clean_text("JOHN SMITH: hi there") == "hi there"
    assert clean_text("a  b\n c") == "a b c"
    assert clean_text("(Applause)") == ""


def test_removes_rolling_duplication(asr_transcript) -> None:
    cleaned = clean_transcript(asr_transcript)
    text = " ".join(s.text for s in cleaned.segments).lower()
    # "we are going to" appears once in the raw stream then repeats; after
    # de-duplication it must survive exactly once.
    assert text.count("we are going to") == 1
    assert text.count("gradient descent") == 1
    assert "[music]" not in text and "[applause]" not in text


def test_cleaning_never_alters_timing(punctuated_transcript) -> None:
    cleaned = clean_transcript(punctuated_transcript)
    starts_before = [s.start_s for s in punctuated_transcript.segments]
    starts_after = [s.start_s for s in cleaned.segments]
    assert starts_after == starts_before


def test_segments_are_reindexed_contiguously(asr_transcript) -> None:
    cleaned = clean_transcript(asr_transcript)
    assert [s.index for s in cleaned.segments] == list(range(len(cleaned.segments)))