import pytest

from ytchat.generation.citations import (
    build_citations,
    build_context,
    extract_markers,
    render_sources,
    terminal_link,
)


def test_extract_markers_dedupes_in_order() -> None:
    assert extract_markers("a [2] b [1] c [2] d [3]") == [2, 1, 3]
    assert extract_markers("no markers here") == []


def test_citations_map_to_the_right_chunks(index) -> None:
    hits = index.retriever("hybrid").search("attention", k=3)
    text, cites, invalid = build_citations("First claim [1]. Second claim [3].", hits, "vid123")
    assert invalid == []
    assert [c.marker for c in cites] == [1, 3]
    assert cites[0].chunk_id == hits[0].chunk.chunk_id
    assert cites[1].chunk_id == hits[2].chunk.chunk_id


def test_citation_urls_and_timestamps_agree(index) -> None:
    hits = index.retriever("hybrid").search("attention", k=2)
    _, cites, _ = build_citations("Claim [1].", hits, "vid123")
    c = cites[0]
    assert c.url == f"https://www.youtube.com/watch?v=vid123&t={int(c.start_s)}s"
    assert c.start_s == pytest.approx(hits[0].chunk.start_s)


def test_invalid_markers_are_stripped_and_reported(index) -> None:
    hits = index.retriever("hybrid").search("attention", k=2)
    text, cites, invalid = build_citations("Real [1]. Invented [7].", hits, "vid123")
    assert invalid == [7]
    assert "[7]" not in text
    assert "[1]" in text
    assert [c.marker for c in cites] == [1]


def test_all_markers_invalid_yields_no_citations(index) -> None:
    hits = index.retriever("hybrid").search("attention", k=2)
    _, cites, invalid = build_citations("Fabricated [9][8].", hits, "vid123")
    assert cites == () and invalid == [9, 8]


def test_context_numbering_is_one_based_and_ordered(index) -> None:
    hits = index.retriever("hybrid").search("attention", k=3)
    context, used = build_context(hits, max_chars=10_000)
    assert context.startswith("[1] (")
    assert "[2] (" in context and "[3] (" in context
    assert used == list(hits)


def test_context_respects_the_char_budget(index) -> None:
    hits = index.retriever("hybrid").search("attention", k=5)
    context, used = build_context(hits, max_chars=200)
    assert 0 < len(used) < len(hits)
    assert used[0] is hits[0], "the budget must never drop the top-ranked excerpt"


def test_terminal_link_is_osc8() -> None:
    link = terminal_link("https://example.com", "12:43")
    assert link.startswith("\033]8;;https://example.com") and "12:43" in link


def test_render_sources_plain_has_no_escape_codes(index) -> None:
    hits = index.retriever("hybrid").search("attention", k=1)
    _, cites, _ = build_citations("Claim [1].", hits, "vid123")
    assert "\033" not in render_sources(cites, plain=True)