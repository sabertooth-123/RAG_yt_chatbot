from __future__ import annotations

from pathlib import Path

import pytest

from ytchat.models import RawTranscript, TranscriptKind, TranscriptSegment, VideoMetadata

VIDEO_ID = "dQw4w9WgXcQ"

from ytchat.config import Settings
from ytchat.embeddings.hashing import HashingEmbedder
from ytchat.ingestion.metadata import StaticMetadataProvider
from ytchat.ingestion.transcript import StaticTranscriptProvider


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    """Offline-safe settings: hashing embedder, small chunks, no network, no real keys.

    ``_env_file=None`` plus the delenv calls keep the developer's real .env out
    of the test run — without them a live API key ends up in pytest failure
    reprs, which is how credentials leak into logs and CI output.
    """
    for var in (
        "YTCHAT_GOOGLE_API_KEY",
        "YTCHAT_GROQ_API_KEY",
        "YTCHAT_OPENROUTER_API_KEY",
        "YTCHAT_OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        embedder="hashing",
        llm_provider="scripted",
        max_chars=160,
        overlap_chars=40,
        min_chars=40,
        top_k=3,
        candidate_k=10,
        enable_whisper_fallback=False,
    )


@pytest.fixture
def providers(punctuated_transcript, metadata):
    return (
        StaticTranscriptProvider({metadata.video_id: punctuated_transcript}),
        StaticMetadataProvider({metadata.video_id: metadata}),
    )


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=128)


@pytest.fixture
def index(settings, providers, embedder, metadata):
    """An indexed video stored at ``settings.db_path``.

    It must be that path, not ``tmp_db`` — tests that reopen the cache use
    ``Repository(settings.db_path)``, and a mismatch means they silently open a
    second, empty database.
    """
    from ytchat.database.repository import Repository
    from ytchat.pipeline import ensure_indexed

    transcripts, metas = providers
    settings.ensure_dirs()
    with Repository(settings.db_path) as repo:
        yield ensure_indexed(
            metadata.video_id, settings, repo,
            transcript_provider=transcripts, metadata_provider=metas, embedder=embedder,
        )


def _seg(i: int, text: str, start: float, dur: float) -> TranscriptSegment:
    return TranscriptSegment(index=i, text=text, start_s=start, duration_s=dur)


@pytest.fixture
def punctuated_transcript() -> RawTranscript:
    """Human-written style captions: punctuated, clean, no rolling duplication."""
    rows = [
        ("Welcome to this lecture on transformers.", 0.0, 3.0),
        ("Today we will discuss the attention mechanism.", 3.0, 3.5),
        ("Attention allows the model to focus on relevant tokens.", 6.5, 4.0),
        ("Self-attention computes queries, keys and values.", 10.5, 4.0),
        ("The scaled dot product is divided by the square root of the dimension.", 14.5, 5.0),
        ("This keeps gradients stable during training.", 19.5, 3.0),
        ("Multi-head attention runs several attention layers in parallel.", 22.5, 4.5),
        ("Each head learns a different relationship between tokens.", 27.0, 4.0),
        ("Positional encodings inject order information into the model.", 31.0, 4.5),
        ("Without them the model would treat the input as a bag of words.", 35.5, 4.5),
    ]
    return RawTranscript(
        video_id=VIDEO_ID,
        language="en",
        kind=TranscriptKind.MANUAL,
        segments=tuple(_seg(i, t, s, d) for i, (t, s, d) in enumerate(rows)),
    )


@pytest.fixture
def asr_transcript() -> RawTranscript:
    """Auto-generated style: no punctuation, rolling duplication, [Music] noise."""
    rows = [
        ("[Music]", 0.0, 2.0),
        ("so today we are going to", 2.0, 1.6),
        ("we are going to talk about neural", 3.6, 1.6),
        ("talk about neural networks and how they", 5.2, 1.8),
        ("networks and how they learn from data", 7.0, 1.9),
        ("learn from data using gradient descent", 8.9, 2.0),
        ("using gradient descent which updates weights", 10.9, 2.1),
        ("which updates weights step by step", 13.0, 1.8),
        ("[Applause]", 14.8, 1.0),
        ("step by step until the loss converges", 15.8, 2.2),
    ]
    return RawTranscript(
        video_id="asrvideo123",
        language="en",
        kind=TranscriptKind.GENERATED,
        segments=tuple(_seg(i, t, s, d) for i, (t, s, d) in enumerate(rows)),
    )


@pytest.fixture
def metadata() -> VideoMetadata:
    return VideoMetadata(
        video_id=VIDEO_ID,
        title="Transformers Explained",
        channel="Test Channel",
        duration_s=40.0,
        language="en",
        transcript_kind=TranscriptKind.MANUAL,
    )


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "cache.sqlite3"

from ytchat.generation.llm import ScriptedLLM


@pytest.fixture
def scripted_llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def session(index, scripted_llm):
    from ytchat.pipeline import ChatSession

    return ChatSession(index, llm=scripted_llm, retriever_name="hybrid")