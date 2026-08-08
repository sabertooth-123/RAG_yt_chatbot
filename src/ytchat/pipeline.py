"""Orchestration: URL → fully indexed, queryable video → conversation.

Cache chain, checked stage by stage:

    video (video_id)
      └─ chunk_set (chunker fingerprint)
           └─ embedding_set (embedder model_id)
                └─ vector index (disk, disposable)

Each stage is skipped independently.  Changing ``--max-chars`` rebuilds chunks
and embeddings but never re-downloads the transcript; changing the embedding
model reuses chunks and rebuilds only vectors.  That property is what makes the
parameter sweeps in the research plan cheap enough to actually run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ytchat.config import Settings
from ytchat.database.repository import Repository
from ytchat.database.vectorstore import VectorStore
from ytchat.embeddings.base import Embedder
from ytchat.embeddings.factory import build_embedder
from ytchat.generation.answerer import Answerer, to_turns
from ytchat.generation.llm import LLM, build_llm
from ytchat.generation.rewriter import QueryRewriter
from ytchat.ingestion.chain import FallbackTranscriptProvider
from ytchat.ingestion.metadata import BestEffortMetadataProvider, MetadataProvider
from ytchat.ingestion.transcript import TranscriptProvider, YouTubeTranscriptProvider
from ytchat.ingestion.url import parse_video_id
from ytchat.models import Answer, Chunk, ScoredChunk, Turn, VideoMetadata
from ytchat.preprocessing.chunking import TranscriptChunker
from ytchat.preprocessing.clean import clean_transcript
from ytchat.retrieval.base import Retriever
from ytchat.retrieval.factory import build_retriever

Progress = Callable[[str], None]


def _noop(_msg: str) -> None:
    return None


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


@dataclass
class IndexStats:
    transcript_cached: bool = False
    chunks_cached: bool = False
    embeddings_cached: bool = False
    n_segments: int = 0
    n_chunks: int = 0
    elapsed_s: float = 0.0

    @property
    def fully_cached(self) -> bool:
        return self.transcript_cached and self.chunks_cached and self.embeddings_cached

    def summary(self) -> str:
        if self.fully_cached:
            return f"Loaded from cache: {self.n_chunks} chunks ({self.elapsed_s:.2f}s)"
        built = [
            name
            for name, cached in (
                ("transcript", self.transcript_cached),
                ("chunks", self.chunks_cached),
                ("embeddings", self.embeddings_cached),
            )
            if not cached
        ]
        return (
            f"Processed {', '.join(built)} → {self.n_chunks} chunks "
            f"from {self.n_segments} caption segments ({self.elapsed_s:.1f}s)"
        )


@dataclass
class VideoIndex:
    """Everything needed to answer questions about one video."""

    metadata: VideoMetadata
    chunks: list[Chunk]
    chunk_set_id: int
    store: VectorStore | None
    embedder: Embedder | None
    settings: Settings
    stats: IndexStats = field(default_factory=IndexStats)

    @property
    def video_id(self) -> str:
        return self.metadata.video_id

    def retriever(self, name: str | None = None) -> Retriever:
        return build_retriever(
            name or self.settings.retriever,
            self.settings,
            self.chunks,
            self.store,
            self.embedder,
        )


def build_transcript_provider(
    settings: Settings, progress: Progress = _noop
) -> TranscriptProvider:
    providers: list[TranscriptProvider] = [
        YouTubeTranscriptProvider(prefer_manual=settings.prefer_manual_captions)
    ]
    if getattr(settings, "enable_whisper_fallback", False):
        from ytchat.ingestion.whisper_provider import WhisperTranscriptProvider

        providers.append(
            WhisperTranscriptProvider(
                audio_dir=settings.audio_dir,
                model_size=settings.whisper_model,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
                vad_filter=settings.whisper_vad_filter,
                keep_audio=settings.keep_downloaded_audio,
                progress=progress,
            )
        )
    return FallbackTranscriptProvider(providers, progress=progress)


def ensure_indexed(
    url_or_id: str,
    settings: Settings,
    repo: Repository,
    *,
    transcript_provider: TranscriptProvider | None = None,
    metadata_provider: MetadataProvider | None = None,
    embedder: Embedder | None = None,
    progress: Progress = _noop,
    force: bool = False,
) -> VideoIndex:
    started = time.perf_counter()
    settings.ensure_dirs()
    video_id = parse_video_id(url_or_id)
    stats = IndexStats()

    if force:
        progress("Force refresh: clearing cached data for this video…")
        repo.clear_video(video_id)

    # ---- stage 1: transcript + metadata ---------------------------------
    transcript = None if force else repo.get_transcript(video_id)
    metadata = None if force else repo.get_video(video_id)

    if transcript is None or metadata is None:
        provider = transcript_provider or build_transcript_provider(settings, progress)
        meta_provider = metadata_provider or BestEffortMetadataProvider()
        progress(f"Fetching transcript for {video_id}…")
        raw = provider.fetch(video_id, settings.languages)
        metadata = meta_provider.fetch(video_id)
        transcript = clean_transcript(raw)
        repo.save_video(metadata, transcript)
        progress(
            f"Transcript: {len(transcript.segments)} segments, {transcript.kind.value}."
        )
    else:
        stats.transcript_cached = True

    stats.n_segments = len(transcript.segments)

    # ---- stage 2: chunks -------------------------------------------------
    chunker_config = settings.chunker_config()
    fingerprint = chunker_config.fingerprint()
    chunk_set_id = repo.get_chunk_set_id(video_id, fingerprint)

    if chunk_set_id is None:
        progress("Chunking transcript…")
        new_chunks = TranscriptChunker(chunker_config).chunk(transcript.segments)
        chunk_set_id = repo.save_chunks(video_id, chunker_config, new_chunks)
    else:
        stats.chunks_cached = True

    chunks = repo.get_chunks(chunk_set_id)  # always reload: chunk_ids come from the DB
    stats.n_chunks = len(chunks)

    # ---- stage 3: embeddings + vector index -----------------------------
    embedder = embedder or build_embedder(settings)
    store = VectorStore.load_or_build(
        repo, video_id, chunk_set_id, embedder.model_id, settings.index_dir
    )

    if store is None:
        progress(f"Embedding {len(chunks)} chunks with {embedder.model_id}…")
        matrix = embedder.encode_documents([c.text for c in chunks])
        chunk_ids = [c.chunk_id for c in chunks if c.chunk_id is not None]
        repo.save_embeddings(chunk_set_id, embedder.model_id, chunk_ids, matrix)
        store = VectorStore.build(matrix, chunk_ids)
        store.save(
            VectorStore.index_path(
                settings.index_dir, video_id, chunk_set_id, embedder.model_id
            )
        )
        progress(f"Index ready ({store.backend} backend, dim={matrix.shape[1]}).")
    else:
        stats.embeddings_cached = True

    stats.elapsed_s = time.perf_counter() - started
    progress(stats.summary())

    return VideoIndex(
        metadata=metadata,
        chunks=chunks,
        chunk_set_id=chunk_set_id,
        store=store,
        embedder=embedder,
        settings=settings,
        stats=stats,
    )


def retrieve(
    index: VideoIndex,
    query: str,
    k: int | None = None,
    retriever: str | None = None,
) -> list[ScoredChunk]:
    return index.retriever(retriever).search(query, k or index.settings.top_k)


def compare_retrievers(
    index: VideoIndex,
    query: str,
    k: int = 5,
    names: Sequence[str] = ("dense", "sparse", "hybrid"),
) -> dict[str, list[ScoredChunk]]:
    """Side-by-side retrieval for the CLI's ``/compare`` command and eval runner."""
    return {name: index.retriever(name).search(query, k) for name in names}


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------


class ChatSession:
    """One conversation about one video.

    History lives in memory for query rewriting and is mirrored to SQLite so
    ``/history`` survives a restart.
    """

    def __init__(
        self,
        index: VideoIndex,
        llm: LLM | None = None,
        retriever_name: str | None = None,
        repo: Repository | None = None,
    ) -> None:
        self.index = index
        self.settings = index.settings
        self.llm = llm or build_llm(self.settings)
        self.retriever_name = retriever_name or self.settings.retriever
        self.repo = repo
        self.history: list[Turn] = []
        self.last_answer: Answer | None = None

        self.rewriter = QueryRewriter(
            self.llm,
            max_turns=self.settings.history_turns_for_rewrite,
            enabled=self.settings.enable_query_rewriting,
        )
        self.answerer = Answerer(self.llm, self.settings)
        self._retriever = self.index.retriever(self.retriever_name)
        self.conversation_id = (
            repo.create_conversation(index.video_id, self.retriever_name) if repo else None
        )

    # -- core ---------------------------------------------------------------
    def ask(self, question: str, k: int | None = None) -> Answer:
        query, rewritten = self.rewriter.rewrite(question, self.history)
        hits = self._retriever.search(query, k or self.settings.top_k)
        answer = self.answerer.answer(
            question=question,
            hits=hits,
            video_id=self.index.video_id,
            video_title=self.index.metadata.title,
            rewritten_query=query if rewritten else None,
            retriever=self.retriever_name,
        )
        user_turn, assistant_turn = to_turns(question, answer)
        self.history.extend([user_turn, assistant_turn])
        self.last_answer = answer

        if self.repo and self.conversation_id is not None:
            self.repo.add_message(self.conversation_id, "user", question)
            self.repo.add_message(
                self.conversation_id, "assistant", answer.text, answer.citations
            )
        return answer

    # -- session controls ---------------------------------------------------
    def change_retriever(self, name: str) -> None:
        self._retriever = self.index.retriever(name)  # raises before mutating state
        self.retriever_name = name

    def clear_history(self) -> None:
        self.history.clear()
        self.last_answer = None

    @property
    def refusal_trace(self):
        return self.answerer.last_trace