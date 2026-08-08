"""Configuration: env vars (.env) < CLI flags.

One ``Settings`` class, one source of truth.  Every tunable that affects output
lives here so an experiment is a config change, not a code change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ytchat.models import ChunkerConfig

RetrieverName = Literal["dense", "sparse", "hybrid"]
FusionMode = Literal["rrf", "weighted"]
LLMProvider = Literal["gemini", "groq", "openrouter", "ollama", "scripted"]

# Free-tier defaults per provider.  Only the model name and the rate limit
# differ; three of the four share one OpenAI-compatible adapter.
PROVIDER_DEFAULTS: dict[str, dict[str, object]] = {
    "gemini":     {"model": "gemini-2.0-flash",               "rpm": 12},
    "groq":       {"model": "llama-3.3-70b-versatile",        "rpm": 25},
    "openrouter": {"model": "deepseek/deepseek-chat-v3:free", "rpm": 15},
    "ollama":     {"model": "qwen2.5:7b-instruct",            "rpm": 0},
    "scripted":   {"model": "scripted",                       "rpm": 0},
}

OPENAI_COMPATIBLE_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}


def default_data_dir() -> Path:
    return Path.home() / ".yt-chat"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="YTCHAT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- storage ----------------------------------------------------------
    data_dir: Path = Field(default_factory=default_data_dir)

    # ---- ingestion --------------------------------------------------------
    languages: list[str] = Field(default=["en", "en-US", "en-GB"])
    prefer_manual_captions: bool = True

    # ---- chunking ---------------------------------------------------------
    # 700 chars (~36s of speech) is the measured optimum, not a guess: the sweep
    # in scripts/sweep_chunk_size.py found it peaks BOTH Recall@5 (0.83) and
    # citation IoU (0.153) across three videos.  Citation precision follows an
    # inverted-U -- it doubles from 400 to 700 and then falls -- because IoU is
    # maximised when chunk duration matches the span in which answers actually
    # appear.  The previous default of 900 sat just past the peak.
    max_chars: int = 700
    overlap_chars: int = 120
    min_chars: int = 200

    # ---- embeddings -------------------------------------------------------
    embedder: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 64

    # ---- retrieval --------------------------------------------------------
    retriever: RetrieverName = "hybrid"
    top_k: int = 5
    candidate_k: int = 30           # pre-fusion / pre-rerank depth
    fusion: FusionMode = "rrf"
    rrf_k: int = 60
    hybrid_alpha: float = 0.5       # weight on dense, only for fusion="weighted"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ---- reranking --------------------------------------------------------
    # Off by default: it adds a model download and real latency, so it should be
    # switched on because the eval numbers justify it, not because it exists.
    # Note that enabling it changes the confidence distribution (sigmoid of a
    # cross-encoder logit, not cosine), so min_score must be recalibrated.
    enable_rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 30     # shortlist depth fed to the cross-encoder
    rerank_batch_size: int = 32

    # ---- refusal ----------------------------------------------------------
    min_score: float = 0.28         # Layer A gate; calibrated in Stage 4
    require_valid_citations: bool = True

    # ---- generation -------------------------------------------------------
    llm_provider: LLMProvider = "gemini"
    llm_model: str | None = None            # None → PROVIDER_DEFAULTS
    temperature: float = 0.0                # extraction, not writing
    max_output_tokens: int = 1024
    max_context_chars: int = 8000
    enable_query_rewriting: bool = True
    history_turns_for_rewrite: int = 4
    llm_requests_per_minute: int | None = None   # None → provider default
    llm_max_retries: int = 4
    llm_timeout_s: float = 60.0

    # ---- transcript fallback ----------------------------------------------
    # Requires ingestion/whisper_provider.py and: pip install 'yt-chat[whisper]'
    enable_whisper_fallback: bool = False
    whisper_model: str = "small"            # tiny|base|small|medium|large-v3
    whisper_device: str = "auto"            # auto|cuda|cpu
    whisper_compute_type: str = "auto"      # float16 on CUDA, int8 on CPU
    whisper_vad_filter: bool = True         # skip silence — big speedup
    keep_downloaded_audio: bool = False

    # ---- api keys (all free tiers) ----------------------------------------
    google_api_key: str | None = None       # aistudio.google.com/apikey
    groq_api_key: str | None = None         # console.groq.com/keys
    openrouter_api_key: str | None = None   # openrouter.ai/keys

    @field_validator("data_dir", mode="after")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser().resolve()

    # ---- derived paths ----------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "ytchat.sqlite3"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def chunker_config(self) -> ChunkerConfig:
        return ChunkerConfig(
            max_chars=self.max_chars,
            overlap_chars=self.overlap_chars,
            min_chars=self.min_chars,
        )

    # ---- llm resolution ---------------------------------------------------
    @property
    def resolved_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return str(PROVIDER_DEFAULTS[self.llm_provider]["model"])

    @property
    def resolved_rpm(self) -> int:
        if self.llm_requests_per_minute is not None:
            return self.llm_requests_per_minute
        return int(PROVIDER_DEFAULTS[self.llm_provider]["rpm"])

    def api_key_for(self, provider: str) -> str | None:
        return {
            "gemini": self.google_api_key,
            "groq": self.groq_api_key,
            "openrouter": self.openrouter_api_key,
            "ollama": "ollama",          # local server ignores it, SDK requires one
            "scripted": "scripted",
        }.get(provider)


def load_settings(**overrides: object) -> Settings:
    """CLI flags win over env/.env.  ``None`` values are dropped so Typer's
    unset options don't clobber configured defaults."""
    clean = {k: v for k, v in overrides.items() if v is not None}
    return Settings(**clean)  # type: ignore[arg-type]
