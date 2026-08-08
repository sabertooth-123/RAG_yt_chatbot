"""Typed error hierarchy. The CLI maps these to friendly messages and exit codes."""

from __future__ import annotations


class YtChatError(Exception):
    """Base class for all expected, user-facing failures."""

    exit_code: int = 1


class InvalidVideoURLError(YtChatError):
    exit_code = 2


class TranscriptUnavailableError(YtChatError):
    """No usable captions (disabled, none in requested languages, or video unavailable)."""

    exit_code = 3


class MetadataUnavailableError(YtChatError):
    exit_code = 4


class CacheError(YtChatError):
    exit_code = 5


class EmbeddingError(YtChatError):
    exit_code = 6


class RetrievalError(YtChatError):
    exit_code = 7


class LLMError(YtChatError):
    exit_code = 8


class ConfigurationError(YtChatError):
    exit_code = 9