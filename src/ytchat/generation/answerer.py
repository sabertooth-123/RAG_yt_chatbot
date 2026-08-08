"""Grounded answer generation with three independent refusal layers.

A. Retrieval gate  — best calibrated confidence < min_score → refuse without
   calling the LLM at all.  Catches out-of-domain questions cheaply.
B. Prompt sentinel — the model emits INSUFFICIENT_CONTEXT rather than
   improvising.  Catches on-topic questions the video never answers.
C. Citation check  — an answer citing no valid excerpt is not grounded, so it
   is downgraded to a refusal.  Catches confident prose with invented sources.

The layers are independent on purpose: each catches a failure mode the others
miss, and each can be disabled to measure its individual contribution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from ytchat.config import Settings
from ytchat.generation.citations import build_citations, build_context
from ytchat.generation.llm import LLM
from ytchat.generation.prompts import (
    ANSWER_PROMPT,
    ANSWER_SYSTEM,
    REFUSAL_MESSAGE,
    REFUSAL_SENTINEL,
)
from ytchat.models import Answer, ScoredChunk, Turn
from ytchat.retrieval.base import best_confidence


@dataclass
class RefusalTrace:
    """Which layer fired, and why — surfaced by ``/debug`` and the eval report."""

    layer: str | None = None
    detail: str = ""


class Answerer:
    def __init__(self, llm: LLM, settings: Settings) -> None:
        self.llm = llm
        self.settings = settings
        self.last_trace = RefusalTrace()

    def answer(
        self,
        question: str,
        hits: Sequence[ScoredChunk],
        video_id: str,
        video_title: str = "",
        rewritten_query: str | None = None,
        retriever: str = "",
    ) -> Answer:
        started = time.perf_counter()
        self.last_trace = RefusalTrace()

        def _refuse(layer: str, detail: str) -> Answer:
            self.last_trace = RefusalTrace(layer, detail)
            return Answer(
                text=REFUSAL_MESSAGE, citations=(), refused=True,
                rewritten_query=rewritten_query, retrieved=tuple(hits),
                retriever=retriever, latency_ms=(time.perf_counter() - started) * 1000,
            )

        # ---- Layer A -----------------------------------------------------
        if not hits:
            return _refuse("A:retrieval", "no chunks retrieved")
        confidence = best_confidence(hits)
        if confidence < self.settings.min_score:
            return _refuse(
                "A:retrieval",
                f"best confidence {confidence:.3f} < min_score {self.settings.min_score}",
            )

        context, used = build_context(hits, self.settings.max_context_chars)
        prompt = ANSWER_PROMPT.format(
            title=video_title or video_id, context=context,
            question=question, sentinel=REFUSAL_SENTINEL,
        )
        raw = self.llm.complete(ANSWER_SYSTEM, prompt, temperature=self.settings.temperature)

        # ---- Layer B -----------------------------------------------------
        stripped = raw.strip()
        if not stripped or REFUSAL_SENTINEL in stripped:
            return _refuse("B:sentinel", "model reported insufficient context")

        # ---- Layer C -----------------------------------------------------
        text, citations, invalid = build_citations(stripped, used, video_id)
        if self.settings.require_valid_citations and not citations:
            detail = (
                f"answer cited only invalid excerpts {invalid}"
                if invalid else "answer contained no citations"
            )
            return _refuse("C:citations", detail)
        if invalid:
            self.last_trace = RefusalTrace("C:partial", f"dropped invalid markers {invalid}")

        return Answer(
            text=text, citations=citations, refused=False,
            rewritten_query=rewritten_query, retrieved=tuple(used),
            retriever=retriever, latency_ms=(time.perf_counter() - started) * 1000,
        )


def to_turns(question: str, answer: Answer) -> tuple[Turn, Turn]:
    return (
        Turn(role="user", content=question),
        Turn(role="assistant", content=answer.text, citations=answer.citations),
    )