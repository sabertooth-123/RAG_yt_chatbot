"""Conversational query rewriting.

"Why is that useful?" retrieves nothing on its own — it has no content words.
Rewriting it against history into "Why is self-attention useful?" is the single
highest-leverage component for multi-turn quality (experiment #4 measures it).

A heuristic gate skips the LLM call when the question is already standalone,
which on a free tier is the difference between one API call per turn and two.
"""

from __future__ import annotations

import re
from typing import Sequence

from ytchat.generation.llm import LLM
from ytchat.generation.prompts import REWRITE_PROMPT, REWRITE_SYSTEM
from ytchat.models import Turn

_REFERENTIAL = re.compile(
    r"\b(it|its|it's|that|this|those|these|they|them|their|he|she|his|her|"
    r"the same|the former|the latter|there|such)\b",
    re.IGNORECASE,
)
_CONTINUATION = re.compile(r"^\s*(and|but|so|then|also|what about|how about|why|ok|okay)\b",
                           re.IGNORECASE)
_MAX_REWRITE_CHARS = 300


def needs_rewrite(question: str, history: Sequence[Turn]) -> bool:
    if not history:
        return False
    q = question.strip()
    if len(q.split()) <= 3:
        return True                          # "why?", "and then?", "the second one"
    return bool(_REFERENTIAL.search(q) or _CONTINUATION.match(q))


def _format_history(history: Sequence[Turn], max_turns: int) -> str:
    recent = list(history)[-max_turns:]
    lines = []
    for turn in recent:
        speaker = "User" if turn.role == "user" else "Assistant"
        content = " ".join(turn.content.split())
        if len(content) > 240:
            content = content[:239] + "…"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


class QueryRewriter:
    def __init__(self, llm: LLM, max_turns: int = 4, enabled: bool = True) -> None:
        self.llm = llm
        self.max_turns = max_turns
        self.enabled = enabled

    def rewrite(self, question: str, history: Sequence[Turn]) -> tuple[str, bool]:
        """Returns ``(query_to_retrieve_with, was_rewritten)``."""
        if not self.enabled or not needs_rewrite(question, history):
            return question, False
        try:
            raw = self.llm.complete(
                REWRITE_SYSTEM,
                REWRITE_PROMPT.format(
                    history=_format_history(history, self.max_turns), question=question
                ),
                temperature=0.0,
            )
        except Exception:
            return question, False           # rewriting is an optimisation, never a hard failure

        candidate = raw.strip().strip('"').split("\n")[0].strip()
        # Guard against a model that ignores instructions and answers instead.
        if not candidate or len(candidate) > _MAX_REWRITE_CHARS:
            return question, False
        return candidate, candidate.lower() != question.strip().lower()