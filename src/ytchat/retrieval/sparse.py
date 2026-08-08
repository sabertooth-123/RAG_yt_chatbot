"""Okapi BM25, implemented directly.

Written rather than imported for three reasons: it is ~70 lines, it removes a
dependency from the default install, and ``k1``/``b`` need to be tunable from
config for the retrieval experiments.

Tokenizer note: hyphenated technical terms emit *both* the compound and its
parts ("self-attention" → "self-attention", "self", "attention"), so a question
phrased either way matches.  Stemming is deliberately off: it helps generic
prose but conflates technical jargon ("training"/"trained"/"train" is fine,
"transformer"/"transform" is not).  It stays a config knob for the ablation.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from ytchat.models import Chunk, ScoredChunk
from ytchat.retrieval.base import CONFIDENCE_KEY, bm25_confidence

_WORD = re.compile(r"[a-z0-9]+(?:[''\-][a-z0-9]+)*")

STOPWORDS = frozenset("""
a an and are as at be but by for from has have he her his i if in into is it its
of on or she that the their them they this to was were what when where which who
will with you your do does did not no so such than then there these those we us
""".split())


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens: list[str] = []
    for match in _WORD.findall(text.lower()):
        if "-" in match or "'" in match:
            tokens.append(match)
            tokens.extend(p for p in re.split(r"[''\-]", match) if p)
        else:
            tokens.append(match)
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


class BM25Index:
    def __init__(
        self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75,
        drop_stopwords: bool = True,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.drop_stopwords = drop_stopwords
        self.doc_tokens = [tokenize(d, drop_stopwords) for d in documents]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.n_docs = len(self.doc_tokens)
        self.avg_len = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

        self.term_freqs: list[Counter[str]] = [Counter(t) for t in self.doc_tokens]
        df: Counter[str] = Counter()
        for tf in self.term_freqs:
            df.update(tf.keys())
        # Lucene-style IDF: always positive, so a term in every document
        # contributes ~0 instead of a negative score.
        self.idf = {
            term: math.log(1.0 + (self.n_docs - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def score_all(self, query: str) -> list[float]:
        q_terms = tokenize(query, self.drop_stopwords)
        scores = [0.0] * self.n_docs
        if not q_terms or self.n_docs == 0:
            return scores
        for term in q_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.term_freqs):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (
                    1 - self.b + self.b * (self.doc_len[i] / self.avg_len or 1.0)
                )
                scores[i] += idf * (f * (self.k1 + 1) / denom)
        return scores


class SparseRetriever:
    def __init__(
        self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75,
        drop_stopwords: bool = True, tau: float = 8.0,
    ) -> None:
        self.chunks = list(chunks)
        self.tau = tau
        self.index = BM25Index([c.text for c in self.chunks], k1=k1, b=b,
                               drop_stopwords=drop_stopwords)

    @property
    def name(self) -> str:
        return "sparse"

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        if not query.strip() or not self.chunks:
            return []
        scores = self.index.score_all(query)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        out: list[ScoredChunk] = []
        for rank, i in enumerate(order[:k]):
            if scores[i] <= 0.0:
                break  # a zero BM25 score means no query term matched at all
            out.append(
                ScoredChunk(
                    chunk=self.chunks[i], score=float(scores[i]), rank=rank,
                    retriever="sparse",
                    components={"bm25": float(scores[i]),
                                CONFIDENCE_KEY: bm25_confidence(scores[i], self.tau)},
                )
            )
        return out