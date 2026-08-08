# Technical Walkthrough

How yt-chat works, and why it is built this way. For results, see
[`evaluation.md`](evaluation.md); for usage, see the [README](../README.md).

**Reading order:** `models.py` → `pipeline.py` → whichever subsystem you care about. Those two
files carry the invariant and the orchestration; everything else is a leaf.

---

## Contents

1. [The invariant everything rests on](#1-the-invariant-everything-rests-on)
2. [Data pipeline](#2-data-pipeline)
3. [Embedding strategy](#3-embedding-strategy)
4. [Storage and caching](#4-storage-and-caching)
5. [Retrieval](#5-retrieval)
6. [Generation and refusal](#6-generation-and-refusal)
7. [Evaluation methodology](#7-evaluation-methodology)
8. [Testing approach](#8-testing-approach)
9. [Extending it](#9-extending-it)

---

## 1. The invariant everything rests on

> Every text-bearing object carries `(start_s, end_s)` in seconds from video start, plus the
> range of source caption segments it came from.

Stated in `models.py`, enforced from ingestion to rendered citation. Get this right and
citations are trivially correct; get it wrong and every timestamp is an approximation nobody
can audit.

The usual failure is at chunking. Captions arrive as 2–8 word cues with their own timings.
Most pipelines concatenate them into one string, chunk the string, then map back by matching
text or by taking the timestamp of whichever cue the chunk started in. Both are lossy, and both
fail *silently* — you get plausible timestamps that are off by seconds to tens of seconds, and
no test catches it because nothing asserts the mapping.

yt-chat instead builds an explicit character-to-time index. See §2.3.

---

## 2. Data pipeline

```
URL ──► video_id ──► RawTranscript ──► cleaned ──► Timeline ──► units ──► Chunk[]
        url.py       transcript.py     clean.py    timeline.py         chunking.py
```

### 2.1 Ingestion — `ingestion/`

`parse_video_id` handles `watch?v=`, `youtu.be/`, `/embed/`, `/shorts/`, `/live/`, `/v/`,
mobile/music/nocookie hosts, and bare 11-character IDs. Pure function, no network, exhaustively
tested.

`TranscriptProvider` is a `typing.Protocol`:

```python
def fetch(self, video_id: str, languages: Sequence[str]) -> RawTranscript: ...
```

`YouTubeTranscriptProvider` prefers **manually-created** captions, falls back to
**auto-generated**, then to a **translation** of any available track, recording which in
`TranscriptKind`. That matters downstream: caption kind determines which chunking path runs
(§2.4) and is a confound worth reporting in any evaluation.

`FallbackTranscriptProvider` chains providers, catching **only** `TranscriptUnavailableError`.
Any other exception propagates — so a genuine bug in a provider surfaces as a crash instead of
being silently swallowed and retried against a slower fallback.

Metadata degrades rather than fails: yt-dlp → oEmbed → placeholder title. A missing title must
never block answering questions.

### 2.2 Cleaning — `preprocessing/clean.py`

Three artifacts, all of which corrupt retrieval if left alone:

**Non-speech annotations.** `[Music]`, `[Applause]`, `(laughter)`. Pure noise in the index.

**Speaker markers.** `>>`, `>> JOHN SMITH:`. These *stack*, and the regex is `^`-anchored, so a
single `sub()` strips only the outermost — hence `_strip_speaker_prefixes` loops until stable.
A real bug, caught by a test asserting `">> SPEAKER: welcome back"` → `"welcome back"`.

**Rolling duplication.** Auto-captions repeat the previous cue's tail:

```
cue 1: "so today we are going to"
cue 2: "we are going to talk about neural"
cue 3: "talk about neural networks and how they"
```

Left in, "we are going to" appears three times, inflating BM25 term frequencies and producing
near-duplicate chunks that crowd out real results. `_strip_rolling_overlap` compares up to 12
trailing words of the previous cue against the leading words of the current one.

**Timing is never altered — only text.** Cues emptied by de-duplication are dropped, and their
time is absorbed by neighbours through interpolation.

### 2.3 The timeline — `preprocessing/timeline.py`

Cleaned segments are concatenated into one string **while recording each segment's character
range**:

```python
@dataclass(frozen=True, slots=True)
class SegmentSpan:
    seg_index: int
    char_start: int      # inclusive
    char_end: int        # exclusive
    t_start: float
    t_end: float
```

`time_at(char_pos)` binary-searches for the owning span and interpolates linearly:

```python
frac = (char_pos - span.char_start) / (span.char_end - span.char_start)
return span.t_start + frac * (span.t_end - span.t_start)
```

Three properties, each pinned by a test:

- **Monotonic** in `char_pos` — timestamps never travel backwards.
- **Exact at boundaries** — `time_at(span.char_start) == span.t_start`.
- **Clamped in gaps** — a position in the joiner between segments returns the *previous*
  segment's end, never the next one's start, so a chunk end never claims audio it does not
  cover.

Zero-duration cues (malformed tracks) get a 1 ms floor so interpolation cannot divide by zero.

This is what makes sub-cue timestamp accuracy possible. A chunk boundary landing mid-caption
gets a real interpolated time, not the cue's start.

### 2.4 Chunking — `preprocessing/chunking.py`

Text → **units** → **chunks**.

**Unit selection is adaptive**, because caption quality varies enormously:

```python
if punctuation_density(text) >= 0.004:   # terminal punctuation per character
    units = sentence_units(text)
else:
    units = word_window_units(text, words_per_window=28)
```

Human-written captions are punctuated, so sentences are the natural unit. Auto-generated
captions frequently contain **no terminal punctuation at all** — splitting those on `.` yields
one enormous unit spanning the whole video. All three benchmark videos use ASR captions, so the
word-window path is the one under test.

Units longer than `max_chars` (rambling ASR run-ons) are split on word boundaries so no chunk
can exceed budget.

Units are packed into chunks up to `max_chars`, with **overlap applied at unit granularity** —
expressed in characters but never cutting a sentence in half. A runt tail chunk below
`min_chars` is merged into its predecessor rather than emitted, since near-empty chunks skew
BM25's average document length.

Each chunk records `start_s`, `end_s` (from the timeline) and `seg_start`, `seg_end`
(provenance back to source cues).

**`max_chars = 700` is measured, not guessed** — see [`evaluation.md` §4](evaluation.md).

---

## 3. Embedding strategy

`Embedder` is a Protocol with a deliberately split interface:

```python
def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...
def encode_query(self, text: str) -> np.ndarray: ...
```

**Separate methods because asymmetric models need different prefixes.** BGE, E5 and GTE are
trained with distinct query and passage representations. Encoding a query without
`"Represent this sentence for searching relevant passages: "` costs several points of Recall@5
— a silent bug with no error message. `sentence_tf.py` detects the family from the model name
and applies the right prefixes.

The prefix state is folded into `model_id`, so changing it invalidates the embedding cache. A
cache key that does not cover everything affecting the vectors is worse than no cache.

All output is L2-normalised, so inner product equals cosine and every index can use plain IP
search.

**`HashingEmbedder`** is a deterministic hashing-trick bag-of-ngrams used by the entire test
suite. Same shapes, normalisation and cache semantics as a real embedder, but no weights to
download. It uses `blake2b` rather than Python's `hash()`, which is randomised per-process and
would silently break the cache across runs.

---

## 4. Storage and caching

SQLite with WAL, `foreign_keys = ON`, and an explicit schema in `database/schema.sql`. No ORM:
the queries are few and readability beats abstraction here.

### The three-stage fingerprint chain

```
videos (video_id)
  └── chunk_sets (video_id, chunker_fingerprint)
        └── embedding_sets (chunk_set_id, embedder model_id)
              └── vector index on disk (derived, disposable)
```

Each stage invalidates independently:

| Change | Re-downloads | Re-chunks | Re-embeds |
|---|---|---|---|
| Nothing | no | no | no |
| `max_chars` | **no** | yes | yes |
| Embedding model | **no** | **no** | yes |
| `--force` | yes | yes | yes |

This is what makes the chunk-size sweep affordable: 4 configs × 3 videos re-downloads nothing.
A naive `video_id`-only cache would have made that experiment cost 12 transcript fetches.

### The index is derived, not authoritative

Vectors live in SQLite as `float32` BLOBs. The FAISS file is a pure cache:
`VectorStore.load_or_build` tries the disk index, verifies its chunk IDs against the database,
and silently rebuilds from BLOBs on mismatch or corruption. Delete `~/.yt-chat/indexes/` and
nothing breaks.

Foreign keys are enforced, which caught a real bug: a conversation row pointing at a video that
did not exist, because a test fixture wrote to a different database than the one it later read.

---

## 5. Retrieval

`Retriever` is a Protocol returning `list[ScoredChunk]`. Four implementations compose freely.

### 5.1 Calibrated confidence — the load-bearing decision

Raw scores are incomparable: cosine sits in `[-1, 1]`, BM25 is unbounded and scales with query
length, RRF tops out near `1/61`. If `min_score` meant three different things depending on
`--retriever`, the refusal threshold would be meaningless and calibration impossible.

So every hit carries **two** numbers:

- `score` — the native ranking score, whatever determines order.
- `components["confidence"]` — calibrated to `[0, 1]` with consistent semantics.

| Retriever | Calibration |
|---|---|
| dense | `clip(cosine, 0, 1)` — negatives mean unrelated |
| sparse | `1 - exp(-s / tau)`, `tau = 8.0` — monotonic squash of unbounded BM25 |
| hybrid | dense-anchored (§5.4) |
| rerank | `sigmoid(logit)` — cross-encoders emit unbounded logits |

### 5.2 Dense

FAISS `IndexFlatIP` over normalised vectors, with an exact NumPy matmul fallback selected
automatically. **Flat, not HNSW**: a video yields hundreds to a few thousand chunks, where an
approximate index trades recall for a speedup nobody needs. Below ~10k vectors the NumPy path
is genuinely faster than FAISS because it skips wrapper overhead.

### 5.3 Sparse

Okapi BM25, implemented directly in ~70 lines. Written rather than imported so `k1`/`b` are
tunable from config and the default install stays light.

Two tokenizer decisions:

- **Compound-aware**: `self-attention` emits the compound *and* its parts, so a question
  phrased either way matches.
- **No stemming**, deliberately. It helps generic prose but conflates technical jargon —
  `transformer`/`transform` is a damaging merge. It remains a config knob for ablation.

Lucene-style IDF (`log(1 + (N - df + 0.5)/(df + 0.5))`) keeps values positive, so a term
appearing in every document contributes ~0 rather than a negative score.

### 5.4 Hybrid

**RRF by default** (`Σ 1/(k + rank)`): rank-based, so the incomparability of cosine and BM25
magnitudes never arises and there is no per-video alpha to tune. Weighted min-max fusion is
retained for the ablation, not because it is known to be better.

**Confidence is dense-anchored, not `max(dense, sparse)`:**

```python
confidence = dense_conf + 0.15 * sparse_conf * (1.0 - dense_conf)
```

The original `max()` was measurably wrong. BM25 sometimes scores highly on an *unanswerable*
question because two rare words coincide; taking the max let that false confidence override
dense's correctly low score, making hybrid a **worse** refusal signal than dense alone. The
current form keeps hybrid ≥ dense so agreement is still rewarded, while capping how much a
lexical coincidence can inflate it. Ordering is unaffected — that is still the fusion score.

### 5.5 Reranking

`RerankingRetriever` is a **decorator** over any base retriever: fetch `candidates` (default 30)
cheaply, rescore with a cross-encoder, return top `k`. Because it wraps rather than replaces, it
composes with dense, sparse and hybrid alike, and the eval harness can measure its contribution
to each independently.

It preserves `base_score` and `base_rank` in `components`, so a report can show how far the
reranker moved each chunk — the difference between "it helped" and "it helped *because*".

Off by default: it adds a model download and real latency, and should be enabled because
numbers justify it.

---

## 6. Generation and refusal

### 6.1 Query rewriting

*"Why is that useful?"* has no content words and retrieves nothing. `QueryRewriter` resolves it
against history into a standalone question.

A **heuristic gate** runs first: no history, or no referential markers and more than three
words, means skip the LLM call entirely. On a free tier that halves the per-turn API cost.

Rewriting failures are non-fatal — an exception or an over-long response falls back to the
original question. It is an optimisation, never a dependency.

### 6.2 Context construction

Retrieved chunks become numbered blocks:

```
[1] (12:43 - 13:20)
the speaker's words here...
```

**Excerpts are numbered by rank, so `[n]` maps directly onto `hits[n-1]`.** No fuzzy matching
between answer text and source — that is where most citation systems quietly go wrong. The
budget is enforced by `max_context_chars`, and the top-ranked excerpt is never dropped.

### 6.3 Three refusal layers

| Layer | Mechanism | Catches | Cost |
|---|---|---|---|
| **A** | `best_confidence(hits) < min_score` → refuse before generating | Out-of-domain questions | Zero API calls |
| **B** | Model emits `INSUFFICIENT_CONTEXT` | On-topic questions the video never answers | One call |
| **C** | Citations validated against supplied excerpts; all-invalid → refuse | Fluent prose with invented sources | One call |

They are independent on purpose: each catches what the others miss, and each can be disabled to
measure its individual contribution.

**Layer A is a cost optimisation, not the safety net.** That distinction drives the threshold
objective (§7.4). When `min_score` was inert at 0.28, correct-refusal was still 1.00 — Layers B
and C carried it.

Layer C catches the classic hallucination shape. `test_layer_c_refuses_an_uncited_answer` feeds
in *"The algorithm was invented in 1997 by Hochreiter and Schmidhuber"* — fluent, plausible,
entirely unsupported — and asserts it is rejected.

### 6.4 Citations

`build_citations` extracts `[n]` markers, validates them against the excerpts actually supplied,
strips invalid ones from the text, and reports them. An answer left with no valid citation is
downgraded to a refusal.

Rendering uses OSC-8 terminal hyperlinks, so timestamps are genuinely clickable in Windows
Terminal, iTerm2, GNOME Terminal and VS Code, degrading to plain URLs elsewhere.

### 6.5 LLM adapters

Gemini has its own adapter; Groq, OpenRouter and local Ollama share one, since all three speak
the OpenAI chat-completions dialect.

Two behaviours matter on free tiers:

- **Request spacing** from a per-provider RPM limit, plus exponential backoff with full jitter.
- **Permanent-condition detection.** `tokens per day`, `TPD`, and `limit: 0` are *not* retried.
  They match "429" and "rate limit", so without this the client burns its whole backoff
  schedule before failing anyway. Learned by watching it happen.

---

## 7. Evaluation methodology

### 7.1 Ground truth as time ranges

Covered in [`evaluation.md` §1](evaluation.md). The short version: chunk IDs would make every
benchmark obsolete the moment chunking changes, and the chunk-size sweep impossible.

### 7.2 Two relevance notions

```python
span_coverage(chunk, span)  # overlap / min(durations) — lenient; drives Recall/MRR/nDCG
span_iou(start, end, span)  # strict IoU — drives citation precision
```

Deliberately in tension. Reporting only one is how RAG projects claim wins they did not earn.

### 7.3 Span-level recall

Recall is **the fraction of gold spans covered**, not "was anything relevant retrieved". A
question whose answer spans two moments is only fully recalled when both are found.

### 7.4 Threshold calibration is free

Layer A depends only on retrieval confidence, so the entire refusal ROC costs **zero LLM
calls**. Follow-ups are excluded, since the sweep calls the retriever directly and no rewriting
happens — including them would count a measurement artifact as a gate failure.

The recommender maximises correct refusal **subject to a false-refusal budget**, not Youden J.
Youden J assumes symmetric costs; here a missed refusal costs one API call and is usually caught
downstream, while a false refusal denies the user an answer the system already retrieved.

### 7.5 The judge

A built-in provider-agnostic `LLMJudge` is the default, because RAGAS and DeepEval default to
OpenAI and need LangChain wrapper plumbing to point elsewhere. Both are available as optional
cross-checks — measuring how much three judges agree tells you how much to trust any of them.

**Scoring is combined into one call.** Four per-metric calls each re-sent the full context,
which exhausted a 100k tokens/day allowance partway through a three-video run. Combining cuts
judge tokens ~4×. `combined=False` restores per-metric scoring for checking whether combining
biases the scores.

**Refusals score 1.0 faithfulness** by construction — they assert nothing, so they cannot be
unfaithful. This is correct *and* it makes faithfulness gameable; see
[`evaluation.md` §6](evaluation.md).

---

## 8. Testing approach

```bash
pytest -q     # 195 tests, ~2 seconds, no network
```

### Offline by construction, not by mocking

Every external dependency sits behind a Protocol with a deterministic double:

| Real | Test double |
|---|---|
| `YouTubeTranscriptProvider` | `StaticTranscriptProvider` |
| `BestEffortMetadataProvider` | `StaticMetadataProvider` |
| `SentenceTransformerEmbedder` | `HashingEmbedder` |
| `GeminiLLM` / `OpenAICompatibleLLM` | `ScriptedLLM` |
| `CrossEncoderScorer` | `HashingScorer` |

No `patch("requests.get")` anywhere. This is a design property — the seams exist because the
architecture needs them, and testability falls out.

`ScriptedLLM` returns a refusal when its queue empties, so an unexpected extra call fails loudly
rather than hanging on a real request.

### Tests worth reading

- **`test_chunking.py`** — the invariant. Timestamps monotonic, bounded by video duration, spans
  non-inverted, coverage ≥ 90%, provenance recorded, output deterministic.
- **`test_pipeline.py`** — the cache chain. Asserts a second run refetches nothing, that
  changing the chunker reuses the transcript, and that changing the embedder reuses chunks.
- **`test_answerer.py`** — refusal. Asserts Layer A refuses **without making an API call**
  (`assert llm.calls == []`), and that fluent uncited prose is rejected.
- **`test_eval_metrics.py`** — the metrics themselves, including nDCG ≤ 1 with overlapping hits,
  the regression test for a real bug.

### The harness needs tests too

Two of the five defects in [`evaluation.md` §8](evaluation.md) were **in the measurement code**:
searching the wrong depth, and nDCG exceeding 1.0. An evaluation harness is a component, not a
trusted oracle.

---

## 9. Extending it

**A new retriever:** implement `search(query, k) -> list[ScoredChunk]`, emit
`components["confidence"]` in `[0, 1]`, register in `retrieval/factory.py`. It works everywhere
immediately, including the eval harness.

**A new LLM provider:** if it speaks OpenAI chat-completions, add a base URL to
`OPENAI_COMPATIBLE_BASE_URLS`. Otherwise implement `complete(system, prompt, temperature)`.

**A new embedder:** implement the Protocol and give it a `model_id` covering *everything* that
changes the vectors, or the cache will serve stale results.

**A new transcript source** (Whisper, uploaded files, another platform): implement
`fetch(video_id, languages) -> RawTranscript` and add it to the fallback chain. Nothing
downstream changes — this is how the Whisper fallback is designed to land.

**A new benchmark:** copy a YAML, use `yt-chat eval draft` to propose spans, then **verify them
by hand**. Retrieval proposing its own ground truth is circular and quietly corrupts every
number downstream.
