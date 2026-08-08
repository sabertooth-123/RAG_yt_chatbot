# yt-chat

Ask questions about any YouTube video and get answers grounded **only** in what the speaker
actually said — with clickable timestamps proving it.

```
$ yt-chat https://youtu.be/zjkBMFhNj_g

  [1hr Talk] Intro to Large Language Models
  Andrej Karpathy · 59:47 · 94 chunks · generated captions
  retriever: dense · llm: groq:llama-3.3-70b-versatile

› What two files make up a large language model?

  A large language model is just two files: a parameters file holding the
  weights — 140 GB for Llama 2 70B, at two bytes per parameter — and a run
  file containing the code that executes them, which can be about 500 lines
  of C with no other dependencies [1][2].

  [1] 1:34   https://youtube.com/watch?v=zjkBMFhNj_g&t=94s
  [2] 2:15   https://youtube.com/watch?v=zjkBMFhNj_g&t=135s

› What is the attention mechanism?

  I cannot find this information in the video.
```

That second answer is the point. The video never discusses attention, so the system says so
instead of answering from general knowledge. **Correct refusal rate is 1.00** across the
benchmark's verified-unanswerable questions.

---

## Contents

- [What it does](#what-it-does)
- [Install](#install)
- [Usage](#usage)
- [Architecture](#architecture)
- [Design decisions](#design-decisions)
- [Evaluation](#evaluation)
- [What measurement changed](#what-measurement-changed)
- [Testing](#testing)
- [Limitations](#limitations)

---

## What it does

| | |
|---|---|
| **Grounded answers** | Uses only the video's transcript. Three independent refusal checks, not one prompt instruction. |
| **Clickable citations** | Every claim carries a `[n]` marker resolving to a `&t=` URL and an OSC-8 terminal hyperlink. |
| **Follow-up questions** | "What is a jailbreak?" → "Why does that work?" is rewritten into a standalone query before retrieval. |
| **Four retrieval strategies** | Dense (FAISS), sparse (BM25), hybrid (RRF or weighted), plus optional cross-encoder reranking. |
| **Processed once** | Three-stage cache. Change chunk size and the transcript is reused; change the embedding model and the chunks are reused. |
| **Measures itself** | Recall@k, Precision@k, MRR, nDCG, timestamp-IoU citation precision, LLM-judged faithfulness, and refusal-threshold calibration. |
| **Free to run** | Groq, Gemini, OpenRouter, or local Ollama. No paid API required. |

## Install

```bash
git clone <your-repo-url> && cd yt-chat
python -m venv .venv && .venv\Scripts\Activate.ps1     # Windows
pip install -e ".[embeddings,llm,metadata,dev]"
```

For GPU embeddings, install the CUDA build of torch **before** the editable install:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Get a free API key ([console.groq.com/keys](https://console.groq.com/keys)) and create `.env`:

```
YTCHAT_LLM_PROVIDER=groq
YTCHAT_GROQ_API_KEY=gsk_...
YTCHAT_MIN_SCORE=0.60
```

Or skip the key entirely and run a local model with `YTCHAT_LLM_PROVIDER=ollama`.

## Usage

```bash
yt-chat https://youtu.be/VIDEO_ID              # interactive session
yt-chat ask https://youtu.be/VIDEO_ID -Q "..."  # one-shot; exit 1 if refused
yt-chat cache stats                             # what's stored locally
yt-chat eval run benchmarks/karpathy_llm_intro.yaml
```

In-session commands: `/ask` `/sources` `/history` `/change-retriever` `/compare` `/video`
`/debug` `/clear` `/help` `/exit`.

`/debug` is worth knowing — it shows which refusal layer fired, the retrieved chunks, and
their calibrated confidences.

Every setting is an env var (`YTCHAT_MAX_CHARS`, `YTCHAT_RETRIEVER`, `YTCHAT_ENABLE_RERANK`,
…), so experiments never require code changes.

## Architecture

```
   YouTube URL
        │
        ▼
   ┌─────────────────────────────────────────┐
   │ INGESTION      url parsing · captions   │   manual > generated > translated
   │                metadata (yt-dlp/oEmbed) │
   └────────────────────┬────────────────────┘
                        ▼
   ┌─────────────────────────────────────────┐
   │ PREPROCESSING  clean · timeline · chunk │   ← the timestamp invariant
   └────────────────────┬────────────────────┘
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
   ┌────────────┐ ┌───────────┐ ┌──────────────┐
   │ EMBEDDINGS │ │   BM25    │ │ SQLite cache │  videos · chunks · vectors
   │ bge-small  │ │  index    │ │ (3-stage FP) │  conversations
   └──────┬─────┘ └─────┬─────┘ └──────────────┘
          └──────┬──────┘
                 ▼
   ┌─────────────────────────────────────────┐
   │ RETRIEVAL   dense │ sparse │ hybrid      │   all emit calibrated
   │             + optional cross-encoder     │   confidence in [0,1]
   └────────────────────┬────────────────────┘
                        ▼
   ┌─────────────────────────────────────────┐
   │ GENERATION  rewrite → context → answer  │   3 refusal layers
   │             → validate citations         │
   └────────────────────┬────────────────────┘
                        ▼
              CLI  (Typer + Rich REPL)

   EVALUATION runs beside all of it: time-range ground truth,
   retrieval metrics, LLM judge, threshold calibration.
```

Module layout mirrors this: `ingestion/ preprocessing/ embeddings/ retrieval/ generation/
database/ evaluation/ cli/`, tied together by `pipeline.py`.

## Design decisions

### The timestamp invariant

Every text-bearing object carries `(start_s, end_s)` and its source segment range. This is
enforced from ingestion through to the rendered citation.

Most YouTube-RAG projects concatenate captions into one string, chunk it, then guess which
timestamp a chunk came from. Instead, cleaned segments are concatenated **while recording
each segment's character range**, so any character offset interpolates to a wall-clock time.
A chunk boundary landing mid-caption still gets an accurate timestamp.

### Punctuation-adaptive chunking

Human-written captions are punctuated, so units are sentences. Auto-generated captions often
have no punctuation at all — splitting those on `.` yields one enormous unit. The chunker
measures punctuation density and falls back to fixed word windows. All three benchmark videos
use ASR captions, so this path is the one under test.

### Three refusal layers, not one prompt

| Layer | Mechanism | Catches |
|---|---|---|
| **A** | Retrieval confidence below `min_score` → refuse **before** calling the LLM | Out-of-domain questions, cheaply |
| **B** | Model emits `INSUFFICIENT_CONTEXT` | On-topic questions the video never answers |
| **C** | Answers citing non-existent excerpts are discarded | Confident prose with invented sources |

Each is independently toggleable so its individual contribution can be measured.

### Calibrated confidence across retrievers

Cosine sits in `[-1,1]`, BM25 is unbounded, RRF tops out near `1/61`. If `min_score` meant
three different things depending on `--retriever`, the refusal threshold would be meaningless.
Every retriever therefore emits both a native ranking score and a calibrated
`components["confidence"]` in `[0,1]`.

### Stage-scoped cache fingerprints

The cache key is a chain: `video → chunk_set(chunker hash) → embedding_set(model id)`. A
chunk-size sweep across 4 configs × 3 videos re-downloads nothing and re-embeds only what
changed. That is what makes the experiments below affordable.

### Time-range ground truth

Benchmarks record correct answers as **time ranges**, never chunk IDs — so a benchmark stays
valid after any change to chunking, embeddings, or retrieval. Two relevance notions follow
from it, deliberately in tension:

- **Retrieval relevance** — lenient overlap. Does the chunk contain the answer? Big chunks win.
- **Citation precision** — strict IoU. Does the timestamp point *at* the answer? Big chunks lose.

Reporting only one of them is how RAG projects claim wins they did not earn.

## Evaluation

Three videos, 35 questions, 24 answerable, 11 verified-unanswerable.

| Video | Length | Chunks |
|---|---|---|
| Intro to Large Language Models | 1:00 | 94 |
| Let's build the GPT Tokenizer | 2:13 | 185 |
| Deep Dive into LLMs like ChatGPT | 3:31 | 311 |

Unanswerable cases are verified by absence — the transcripts contain zero occurrences of the
relevant terms. Three questions are **answerable in one benchmark and verified-absent in
another**, a cross-video control that catches a system answering from world knowledge instead
of from the video in front of it.

### Retrieval

Recall@5 at the default `max_chars=700`. Gold spans were assigned by reading each full
transcript independently of retriever output.

| Video | dense | sparse | hybrid |
|---|---|---|---|
| Intro (140 chunks) | 0.62 | 0.50 | **0.81** |
| Tokenizer (271) | **1.00** | 0.56 | 0.83 |
| Deep Dive (467) | **0.86** | 0.79 | 0.79 |
| **Mean** | **0.83** | 0.62 | 0.81 |

### Retriever ranking is not stable across chunk sizes

The same benchmark at `max_chars=900` gives dense 0.78, sparse 0.67, **hybrid 0.81** — hybrid
ahead. At 700 it is dense 0.83, hybrid 0.81 — dense ahead. **The ranking flips on a chunking
parameter**, and the gap between the two is smaller than the effect of that parameter.

So the honest conclusion is not "hybrid wins" or "dense wins." It is that with 24 answerable
questions from one speaker, **no retriever dominates, and any published ranking is conditional
on a chunking config most papers do not report.** Sparse is consistently last; that much holds.

### By question type — this *does* hold

| Type | n | dense | sparse | hybrid |
|---|---|---|---|---|
| Lexical | 12 | **1.00** | 0.71 | 0.88 |
| Semantic | 11 | 0.73 | 0.55 | **0.82** |

This pattern is stable at both 700 and 900 chars, which makes it the more trustworthy result.

**Dense wins on lexical questions — the category BM25 is supposed to own.** The cause is
specific to this medium: ASR mangles technical terms. The transcripts say *"Tik token"*,
*"sentence piece"*, *"B PA en coding"*. A user typing `tiktoken` or `BPE` gets no keyword
match at all, while the embedding model still lands nearby. **Auto-captions erase BM25's home
advantage**, which is why sparse never leads on this corpus.

Hybrid's edge is on semantic paraphrase questions, where blending a weak lexical signal with
dense retrieval breaks ties that pure similarity gets wrong.

An earlier version of this README claimed dense degrades as the index grows, based on the
900-char run. It does not reproduce at 700. That claim was an artifact of one configuration
and has been removed rather than quietly kept.

### Chunk size: recall and citation precision peak together

Sweeping `max_chars` costs nothing — retrieval metrics only, and the staged cache re-chunks
without re-downloading anything. Pooled across all three videos:

| `max_chars` | mean chunk | Recall@5 | Citation IoU |
|---|---|---|---|
| 400 | 18.0s | 0.75 | 0.076 |
| **700** | **36.4s** | **0.83** | **0.153** |
| 1000 | 54.6s | 0.79 | 0.148 |
| 1500 | 85.3s | 0.82 | 0.129 |

**Citation precision doubles from 400 to 700 chars, then declines.** It is an inverted-U, not
the monotonic tradeoff usually assumed. The geometry explains it: IoU is maximised when chunk
duration matches the span in which answers actually appear — 30–90 seconds in this corpus. A
20-second chunk sits *inside* the answer and scores `20/90`; an 85-second chunk *swallows* it
and scores `90/300`. Both extremes lose.

Recall peaks at the same point, so at this operating point there is no tradeoff to manage —
just a default worth measuring. `max_chars` is now 700; it was 900, slightly past the peak.

The tradeoff would reappear with much finer gold spans. The transferable rule is **match chunk
duration to the granularity at which answers occur in your content**, not "smaller chunks give
better citations."

### Refusal calibration

Sweeping `min_score` costs **zero LLM calls**, because Layer A depends only on retrieval
confidence.

| Video | Recommended | Correct refusal | False refusal |
|---|---|---|---|
| Intro | 0.60 | 50% | 14% |
| Tokenizer | 0.70 | 100% | 11% |
| Deep Dive | 0.65 | 75% | 0% |

`min_score = 0.65` is the setting that generalises. Note that **no threshold below 0.55 does
anything at all** — see below.

These recommendations shifted upward when `max_chars` moved from 900 to 700, because smaller
chunks are more topically concentrated and score higher. **A confidence threshold is not
portable across chunking configurations** — it has to be recalibrated whenever chunking or the
embedding model changes. The calibration sweep is free, so this is cheap to honour.

The recommendation optimises correct refusal subject to a false-refusal budget, deliberately
*not* Youden J. Layer A is a cost optimisation; Layers B and C are the safety net. A missed
refusal costs one API call and is usually caught downstream, while a false refusal denies the
user an answer the system had already retrieved. Under Youden J the tool recommended 0.65 on
the 900-char run, which refused questions whose `recall@k` was 1.00 — the right passage was
found and then discarded.

### Generation

All 9 jobs complete — 3 benchmarks × 3 retrievers, 72 answer attempts, 33 refusal
opportunities, judged by `llama-3.3-70b-versatile`.

| Retriever | Recall@5 | CitePrec | Faithful | AnsRel | Correct | FalseRefuse | CorrectRefuse |
|---|---|---|---|---|---|---|---|
| **dense** | **0.85** | **0.29** | 0.77 | **0.69** | **0.62** | **0.08** | **1.00** |
| hybrid | 0.83 | 0.26 | 0.83 | 0.68 | 0.54 | 0.17 | 0.91 |
| sparse | 0.60 | 0.19 | **0.86** | 0.51 | 0.56 | 0.38 | **1.00** |

**32 of 33 unanswerable questions were correctly refused (97%).** Dense refused all 11 while
wrongly refusing only 8% of answerable ones — the best trade of the three, and the shipped
default.

**Sparse wins faithfulness and nothing else.** It posts the highest faithfulness (0.86) with
the worst answer relevance (0.51) and the worst false-refusal rate (0.38), because a refusal
asserts nothing and therefore scores 1.0 by construction. *Trustworthy because it barely
answers.*

**The single miss is more interesting than the 32 successes.** A convolutional-network question
asked of the tokenizer video passed all three refusal layers — and the judge scored the answer
**1.0 faithful**, correctly, because the model stretched genuinely-retrieved passages into an
answer and cited them properly. Nothing was fabricated. Only `context_relevance` caught it, at
0.20. **Faithfulness is structurally blind to off-topic-but-grounded answers**: it asks whether
a claim is supported, never whether the question should have been answered.

Five false refusals had `recall@k = 1.00` — the correct passage retrieved, then discarded by
Layer A at `min_score=0.65`, which per-video calibration shows is too aggressive for one of the
three videos. It was deliberately left unchanged mid-experiment so the nine runs stay
comparable.

Full breakdown, including per-job numbers and the failure list, in
[`docs/evaluation.md`](docs/evaluation.md).

## What measurement changed

Five defects that reading the code would never have surfaced. Each was found by the
evaluation harness, and each changed the system.

**1. The refusal threshold was inert.** `min_score` was set to 0.28 by guesswork. Measurement
showed bge-small never scores below ~0.45 on ordinary English prose — even for entirely
unrelated passages — so Layer A never once fired. Unrelated chunks about *hallucination*,
*System 1 thinking*, and a *Sephora discount* all scored 0.49–0.54 against "What is
attention?". **Cosine similarity is a good ranker and a poor classifier.**

**2. Hybrid fusion made refusals worse.** Hybrid took `max(dense_conf, sparse_conf)`. BM25
sometimes scores highly on an unanswerable question because a couple of rare words coincide —
that false confidence overrode dense's correctly low score. Switching to a dense-anchored
formula doubled correct-refusal rate at the same threshold.

**3. The harness measured the wrong depth.** Evaluation searched 30 candidates while
production's Layer A sees 5, so reported confidences were systematically higher than anything
the running system encounters. Caught only because two tools disagreed with each other.

**4. nDCG exceeded 1.0.** Normalising by the number of gold spans rather than the number of
relevant retrieved chunks let DCG accumulate more terms than IDCG had, yielding 1.01 and 1.09.

**5. Faithfulness is gameable by refusing.** Sparse scored a *perfect* 1.00 faithfulness — by
refusing half the questions. Refusals assert nothing, so they score 1.0 by construction. A
system that refuses everything looks flawless. **Never report faithfulness without a
false-refusal rate beside it.**

## Testing

```bash
pytest -q     # 195 tests, ~2 seconds
```

No network, no API keys, no model downloads. Every external dependency sits behind a
`typing.Protocol` with a deterministic offline double — `StaticTranscriptProvider`,
`HashingEmbedder`, `ScriptedLLM`, `HashingScorer`. That is a design property, not a mocking
convention.

Tests worth reading: `test_chunking.py` proves timestamps survive cleaning, concatenation,
splitting, serialisation and reload; `test_answerer.py` asserts Layer A refuses *without
making an API call*, and that a fluent, plausible, entirely uncited answer is rejected.

## Limitations

- **No captions, no answers.** Videos with captions disabled (~15% of YouTube) fail cleanly.
  A local Whisper fallback is designed and wired but not yet implemented.
- **English only.** bge-small is English; other languages work via translated caption tracks
  with degraded retrieval.
- **Small n.** 24 answerable questions across 3 videos, all from one speaker. Differences
  below ~0.15 are one or two questions changing sides. Treat every table here as directional.
- **Ground truth from transcripts, not viewing.** Gold spans were assigned by reading the
  full transcripts. For talks where all information is spoken this is equivalent evidence,
  but it would not hold for visually-dependent content.
- **Judged metrics incomplete.** See the queue note above.

## License

MIT
