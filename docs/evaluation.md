# Evaluation

Everything here is reproducible from the repository:

```bash
yt-chat eval run benchmarks/karpathy_llm_intro.yaml --no-judge
python scripts/sweep_chunk_size.py
python scripts/run_eval_queue.py
```

**Configuration:** `max_chars=700`, `overlap_chars=120`, `top_k=5`, `candidate_k=30`,
`embedding_model=BAAI/bge-small-en-v1.5`, `min_score=0.65`.

---

## 1. The benchmark

Three videos, 35 questions, **24 answerable** and **11 verified-unanswerable**.

| Benchmark | Video | Length | Chunks @700 |
|---|---|---|---|
| `karpathy_llm_intro` | [1hr Talk] Intro to Large Language Models | 1:00 | 140 |
| `karpathy_tokenizer` | Let's build the GPT Tokenizer | 2:13 | 271 |
| `karpathy_deepdive` | Deep Dive into LLMs like ChatGPT | 3:31 | 467 |

### Ground truth is time ranges, not chunk IDs

A benchmark that says "the answer is in chunk 47" dies the moment chunk size changes. One that
says "the answer is at 12:43–13:20" survives every re-chunking, every embedding model, and
every retriever. That single decision is what makes the sweep in §4 possible at all.

Two relevance notions follow, deliberately in tension:

- **Retrieval relevance** — `overlap / min(chunk_duration, span_duration) >= 0.30`. Lenient:
  does the chunk *contain* the answer? Large chunks win.
- **Citation precision** — strict intersection-over-union. Does the timestamp point *at* the
  answer? Large chunks lose.

### How gold spans were assigned

Each full transcript was read end to end and answer locations identified independently. **Gold
spans were not taken from retriever output** — a system that proposes its own ground truth
grades its own homework, and every downstream number becomes meaningless.

Spans are kept to 30–90 seconds. A five-minute span makes recall trivially easy and drives IoU
to zero.

*Caveat:* spans derive from transcript text, not from watching. For talks where all information
is spoken this is equivalent evidence; it would not hold for visually-dependent content.

### Unanswerable questions are verified by absence

Each was confirmed by searching the full transcript. The intro talk contains zero occurrences
of *attention*, *backpropagation*, or *positional encoding*; the deep dive contains zero of
*jailbreak*, *prompt injection*, *operating system*, or *sentence piece*.

**Cross-video controls:** three questions are answerable in one benchmark and verified-absent
in another. *"What is a jailbreak attack?"* is answerable from the intro talk and absent from
the deep dive. A system answering from world knowledge rather than from the video in front of
it fails these, and the failure is unambiguous.

---

## 2. Retrieval

Recall@5, `max_chars=700`.

| Video | dense | sparse | hybrid |
|---|---|---|---|
| Intro (140 chunks) | 0.62 | 0.50 | **0.81** |
| Tokenizer (271) | **1.00** | 0.56 | 0.83 |
| Deep Dive (467) | **0.86** | 0.79 | 0.79 |
| **Mean** | **0.83** | 0.62 | 0.81 |

Full metrics:

| Retriever | Recall@5 | Precision@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| dense | **0.83** | 0.33 | **0.75** | **0.77** |
| sparse | 0.62 | 0.32 | 0.61 | 0.61 |
| hybrid | 0.81 | **0.34** | 0.69 | 0.70 |

### The ranking is not stable across chunk sizes

At `max_chars=900` the same benchmark gives dense 0.78, sparse 0.67, **hybrid 0.81**. At 700 it
gives **dense 0.83**, hybrid 0.81.

**The ranking flips on a chunking parameter, and the dense–hybrid gap is smaller than the
effect of that parameter.** With 24 answerable questions from a single speaker, no retriever
dominates. Any published "hybrid beats dense" claim is conditional on a chunking configuration
that is usually not reported.

What does hold: **sparse is consistently last**, at both chunk sizes and on every video.

---

## 3. Where each retriever wins

| Question type | n | dense | sparse | hybrid |
|---|---|---|---|---|
| Lexical | 12 | **1.00** | 0.71 | 0.88 |
| Semantic | 11 | 0.73 | 0.55 | **0.82** |

This pattern reproduces at both 700 and 900 chars, which makes it more trustworthy than the
aggregate ranking.

**Dense wins lexical questions — the category BM25 is supposed to own.** The cause is specific
to this medium. ASR mangles technical terminology: the transcripts render *tiktoken* as
*"Tik token"*, *SentencePiece* as *"sentence piece"*, *BPE* as *"B PA en coding"*. A user
typing the correct term gets **no keyword match at all**, while the embedding model still lands
in the right neighbourhood.

**Auto-generated captions erase BM25's home advantage.** This is a general result for RAG over
ASR content, not a quirk of these three videos, and it argues against assuming keyword search
will handle jargon in any speech-derived corpus.

Hybrid's edge is on semantic paraphrase, where a weak lexical signal breaks ties that pure
similarity gets wrong.

---

## 4. Chunk size: recall and citation precision peak together

Pooled across all three videos, `overlap` held at 17% of `max_chars`. Free to run — retrieval
metrics only.

| `max_chars` | mean chunk | Recall@5 | Citation IoU |
|---|---|---|---|
| 400 | 18.0s | 0.75 | 0.076 |
| **700** | **36.4s** | **0.83** | **0.153** |
| 1000 | 54.6s | 0.79 | 0.148 |
| 1500 | 85.3s | 0.82 | 0.129 |

**Citation precision doubles from 400 to 700, then declines.** It is an inverted-U, not the
monotonic tradeoff usually assumed.

The geometry explains it. IoU peaks when chunk duration matches the span in which answers
actually appear — 30–90 seconds here. A 20-second chunk sits *inside* the answer and scores
`20/90`; an 85-second chunk *swallows* it and scores `90/300`. Both extremes lose.

Recall peaks at the same point, so at this operating point **there is no tradeoff to manage —
only a default worth measuring.** The default moved from 900 to 700 as a result.

*Caveat:* the IoU proxy averages over all five retrieved chunks, including irrelevant ones
scoring zero, so it partly reflects retrieval quality. But recall differs by 10% between 400
and 700 while IoU differs by 100% — geometry dominates.

The transferable rule is **match chunk duration to the granularity at which answers occur in
your content**, not "smaller chunks give better citations."

---

## 5. Refusal calibration

Sweeping `min_score` costs **zero LLM calls**: Layer A depends only on retrieval confidence.
Follow-up questions are excluded, since the sweep calls the retriever directly and no query
rewriting happens.

| Video | Recommended | Correct refusal | False refusal |
|---|---|---|---|
| Intro | 0.60 | 50% | 14% |
| Tokenizer | 0.70 | 100% | 11% |
| Deep Dive | 0.65 | 75% | 0% |

`min_score = 0.65` generalises; it is the shipped default.

### Thresholds are not portable

These recommendations sat at 0.60/0.65/0.60 when `max_chars` was 900. Smaller chunks are more
topically concentrated and score higher, so the whole distribution shifted.

**A confidence threshold must be recalibrated whenever chunking or the embedding model
changes.** The sweep is free, so there is no excuse not to.

### The objective is not Youden J

The recommender maximises correct refusal *subject to a false-refusal budget* (default 15%).
Youden J assumes both error types cost the same. They do not: Layer A is a cost optimisation
while Layers B and C are the actual safety net. A missed refusal costs one API call and is
usually caught downstream; a false refusal denies the user an answer the system had already
found.

Under Youden J the tool recommended 0.65 on the 900-char run, which refused cases with
`recall@k = 1.00` — the correct passage retrieved, then discarded by the gate.

---

## 6. Generation

Complete: all 9 jobs (3 benchmarks × 3 retrievers), `max_chars=700`, `min_score=0.65`, judged
by `llama-3.3-70b-versatile`. 72 answer attempts and 33 refusal opportunities.

Run over four days via `scripts/run_eval_queue.py`, because Groq's free tier caps daily tokens
at 100,000 and the full sweep costs roughly three times that.

| Retriever | Recall@5 | CitePrec | Faithful | AnsRel | Correct | FalseRefuse | CorrectRefuse |
|---|---|---|---|---|---|---|---|
| **dense** | **0.85** | **0.29** | 0.77 | **0.69** | **0.62** | **0.08** | **1.00** |
| hybrid | 0.83 | 0.26 | 0.83 | 0.68 | 0.54 | 0.17 | 0.91 |
| sparse | 0.60 | 0.19 | **0.86** | 0.51 | 0.56 | 0.38 | **1.00** |

**Dense wins six of seven metrics.** Sparse wins exactly one — faithfulness — while refusing
38% of the questions it could have answered.

**32 of 33 unanswerable questions were correctly refused (97%).** Dense and sparse refused all
11; hybrid missed one.

### Faithfulness is gameable by refusing

Sparse posts the **highest faithfulness of any retriever (0.86)** while having the **worst
answer relevance (0.51)** and the **worst false-refusal rate (0.38)**. A refusal asserts
nothing, so it cannot be unfaithful and scores 1.0 by construction.

Read alongside its refusal rate, sparse's row says *"trustworthy because it barely answers."*

**Never report faithfulness without a false-refusal rate beside it.** This is a trap in the
metric itself, not in any implementation, and it applies to RAGAS and DeepEval faithfulness
scores identically.

### The one miss: all three layers passed a question that should have been refused

```
karpathy_tokenizer / hybrid / unans-cnn
  "How do convolutional neural networks process images?"
  (the transcript contains zero occurrences of "convolution")

  top_confidence      0.6704     passed Layer A (min_score 0.65)
  refused             0          Layers B and C also passed it
  faithfulness        1.0000     the judge scored it FAITHFUL
  answer_relevance    0.8000
  context_relevance   0.2000     the only metric that caught it
```

**This is not a classic hallucination.** Faithfulness was 1.0 because the answer genuinely was
supported by the retrieved passages — the model found material in the tokenizer transcript it
could stretch into a CNN answer, and cited it correctly. Nothing was fabricated. Every layer
behaved as designed. The failure was that the question should not have been answered at all.

**Faithfulness is structurally blind to this failure mode**, because it asks only *"is this
supported?"* and never *"should this have been answered?"* Context relevance caught it at 0.20.

Two supporting observations:

- Dense (0.651) and sparse (0.708) also cleared Layer A on this same question and refused at
  the generation layer instead. The difference was which passages hybrid retrieved, not a
  systematic weakness in fusion.
- Sparse scored **0.708 confidence on this unanswerable question** — higher than several
  genuinely answerable ones. No threshold cleanly separates the two populations, which is the
  quantitative form of the §5 argument.

### Five false refusals had already retrieved the answer

| Case | Retriever(s) | recall@k |
|---|---|---|
| `llama-training-cost` | dense, hybrid | 1.00 |
| `reversal-curse` | hybrid, sparse | 1.00 |
| `hallucination-cause` | sparse | 1.00 |

The correct passage was retrieved and then discarded by Layer A at 0.65. This is the cost of a
single global threshold: per-video calibration recommended 0.60 / 0.70 / 0.65, and 0.65 is too
aggressive for the intro talk.

`min_score` was **deliberately not changed mid-experiment** — jobs 1–2 had already run at 0.65,
and altering configuration partway would have made the nine results incomparable. Lowering it
to 0.60 is the obvious follow-up, and should cost nothing in correct refusal, since
correct-refusal was already 1.00 back when `min_score` was inert at 0.28 and Layers B and C
carried the entire load.

---

## 7. Threats to validity

- **n = 24 answerable questions across 3 videos, all one speaker.** Differences below ~0.15
  are one or two questions changing sides. Every table here is directional.
- **Retriever ranking is configuration-dependent** (§2). Do not quote it without the config.
- **Ground truth from transcripts, not viewing** (§1).
- **A single follow-up case** scores 0.00 across all retrievers and needs individual
  investigation rather than being averaged in.
- **`min_score` was 0.65 throughout**, which per-video calibration shows is too aggressive for
  one of the three videos. Absolute false-refusal rates are therefore pessimistic.
- **One embedding model, one judge model.** Judge agreement against human labels has not been
  measured, so the generation numbers carry unquantified judge bias — and §6 shows the judge
  scoring an off-topic answer as perfectly faithful, which is a concrete instance of that risk.

---

## 8. Defects this harness found

Six, none of which code review would have surfaced.

| # | Defect | How it was caught |
|---|---|---|
| 1 | `min_score` was inert — bge-small never scores below ~0.45 on English prose, so Layer A never fired | Calibration sweep flat at 0.00 across the whole lower range |
| 2 | Hybrid's `max(dense, sparse)` confidence made refusals *worse* than dense alone | BM25 scoring 0.618 on an unanswerable question, overriding dense's correct 0.551 |
| 3 | The harness searched 30 candidates where production sees 5, inflating every reported confidence | Two tools disagreeing with each other |
| 4 | nDCG returned 1.01 and 1.09 | Impossible value in the output table |
| 5 | Faithfulness is gameable by refusing | Sparse posting the highest faithfulness while refusing 38% of answerable questions |
| 6 | Faithfulness is blind to off-topic-but-grounded answers | A CNN question answered from a tokenizer transcript, scored 1.0 faithful; only context relevance (0.20) flagged it |

Defects 3 and 4 were in the measurement code itself. Defects 5 and 6 are in the *metric
definitions* — faithfulness rewards silence and cannot see topical drift. Together they are the
argument for treating an evaluation harness as a component that needs testing, not as a trusted
oracle, and for never reading a single metric in isolation.
