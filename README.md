# yt-chat
blog link: https://evergreen-learning-5e5.notion.site/Built-a-Youtube-Q-A-bot-that-refuses-to-guess-3accb2d4f8b4805a89e3ee559101cc11
Ask questions about a YouTube video from your terminal. You get an answer plus
timestamps you can click to jump to that exact moment.

If the video doesn't cover what you asked, it says so instead of making something up.

```
$ yt-chat https://youtu.be/zjkBMFhNj_g

  [1hr Talk] Intro to Large Language Models
  Andrej Karpathy · 59:47 · 140 chunks

› What two files make up a large language model?

  A parameters file holding the model weights (140 GB for Llama 2 70B) and a
  run file with the code that executes them, about 500 lines of C [1][2].

  [1] 1:34   https://youtube.com/watch?v=zjkBMFhNj_g&t=94s
  [2] 2:15   https://youtube.com/watch?v=zjkBMFhNj_g&t=135s

› What is the attention mechanism?

  I cannot find this information in the video.
```

That last bit is the whole point. This video never mentions attention, so it
refuses. Across 33 test questions the videos genuinely don't answer, it refused 32.

---

## Setup

```bash
git clone https://github.com/sabertooth-123/RAG_yt_chatbot
cd RAG_yt_chatbot
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[embeddings,llm,metadata,dev]"
```

If you have an NVIDIA GPU, install the CUDA build of torch first, otherwise it
runs on CPU and everything is slower:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Grab a free API key from [Groq](https://console.groq.com/keys), copy
`.env.example` to `.env`, and paste it in. Or set `YTCHAT_LLM_PROVIDER=ollama`
and run a model locally with no key at all.

---

## Commands

### Chat with a video

```bash
yt-chat https://youtu.be/VIDEO_ID
```

Opens a chat session. First run takes a minute or two while it downloads the
subtitles and processes them. After that it's instant, because everything is
cached.

You can also just paste the video ID:

```bash
yt-chat VIDEO_ID
```

Options you can add:

| Option | What it does |
|---|---|
| `--retriever dense` | Pick the search method: `dense`, `sparse`, or `hybrid` |
| `--top-k 8` | How many passages to look at (default 5) |
| `--llm groq` | Which AI service to use this run |
| `--force` | Ignore the cache and reprocess the video |
| `--quiet` | Hide the progress messages |

### Inside the chat

Just type your question. Or use these:

| Command | What it does |
|---|---|
| `/ask <question>` | Same as typing the question |
| `/sources` | Shows where the last answer came from, with quotes |
| `/history` | The conversation so far |
| `/change-retriever hybrid` | Switch search method without restarting |
| `/compare <question>` | Runs all three search methods side by side |
| `/video` | Video title, length, number of passages, models in use |
| `/debug` | Why the last answer came out how it did — scores, and which check refused it |
| `/clear` | Forget the conversation |
| `/help` | List these |
| `/exit` | Quit |

Follow-up questions work. Ask "What is a jailbreak?" then "Why does that work?"
and it figures out what "that" means before searching.

### Ask one question and exit

```bash
yt-chat ask https://youtu.be/VIDEO_ID -Q "What is the LLM OS?"
```

Handy for scripts. Exits with code `0` if it answered, `1` if it refused.

### Manage the cache

```bash
yt-chat cache stats
```

Shows how many videos and passages are stored locally.

```bash
yt-chat cache clear --video https://youtu.be/VIDEO_ID
```

Deletes everything saved for one video. It won't wipe the whole cache in one
command, on purpose.

### Measure how well it works

```bash
yt-chat eval calibrate benchmarks/karpathy_llm_intro.yaml -r dense
```

Finds the best cutoff for when to refuse. Costs nothing — no AI calls involved.

```bash
yt-chat eval run benchmarks/karpathy_llm_intro.yaml --no-judge
```

Compares all three search methods. `--no-judge` skips the AI grading, which
makes it free.

```bash
yt-chat eval run benchmarks/karpathy_llm_intro.yaml -o docs/evaluation.md
```

The full version, including answer quality. This one uses a lot of API calls.

```bash
yt-chat eval draft https://youtu.be/VIDEO_ID -Q "your question"
```

Suggests time ranges for a new test question. Check them yourself before
trusting them — it's only guessing.

### Scripts

```bash
python scripts/run_eval_queue.py
```

Runs the full evaluation across all videos and search methods. Free API tiers
cap you daily, so it saves progress and picks up where it left off. Run it
again the next day.

```bash
python scripts/run_eval_queue.py --list
```

Shows what's done and what's left.

```bash
python scripts/merge_eval.py --markdown
```

Combines all the results into one table.

```bash
python scripts/sweep_chunk_size.py
```

Tests different passage sizes. Free to run.

### Tests

```bash
pytest -q
```

195 tests, about two seconds. No internet, no API key, no model downloads
needed.

---

## Settings

Everything lives in `.env`, so you can change how it behaves without touching
any code.

| Setting | What it does |
|---|---|
| `YTCHAT_LLM_PROVIDER` | `groq`, `gemini`, `openrouter`, or `ollama` |
| `YTCHAT_RETRIEVER` | Default search method |
| `YTCHAT_MIN_SCORE` | How sure it has to be before answering. Higher = refuses more |
| `YTCHAT_TOP_K` | How many passages to send to the AI |
| `YTCHAT_MAX_CHARS` | Passage size. Changing this re-cuts the transcript |
| `YTCHAT_EMBEDDING_MODEL` | Which model converts text to numbers |
| `YTCHAT_ENABLE_RERANK` | Turn on the slower, more accurate second pass |
| `YTCHAT_ENABLE_QUERY_REWRITING` | Turn follow-up rewriting on or off |

One warning: `MIN_SCORE` is tuned for the current passage size and embedding
model. Change either of those and you should re-run `eval calibrate`.

---

## How it works

```
YouTube URL
    ↓
get subtitles (with timestamps)
    ↓
clean them (strip [Music], remove repeated lines)
    ↓
cut into ~700 character passages, keeping the timing
    ↓
convert each passage to numbers (embeddings) → FAISS + SQLite
    ↓
search (meaning-based, keyword, or both)
    ↓
send only the top passages to the AI with strict instructions
    ↓
check the answer's sources are real → answer + timestamps
```

The part I'd point at is the timing. Most projects glue the subtitles together
and then guess which timestamp a passage came from. This one records which
characters belong to which moment, so even a passage that starts mid-sentence
gets an accurate time.

There are three separate checks that can refuse a question:

1. Nothing relevant was found — refuses before calling the AI at all
2. The AI itself says the passages don't answer it
3. The answer cites a source that doesn't exist, so it gets thrown away

No LangChain or LlamaIndex here. The keyword search, the rank fusion, and the
whole evaluation setup are written directly.

---

## Results

Tested on three Karpathy videos totalling 6h45m, with 35 questions I wrote and
checked by hand. 24 are answerable, 11 aren't.

| Search method | Finds answer | Faithful | Wrongly refused | Correctly refused |
|---|---|---|---|---|
| dense | 0.85 | 0.77 | 0.08 | 100% |
| hybrid | 0.83 | 0.83 | 0.17 | 91% |
| sparse | 0.60 | 0.86 | 0.38 | 100% |

Dense is the default because it wins on nearly everything.

Sparse looks best on "faithful" only because it refuses 38% of the questions it
could have answered — a refusal can't be unfaithful, so it scores full marks
automatically. Worth knowing if you ever report that metric on its own.

Full breakdown, including the eight things measurement caught that I'd never
have found by reading the code, is in [docs/evaluation.md](docs/evaluation.md).

---

## What doesn't work

- **Videos with subtitles turned off.** About 15% of YouTube. Would need speech
  recognition, which isn't built yet.
- **Non-English videos** work through translated subtitles, but retrieval is
  noticeably worse.
- **Small sample.** 24 answerable questions from one speaker. Differences under
  about 0.15 in that table are one or two questions changing sides, so don't
  read too much into them.

---

MIT licensed.
