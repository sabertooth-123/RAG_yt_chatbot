"""Generate docs/yt-chat-walkthrough.pdf -- the plain-language project overview.

    python scripts/build_walkthrough.py

Deliberately written for a non-specialist reader: no jargon without a plain
gloss.  The technical depth lives in docs/technical_walkthrough.md.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(__file__).resolve().parent.parent / "docs" / "yt-chat-walkthrough.pdf"

INK = colors.HexColor("#1a1a2e")
ACCENT = colors.HexColor("#0f6f74")
ACCENT_LT = colors.HexColor("#e6f2f2")
MUTED = colors.HexColor("#5a6472")
RULE = colors.HexColor("#d6dce3")
CODE_BG = colors.HexColor("#f5f6f8")

PAGE_W, PAGE_H = A4
MARGIN = 1.9 * cm
W = PAGE_W - 2 * MARGIN

ss = getSampleStyleSheet()


def style(name, **kw):
    base = kw.pop("parent", ss["BodyText"])
    return ParagraphStyle(name, parent=base, **kw)


TitleS = style("TitleS", parent=ss["Title"], fontName="Helvetica-Bold",
               fontSize=30, leading=35, textColor=INK, spaceAfter=6)
SubtitleS = style("SubtitleS", fontSize=13.5, leading=19, textColor=ACCENT,
                  alignment=TA_CENTER, fontName="Helvetica", spaceAfter=4)
TagS = style("TagS", fontSize=9.5, leading=14, textColor=MUTED,
             alignment=TA_CENTER, fontName="Helvetica")
H1 = style("H1", fontName="Helvetica-Bold", fontSize=17, leading=21,
           textColor=ACCENT, spaceBefore=20, spaceAfter=9)
H2 = style("H2", fontName="Helvetica-Bold", fontSize=12.5, leading=16,
           textColor=INK, spaceBefore=13, spaceAfter=5)
Body = style("Body", fontSize=10, leading=15.2, textColor=INK,
             alignment=TA_LEFT, spaceAfter=7)
Bullet = style("Bullet", parent=Body, leftIndent=13, bulletIndent=3, spaceAfter=4)
Small = style("Small", fontSize=8.6, leading=12.4, textColor=MUTED, spaceAfter=6)
Code = style("Code", fontName="Courier-Bold", fontSize=8.8, leading=13,
             textColor=colors.HexColor("#0b3d3f"), backColor=CODE_BG,
             borderPadding=(5, 6, 5, 6), leftIndent=2, spaceBefore=3, spaceAfter=8)
CellB = style("CellB", fontSize=9, leading=12.8, textColor=INK, spaceAfter=0)
CellC = style("CellC", fontName="Courier-Bold", fontSize=8.4, leading=12,
              textColor=colors.HexColor("#0b3d3f"), spaceAfter=0)
CellH = style("CellH", fontName="Helvetica-Bold", fontSize=8.8, leading=12,
              textColor=colors.white, spaceAfter=0)
Lead = style("Lead", fontSize=11, leading=17, textColor=MUTED, spaceAfter=10)


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def code(t: str):
    return Paragraph(esc(t), Code)


def bullets(items):
    return [Paragraph(t, Bullet, bulletText="•") for t in items]


def table(rows, widths, header=True, code_col=0):
    data = []
    for r_i, row in enumerate(rows):
        cells = []
        for c_i, cell in enumerate(row):
            if r_i == 0 and header:
                cells.append(Paragraph(esc(str(cell)), CellH))
            elif c_i == code_col:
                cells.append(Paragraph(esc(str(cell)), CellC))
            else:
                cells.append(Paragraph(str(cell), CellB))
        data.append(cells)
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafbfc")]),
    ]
    if header:
        cmds.append(("BACKGROUND", (0, 0), (-1, 0), ACCENT))
    t.setStyle(TableStyle(cmds))
    return t


def callout(title, text):
    inner = [Paragraph(f"<b>{title}</b>", CellB), Spacer(1, 3), Paragraph(text, CellB)]
    t = Table([[inner]], colWidths=[W], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ACCENT_LT),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, ACCENT),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
    ]))
    return [Spacer(1, 4), t, Spacer(1, 10)]


def decorate(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, PAGE_H - MARGIN + 0.5 * cm, "yt-chat  |  Project Walkthrough")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN, PAGE_H - MARGIN + 0.25 * cm,
                    PAGE_W - MARGIN, PAGE_H - MARGIN + 0.25 * cm)
        canvas.drawRightString(PAGE_W - MARGIN, 1.1 * cm, str(doc.page))
    canvas.restoreState()


story = []

# ------------------------------------------------------------------ cover
story += [
    Spacer(1, 3.2 * cm),
    Paragraph("yt-chat", TitleS),
    Paragraph("Ask questions about any YouTube video<br/>and get answers with clickable timestamps",
              SubtitleS),
    Spacer(1, 0.5 * cm),
    Paragraph("A Complete Project Walkthrough", TagS),
    Paragraph("Python &middot; RAG &middot; Command Line", TagS),
    Spacer(1, 1.4 * cm),
]
story.append(table([["", ""], [
    "What it is",
    "A terminal app that reads a YouTube video's subtitles and answers your questions about it."
], [
    "The rule it follows",
    "It only uses what the speaker actually said. If the video does not cover it, it says so "
    "instead of guessing."
], [
    "What you get back",
    "A short answer, plus timestamps you can click to jump to that exact moment in the video."
], [
    "Built with",
    "Python, SQLite, FAISS, Sentence-Transformers, Typer, and a free LLM API (Groq, Gemini, "
    "OpenRouter) or a local model."
], [
    "Tested on",
    "3 videos totalling 6 hours 45 minutes, with 35 hand-written questions."
], [
    "Test suite",
    "195 tests, run in about 2 seconds, with no internet needed."
]], [4.1 * cm, W - 4.1 * cm], header=False, code_col=99))
story.append(PageBreak())

# ------------------------------------------------------------------ part 1
story += [
    Paragraph("1. What This Project Does", H1),
    Paragraph(
        "Imagine watching a three-hour technical talk and wanting to know one specific thing. "
        "Scrubbing the timeline is slow. Searching the web gives answers from somewhere else, "
        "which may not match what this particular speaker said.", Body),
    Paragraph(
        "<b>yt-chat</b> solves that. You give it a YouTube link. It reads the video's subtitles, "
        "then lets you ask questions in plain English. Every answer comes with the timestamp "
        "where the speaker said it, so you can verify it yourself in one click.", Body),
]
story += callout(
    "The one promise this project makes",
    "The assistant never invents an answer. If the information is not in the video, it replies: "
    "<i>\"I cannot find this information in the video.\"</i> Three separate safety checks enforce "
    "this. Measured across 11 questions the videos genuinely do not answer, it refused "
    "<b>every single one</b>."
)
story += [
    Paragraph("How it works, step by step", H2),
    Paragraph("Six stages. The first time you use a video it runs all six. Every time after "
              "that it loads saved results instantly.", Body),
]
story.append(table([
    ["Stage", "What happens"],
    ["1. Get the words",
     "Downloads the video's subtitles, including the exact time each line was spoken."],
    ["2. Clean them up",
     "Removes noise like [Music] and speaker labels. Auto-subtitles repeat the previous line's "
     "ending, so those duplicates are stripped too."],
    ["3. Cut into pieces",
     "Splits the transcript into passages of about 700 characters, roughly 36 seconds of "
     "speech. Each piece keeps its start and end time."],
    ["4. Turn into numbers",
     "Converts each piece into a list of numbers that captures its meaning, so similar ideas "
     "end up close together."],
    ["5. Find the right pieces",
     "When you ask something, it finds the handful of passages most likely to contain the "
     "answer."],
    ["6. Write the answer",
     "Sends only those passages to an AI model with strict instructions: use nothing else, "
     "cite your sources, and refuse if the answer is not there."],
], [4.3 * cm, W - 4.3 * cm], code_col=99))
story += callout(
    "The clever part: timestamps never get lost",
    "Most similar projects glue the subtitles into one block of text and then guess which "
    "timestamp a passage came from. This project instead records exactly which characters "
    "belong to which moment, so a passage starting halfway through a subtitle line still gets "
    "an accurate time. That is what makes the citations trustworthy rather than approximate."
)
story.append(PageBreak())

# ------------------------------------------------------------------ part 2
story += [Paragraph("2. Features", H1),
          Paragraph("Everything below is built and covered by tests.", Lead),
          Paragraph("Reading the video", H2)]
story += bullets([
    "Accepts every YouTube link format: normal links, short youtu.be links, Shorts, embeds, "
    "live links, or just the bare video ID.",
    "Prefers human-written subtitles; falls back to auto-generated, then translated ones.",
    "Adapts to subtitle quality. Human subtitles have punctuation, so it splits on sentences. "
    "Auto-subtitles often have none, so it switches to fixed-size word windows.",
    "Fetches title, channel and duration where available, but never lets a missing title stop "
    "you asking questions.",
])
story += [Paragraph("Answering questions", H2)]
story += bullets([
    "Answers come only from the video, in one to four sentences, in the speaker's own terms.",
    "<b>Three independent refusal checks.</b> If nothing relevant was found it refuses without "
    "calling the AI at all, saving an API call. The AI is also instructed to signal when the "
    "passages do not answer the question. Finally, any answer citing a source that does not "
    "exist is thrown away.",
    "Every sentence is tagged with a source number, and every source becomes a clickable "
    "timestamp link.",
    "Follow-up questions work. Ask \"What is a jailbreak?\" then \"Why does that work?\" and it "
    "rewrites the second into a full question before searching.",
])
story += [Paragraph("Searching", H2)]
story += bullets([
    "<b>Meaning-based search</b> finds the right passage even when you use different words.",
    "<b>Keyword search</b> matches exact names, numbers and jargon.",
    "<b>Hybrid search</b> combines both.",
    "<b>Reranking</b> (optional) takes the top 30 results and rescores them with a slower, "
    "more accurate model.",
    "You can switch between all four at any time, even mid-conversation.",
])
story += [Paragraph("Speed, storage and measurement", H2)]
story += bullets([
    "Each video is processed once, then stored locally, so the second run is instant.",
    "The cache is precise about what to redo. Change the passage size and it re-cuts but does "
    "not re-download. Change the AI model and it keeps the passages.",
    "A built-in evaluation system scores how well it finds the right passages and how "
    "trustworthy the answers are.",
    "Correct answers are defined by <i>time ranges</i>, not passage numbers, so the test set "
    "stays valid even after changing how the transcript is cut up.",
    "It measures something most projects skip: whether the timestamp actually points at the "
    "right moment, not just whether the answer was right.",
    "It can tune its own refusal threshold without spending a single API call.",
])
story += [Paragraph("Choice of AI model (all free)", H2)]
story.append(table([
    ["Option", "Notes"],
    ["groq", "Recommended. Free, very fast, runs Llama 3.3 70B. Free key required."],
    ["gemini", "Google AI Studio free tier. Free key required."],
    ["openrouter", "Several free community models. Free key required."],
    ["ollama", "Runs entirely on your own computer. No key, no limits, works offline."],
], [3.2 * cm, W - 3.2 * cm]))
story.append(PageBreak())

# ------------------------------------------------------------------ part 3
story += [Paragraph("3. Commands", H1),
          Paragraph("Every command starts with <font face='Courier-Bold'>yt-chat</font>. Run "
                    "them from the project folder with the virtual environment active.", Lead),
          Paragraph("Starting a conversation", H2)]
story.append(code("yt-chat https://youtu.be/zjkBMFhNj_g"))
story += [Paragraph("Or paste just the video ID:", Body)]
story.append(code("yt-chat zjkBMFhNj_g"))
story += [Paragraph("Useful options:", Body)]
story.append(table([
    ["Option", "What it does"],
    ["--retriever dense", "Search method: dense, sparse or hybrid."],
    ["--top-k 8", "How many passages to look at. More context, slower and pricier."],
    ["--llm groq", "Which AI service to use for this run."],
    ["--force", "Ignore the saved cache and reprocess from scratch."],
    ["--quiet", "Hide progress messages."],
], [4.5 * cm, W - 4.5 * cm]))

story += [Paragraph("Commands inside the chat", H2),
          Paragraph("Once the session is open, just type your question. Or use these:", Body)]
story.append(table([
    ["Command", "What it does"],
    ["/ask What is the LLM OS?", "Asks a question. Typing the question directly works too."],
    ["/sources", "Shows sources for the last answer, with quotes."],
    ["/history", "Shows the whole conversation so far."],
    ["/change-retriever sparse", "Switches search method without restarting."],
    ["/compare what is a jailbreak", "Runs all three search methods side by side."],
    ["/video", "Title, length, number of passages, models in use."],
    ["/debug", "Explains the last answer: passages used, their scores, and which safety check "
               "refused it if it refused."],
    ["/clear", "Forgets the conversation so far."],
    ["/help", "Lists these commands."],
    ["/exit", "Quits."],
], [5.5 * cm, W - 5.5 * cm]))

story += [Paragraph("Asking a single question", H2),
          Paragraph("For scripts, or when you want one answer without a chat session.", Body)]
story.append(code('yt-chat ask https://youtu.be/zjkBMFhNj_g -Q "What is the LLM OS?"'))
story += [Paragraph("It exits with code 0 if it answered and 1 if it refused, so it can be "
                    "used inside other scripts.", Body),
          Paragraph("Managing saved data", H2)]
story.append(code("yt-chat cache stats"))
story.append(code("yt-chat cache clear --video https://youtu.be/zjkBMFhNj_g"))
story += [Paragraph("The second deletes everything saved for one video. It deliberately "
                    "refuses to wipe the whole cache in one command.", Body)]
story.append(PageBreak())

story += [Paragraph("Measuring quality", H2),
          Paragraph("Finds the best refusal threshold. Costs nothing, because it makes no AI "
                    "calls:", Body)]
story.append(code("yt-chat eval calibrate benchmarks/karpathy_llm_intro.yaml -r dense"))
story += [Paragraph("Compares all search methods. Add --no-judge to skip AI scoring and make "
                    "it free:", Body)]
story.append(code("yt-chat eval run benchmarks/karpathy_llm_intro.yaml --no-judge"))
story += [Paragraph("Runs the full evaluation, resuming wherever the daily free-tier "
                    "allowance ran out:", Body)]
story.append(code("python scripts/run_eval_queue.py"))
story += [Paragraph("Tests how passage size affects quality. Free to run:", Body)]
story.append(code("python scripts/sweep_chunk_size.py"))

story += [Paragraph("Changing settings", H2),
          Paragraph("Every setting can be changed through the <font face='Courier'>.env</font> "
                    "file, without touching code. This is how experiments are run.", Body)]
story.append(table([
    ["Setting", "Meaning"],
    ["YTCHAT_LLM_PROVIDER", "Which AI service: groq, gemini, openrouter or ollama."],
    ["YTCHAT_RETRIEVER", "Default search method."],
    ["YTCHAT_MIN_SCORE", "How confident it must be before answering. Higher refuses more often."],
    ["YTCHAT_TOP_K", "How many passages to send to the AI."],
    ["YTCHAT_MAX_CHARS", "Size of each passage. Changing this re-cuts the transcript."],
    ["YTCHAT_EMBEDDING_MODEL", "Which model turns text into numbers."],
    ["YTCHAT_ENABLE_RERANK", "Turn the slower, more accurate reranking on or off."],
    ["YTCHAT_ENABLE_QUERY_REWRITING", "Turn follow-up question rewriting on or off."],
], [6.2 * cm, W - 6.2 * cm]))
story += [Paragraph("Running the tests", H2)]
story.append(code("pytest -q"))
story += [Paragraph("All 195 tests run in about two seconds and need no internet, no API key, "
                    "and no model downloads.", Body)]
story.append(PageBreak())

# ------------------------------------------------------------------ part 4
story += [Paragraph("4. The Main Files", H1),
          Paragraph("About 35 source files. These are the ones that matter if you want to "
                    "understand or change how it works.", Lead),
          Paragraph("The heart of the system", H2)]
story.append(table([
    ["File", "What it does"],
    ["pipeline.py", "<b>Start here.</b> Ties everything together: takes a URL and returns a "
                    "ready-to-query video, skipping any stage already cached. Also holds the "
                    "conversation manager."],
    ["models.py", "The core data types. The rule that every piece of text carries a start and "
                  "end time is written down here."],
    ["config.py", "Every adjustable setting in one place, readable from the .env file."],
], [3.6 * cm, W - 3.6 * cm]))

story += [Paragraph("Turning a video into searchable pieces", H2)]
story.append(table([
    ["File", "What it does"],
    ["ingestion/url.py", "Understands every shape of YouTube link."],
    ["ingestion/transcript.py", "Downloads subtitles and picks the best available track."],
    ["preprocessing/clean.py", "Removes noise, speaker labels, and the repeated text that "
                               "auto-subtitles produce."],
    ["preprocessing/timeline.py", "<b>The key trick.</b> Maps every character position back to "
                                  "a moment in the video, so any cut point gets an accurate "
                                  "timestamp."],
    ["preprocessing/chunking.py", "Cuts the transcript into passages, choosing sentences or "
                                  "word windows depending on whether the subtitles have "
                                  "punctuation."],
], [4.6 * cm, W - 4.6 * cm]))

story += [Paragraph("Searching and answering", H2)]
story.append(table([
    ["File", "What it does"],
    ["database/vectorstore.py", "The search index. Rebuilds itself if the file goes missing."],
    ["retrieval/dense.py", "Meaning-based search."],
    ["retrieval/sparse.py", "Keyword search, written from scratch so it can be tuned."],
    ["retrieval/hybrid.py", "Combines both."],
    ["retrieval/rerank.py", "Optional second pass with a slower, more accurate model."],
    ["generation/llm.py", "Talks to Groq, Gemini, OpenRouter or a local model. Handles rate "
                          "limits and retries."],
    ["generation/prompts.py", "The exact instructions given to the AI, kept separate because "
                              "changing them changes the results."],
    ["generation/answerer.py", "The three refusal checks live here."],
    ["generation/citations.py", "Turns source numbers into clickable timestamp links, and "
                                "throws out invented ones."],
], [4.6 * cm, W - 4.6 * cm]))

story += [Paragraph("Storage, interface and measurement", H2)]
story.append(table([
    ["File", "What it does"],
    ["database/repository.py", "All reading and writing to the local database."],
    ["cli/app.py", "All the terminal commands."],
    ["cli/repl.py", "The interactive chat session and its slash commands."],
    ["evaluation/dataset.py", "The test-question format, with answers as time ranges."],
    ["evaluation/retrieval_metrics.py", "Scores how well the search worked."],
    ["evaluation/runner.py", "Runs the whole evaluation and the threshold tuning."],
    ["scripts/run_eval_queue.py", "Resumable evaluation queue for free-tier daily limits."],
    ["scripts/sweep_chunk_size.py", "The passage-size experiment."],
], [5.2 * cm, W - 5.2 * cm]))
story.append(PageBreak())

# ------------------------------------------------------------------ part 5
story += [
    Paragraph("5. What the Measurements Found", H1),
    Paragraph("Six findings from testing against 35 hand-written questions across three "
              "videos. All are real measurements that changed the system.", Lead),

    Paragraph("Finding 1: the confidence threshold was doing nothing", H2),
    Paragraph("The system refuses when it is not confident enough. That threshold was set by "
              "guesswork. Measurement showed the scoring model never produces a score below "
              "about 0.45 for ordinary English, even for completely unrelated passages, so the "
              "check never once triggered.", Body),

    Paragraph("Finding 2: combining searches made refusals worse", H2),
    Paragraph("Hybrid search originally took the higher confidence of its two methods. Keyword "
              "search sometimes scores highly on a question the video never answers, simply "
              "because a couple of rare words happen to appear. That false confidence overrode "
              "the meaning-based score, which had correctly rated it low.", Body),

    Paragraph("Finding 3: the measuring tool measured the wrong thing", H2),
    Paragraph("The evaluation looked at 30 passages while the live system only ever looks at "
              "five, making reported confidence higher than anything the real system sees. "
              "Caught because two tools disagreed with each other.", Body),

    Paragraph("Finding 4: a scoring formula could exceed its own maximum", H2),
    Paragraph("One ranking metric returned 1.09 on a scale that stops at 1.0, because several "
              "retrieved passages can overlap the same correct answer. Only visible because an "
              "impossible number appeared in a results table.", Body),

    Paragraph("Finding 5: the trustworthiness score can be gamed by refusing", H2),
    Paragraph("Keyword search scored a perfect 1.00 on faithfulness by refusing half the "
              "questions. A refusal claims nothing, so it cannot be unfaithful. A system that "
              "refuses everything looks flawless. Faithfulness must always be reported next to "
              "a false-refusal rate.", Body),

    Paragraph("Finding 6: passage size has a sweet spot, not a tradeoff", H2),
    Paragraph("The usual assumption is that smaller passages give tighter timestamps but find "
              "answers less often. Testing four sizes across all three videos showed both "
              "measures peak at the same point, around 700 characters or 36 seconds of speech. "
              "Timestamp accuracy doubles from 400 to 700 characters and then falls again, "
              "because accuracy is highest when a passage is about as long as the answer "
              "itself. The default was changed as a result.", Body),
]
story.append(table([
    ["Passage size", "Length", "Answer found", "Timestamp accuracy"],
    ["400 characters", "18 seconds", "0.75", "0.076"],
    ["<b>700 characters</b>", "<b>36 seconds</b>", "<b>0.83</b>", "<b>0.153</b>"],
    ["1000 characters", "55 seconds", "0.79", "0.148"],
    ["1500 characters", "85 seconds", "0.82", "0.129"],
], [4.0 * cm, 3.0 * cm, 3.4 * cm, W - 10.4 * cm], code_col=99))

story += callout(
    "Why these matter",
    "None of these six could have been found by reading the code or by checking that the "
    "program runs. They were only visible once the system was measured against questions with "
    "known answers. Two of them were bugs in the measuring code itself, which is the argument "
    "for treating an evaluation harness as something that needs testing, not as a trusted "
    "authority."
)
story.append(PageBreak())

# ------------------------------------------------------------------ part 6
story += [Paragraph("6. Current Status", H1)]
story.append(table([
    ["Area", "State"],
    ["Video processing", "<b>Working.</b> Any video that has subtitles."],
    ["Question answering", "<b>Working.</b> Verified on three videos totalling 6h45m."],
    ["Refusing to guess", "<b>Working.</b> Refused all 11 unanswerable questions. All three "
                          "checks covered by tests."],
    ["Clickable citations", "<b>Working.</b>"],
    ["Follow-up questions", "<b>Working.</b>"],
    ["Caching", "<b>Working.</b> Second run of a video is instant."],
    ["Four search methods", "<b>Working.</b> Including optional reranking."],
    ["Test suite", "<b>Working.</b> 195 tests, about 2 seconds, fully offline."],
    ["Measuring search quality", "<b>Done.</b> Three videos, 35 questions, results published."],
    ["Measuring answer quality",
     "<b>Done.</b> All nine runs complete, spread over four days because the free AI tier caps "
     "daily usage. 32 of 33 unanswerable questions correctly refused."],
    ["Videos without subtitles",
     "<b>Not built.</b> Roughly 15% of YouTube. Would need local speech recognition."],
], [4.8 * cm, W - 4.8 * cm]))

story += [Paragraph("The most revealing single result", H2)]
story += [Paragraph(
    "Out of 33 questions the videos genuinely do not answer, 32 were refused. The one that "
    "slipped through is more instructive than the 32 that did not. A question about "
    "convolutional neural networks, asked of the tokenizer video, was answered. The scoring "
    "system rated that answer <b>perfectly faithful</b> &mdash; and it was right to. The model "
    "did not invent anything; it found passages it could stretch into an answer and quoted them "
    "correctly. Every safety check behaved exactly as designed. The problem was that the "
    "question should never have been answered at all.", Body)]
story += [Paragraph(
    "Only one measure caught it: the score for how much of the retrieved material was actually "
    "relevant, which came out at 0.20. The lesson is that a trustworthiness score asks whether "
    "a claim is supported, never whether the question should have been answered &mdash; so it "
    "can never be read on its own.", Body)]

story += [Paragraph("What is left", H2)]
story += bullets([
    "Add speech recognition for videos with no subtitles.",
    "Publish the repository.",
])

story += callout(
    "An honest note on the numbers",
    "All results come from 35 questions across three videos by a single speaker. At that size, "
    "any difference smaller than about 0.15 is noise rather than a real effect. One finding "
    "demonstrates this directly: which search method ranks best <i>changes</i> depending on the "
    "passage size, so no single method can be declared the winner from this data."
)

story += [
    Spacer(1, 0.4 * cm),
    Paragraph("Built with Python 3.12. Requires a free API key from Groq, Google AI Studio or "
              "OpenRouter, or a local model through Ollama. Full technical detail in "
              "docs/technical_walkthrough.md; full results in docs/evaluation.md.", Small),
]


class Doc(BaseDocTemplate):
    pass


doc = Doc(str(OUT), pagesize=A4,
          leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN,
          title="yt-chat - Project Walkthrough", author="Pranav Gupta",
          subject="CLI YouTube Video Intelligence Assistant")
frame = Frame(MARGIN, MARGIN, W, PAGE_H - 2 * MARGIN, id="body")
doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=decorate)])

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.build(story)
print(f"Written: {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
