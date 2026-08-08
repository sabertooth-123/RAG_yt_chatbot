"""Resumable evaluation queue.

Free-tier LLM providers cap tokens per day, and a full judged sweep
(3 benchmarks x 3 retrievers) costs roughly three times that budget.  Rather
than a single run that dies partway and loses its work, this walks a job list,
writes each result as it completes, and skips anything already on disk.

Re-run it as often as you like.  Each invocation advances as far as the
remaining budget allows and stops cleanly when the cap is hit:

    python scripts/run_eval_queue.py

    --list      show queue status without running anything
    --reset ID  delete one job's output so it re-runs
    --no-judge  retrieval metrics only (free, no LLM calls)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ytchat.config import Settings  # noqa: E402
from ytchat.database.repository import Repository  # noqa: E402
from ytchat.errors import LLMError  # noqa: E402
from ytchat.evaluation.dataset import Benchmark  # noqa: E402
from ytchat.evaluation.report import Report  # noqa: E402
from ytchat.evaluation.runner import EvalRunner  # noqa: E402

BENCHMARKS = ["karpathy_llm_intro", "karpathy_tokenizer", "karpathy_deepdive"]
RETRIEVERS = ["dense", "hybrid", "sparse"]   # cheapest-signal-first ordering

RESULTS = ROOT / "results"
DOCS = ROOT / "docs"

# Providers report an exhausted daily allowance in several dialects.
CAP_MARKERS = ("tokens per day", "requests per day", "tpd", "rpd", "limit: 0", "per day")


def job_id(bench: str, retriever: str) -> str:
    return f"{bench}__{retriever}"


def csv_path(bench: str, retriever: str) -> Path:
    return RESULTS / f"{job_id(bench, retriever)}.csv"


def md_path(bench: str, retriever: str) -> Path:
    return DOCS / f"eval_{job_id(bench, retriever)}.md"


def is_done(bench: str, retriever: str) -> bool:
    p = csv_path(bench, retriever)
    return p.exists() and p.stat().st_size > 200


def queue() -> list[tuple[str, str]]:
    return [(b, r) for b in BENCHMARKS for r in RETRIEVERS]


def show_status() -> None:
    jobs = queue()
    done = [j for j in jobs if is_done(*j)]
    print(f"{len(done)}/{len(jobs)} jobs complete\n")
    for bench in BENCHMARKS:
        marks = " ".join(
            f"{r}={'OK ' if is_done(bench, r) else 'pending'}" for r in RETRIEVERS
        )
        print(f"  {bench:<22} {marks}")
    if len(done) == len(jobs):
        print("\nQueue complete. Run scripts/merge_eval.py to combine the results.")


def looks_like_cap(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in CAP_MARKERS)


def run_one(bench_name: str, retriever: str, judge: bool) -> None:
    bench = Benchmark.from_yaml(ROOT / "benchmarks" / f"{bench_name}.yaml")
    settings = Settings()
    RESULTS.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)

    with Repository(settings.db_path) as repo:
        runner = EvalRunner(
            bench, settings, repo,
            progress=lambda m: print(f"    {m}", flush=True),
        )
        results = [
            runner.evaluate_full(retriever, judge=judge) if judge
            else runner.evaluate_retrieval(retriever)
        ]
        thresholds = runner.calibrate_threshold(retriever)

    report = Report(
        benchmark=bench, results=results, thresholds=thresholds,
        settings_summary={
            "retriever": retriever,
            "top_k": settings.top_k,
            "max_chars": settings.max_chars,
            "min_score": settings.min_score,
            "embedding_model": settings.embedding_model,
            "llm": settings.resolved_model if judge else "n/a (retrieval only)",
        },
    )
    md_path(bench_name, retriever).write_text(report.to_markdown(), encoding="utf-8")
    report.to_csv(csv_path(bench_name, retriever))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show status, run nothing")
    ap.add_argument("--reset", metavar="JOB_ID", help="delete one job's output")
    ap.add_argument("--no-judge", action="store_true", help="retrieval only (free)")
    args = ap.parse_args()

    if args.reset:
        for bench in BENCHMARKS:
            for r in RETRIEVERS:
                if job_id(bench, r) == args.reset:
                    csv_path(bench, r).unlink(missing_ok=True)
                    md_path(bench, r).unlink(missing_ok=True)
                    print(f"Reset {args.reset}")
                    return 0
        print(f"No such job: {args.reset}")
        return 2

    if args.list:
        show_status()
        return 0

    pending = [j for j in queue() if not is_done(*j)]
    if not pending:
        print("Queue already complete.")
        show_status()
        return 0

    print(f"{len(pending)} job(s) pending.\n")
    completed = 0
    for bench, retriever in pending:
        print(f"[{job_id(bench, retriever)}] running...", flush=True)
        try:
            run_one(bench, retriever, judge=not args.no_judge)
        except LLMError as exc:
            if looks_like_cap(exc):
                print(
                    f"\n  Daily token allowance exhausted at {job_id(bench, retriever)}.\n"
                    f"  {completed} job(s) completed and saved this run.\n"
                    f"  Re-run this script after the allowance resets; finished jobs\n"
                    f"  are skipped automatically.\n"
                )
                show_status()
                return 0        # not a failure: the queue is designed to resume
            print(f"\n  LLM error: {exc}\n")
            return 8
        except KeyboardInterrupt:
            print("\n  Interrupted. Completed jobs are saved.\n")
            show_status()
            return 130
        completed += 1
        print(f"  done -> {csv_path(bench, retriever).name}\n", flush=True)

    print(f"All {completed} job(s) complete.")
    show_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
