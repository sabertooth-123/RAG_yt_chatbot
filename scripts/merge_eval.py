"""Merge completed evaluation jobs into a summary.

Works on partial results -- run it any time to see what the queue has banked so
far.  Reports per-retriever aggregates and flags the two failure modes that
matter: false refusals (an answer was retrieved and then discarded) and missed
refusals (an unanswerable question got answered).

    python scripts/merge_eval.py
    python scripts/merge_eval.py --markdown        # emit a docs-ready table
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

BENCHMARKS = ["karpathy_llm_intro", "karpathy_tokenizer", "karpathy_deepdive"]
RETRIEVERS = ["dense", "hybrid", "sparse"]


def load(bench: str, retriever: str) -> list[dict] | None:
    path = RESULTS / f"{bench}__{retriever}.csv"
    if not path.exists() or path.stat().st_size < 200:
        return None
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key)]
    return sum(vals) / len(vals) if vals else 0.0


def summarise(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"] == "1"]
    unanswerable = [r for r in rows if r["answerable"] == "0"]
    judged = [r for r in rows if r.get("faithfulness")]
    return {
        "n_ans": len(answerable),
        "n_una": len(unanswerable),
        "recall": mean(answerable, "recall_at_k"),
        "cite_prec": mean(answerable, "citation_precision"),
        "faithful": mean(judged, "faithfulness"),
        "ans_rel": mean(judged, "answer_relevance"),
        "ctx_rel": mean(judged, "context_relevance"),
        "correct": mean(answerable, "correctness"),
        "false_refuse": (
            sum(1 for r in answerable if r["refused"] == "1") / len(answerable)
            if answerable else 0.0
        ),
        "corr_refuse": (
            sum(1 for r in unanswerable if r["refused"] == "1") / len(unanswerable)
            if unanswerable else 0.0
        ),
    }


def failures(rows: list[dict]) -> tuple[list[str], list[str]]:
    """(false refusals with the passage already retrieved, missed refusals)."""
    false_refusals, missed = [], []
    for r in rows:
        if r["answerable"] == "1" and r["refused"] == "1":
            note = f"{r['case_id']} (recall={float(r['recall_at_k']):.2f}"
            note += ", passage WAS retrieved)" if float(r["recall_at_k"]) > 0 else ")"
            false_refusals.append(note)
        if r["answerable"] == "0" and r["refused"] == "0":
            missed.append(r["case_id"])
    return false_refusals, missed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true", help="emit a docs-ready table")
    args = ap.parse_args()

    jobs = {}
    for bench in BENCHMARKS:
        for retriever in RETRIEVERS:
            rows = load(bench, retriever)
            if rows:
                jobs[(bench, retriever)] = rows

    total = len(BENCHMARKS) * len(RETRIEVERS)
    if not jobs:
        print("No completed jobs yet. Run scripts/run_eval_queue.py first.")
        return 0

    print(f"{len(jobs)}/{total} jobs complete\n")

    # ---- per retriever, pooled over whatever benchmarks are done ----------
    pooled: dict[str, list[dict]] = defaultdict(list)
    for (bench, retriever), rows in jobs.items():
        pooled[retriever].extend(rows)

    header = (f"{'retriever':<10}{'n':>4}{'recall':>8}{'cite':>7}{'faith':>7}"
              f"{'ansrel':>8}{'correct':>9}{'falseRef':>10}{'corrRef':>9}")
    print(header)
    print("-" * len(header))
    for retriever in RETRIEVERS:
        if retriever not in pooled:
            continue
        s = summarise(pooled[retriever])
        print(f"{retriever:<10}{s['n_ans']:>4}{s['recall']:>8.2f}{s['cite_prec']:>7.2f}"
              f"{s['faithful']:>7.2f}{s['ans_rel']:>8.2f}{s['correct']:>9.2f}"
              f"{s['false_refuse']:>10.2f}{s['corr_refuse']:>9.2f}")

    # ---- per job ---------------------------------------------------------
    print("\nper job")
    for (bench, retriever), rows in sorted(jobs.items()):
        s = summarise(rows)
        print(f"  {bench:<20} {retriever:<7} recall={s['recall']:.2f} "
              f"faith={s['faithful']:.2f} falseRef={s['false_refuse']:.2f} "
              f"corrRef={s['corr_refuse']:.2f}")

    # ---- failures worth reading by hand ----------------------------------
    print("\nfailures")
    any_failures = False
    for (bench, retriever), rows in sorted(jobs.items()):
        fr, missed = failures(rows)
        if not fr and not missed:
            continue
        any_failures = True
        print(f"  {bench} / {retriever}")
        for note in fr:
            print(f"    FALSE REFUSAL   {note}")
        for case in missed:
            print(f"    MISSED REFUSAL  {case}  <-- answered an unanswerable question")
    if not any_failures:
        print("  none")

    if args.markdown:
        print("\n\n| Retriever | Recall@5 | CitePrec | Faithful | AnsRel | Correct | "
              "FalseRefuse | CorrectRefuse |")
        print("|---|---|---|---|---|---|---|---|")
        for retriever in RETRIEVERS:
            if retriever not in pooled:
                continue
            s = summarise(pooled[retriever])
            print(f"| {retriever} | {s['recall']:.2f} | {s['cite_prec']:.2f} | "
                  f"{s['faithful']:.2f} | {s['ans_rel']:.2f} | {s['correct']:.2f} | "
                  f"{s['false_refuse']:.2f} | **{s['corr_refuse']:.2f}** |")

    if len(jobs) < total:
        print(f"\n{total - len(jobs)} job(s) still pending -- "
              f"run scripts/run_eval_queue.py after the daily allowance resets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
