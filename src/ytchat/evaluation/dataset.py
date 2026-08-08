"""Benchmark schema.

Ground truth is time ranges, not chunk IDs — so a benchmark written today stays
valid after you change chunk size, embedding model, or retriever.  That property
is what makes the parameter sweeps meaningful rather than self-referential.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ytchat.errors import ConfigurationError
from ytchat.ingestion.url import parse_video_id


@dataclass(frozen=True, slots=True)
class TimeSpan:
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.end_s < self.start_s:
            raise ConfigurationError(f"Inverted span: {self.start_s} > {self.end_s}")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @classmethod
    def parse(cls, value: Any) -> "TimeSpan":
        """Accepts ``[12.5, 40.0]``, ``"12:43-13:20"``, or ``{start: .., end: ..}``."""
        if isinstance(value, dict):
            return cls(float(value["start"]), float(value["end"]))
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return cls(_to_seconds(value[0]), _to_seconds(value[1]))
        if isinstance(value, str) and "-" in value:
            lo, hi = value.split("-", 1)
            return cls(_to_seconds(lo.strip()), _to_seconds(hi.strip()))
        raise ConfigurationError(f"Cannot parse time span: {value!r}")


def _to_seconds(value: Any) -> float:
    """``90``, ``"90"``, ``"1:30"``, ``"1:02:07"`` → seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).strip().split(":")
    if len(parts) == 1:
        return float(parts[0])
    total = 0.0
    for part in parts:
        total = total * 60 + float(part)
    return total


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    question: str
    expected_answer: str | None = None
    relevant_spans: tuple[TimeSpan, ...] = ()
    answerable: bool = True
    tags: tuple[str, ...] = ()          # lexical | semantic | multi-hop | temporal
    followup_to: str | None = None      # id of the preceding case, for multi-turn


@dataclass(frozen=True, slots=True)
class Benchmark:
    name: str
    video_url: str
    cases: tuple[EvalCase, ...]
    notes: str = ""

    @property
    def video_id(self) -> str:
        return parse_video_id(self.video_url)

    @property
    def answerable(self) -> tuple[EvalCase, ...]:
        return tuple(c for c in self.cases if c.answerable)

    @property
    def unanswerable(self) -> tuple[EvalCase, ...]:
        return tuple(c for c in self.cases if not c.answerable)

    # -- io -----------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Benchmark":
        try:
            import yaml
        except ImportError as exc:
            raise ConfigurationError("Benchmarks need PyYAML: pip install pyyaml") from exc

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        cases = tuple(
            EvalCase(
                id=str(c["id"]),
                question=c["question"],
                expected_answer=c.get("expected_answer"),
                relevant_spans=tuple(TimeSpan.parse(s) for s in c.get("relevant_spans", [])),
                answerable=bool(c.get("answerable", True)),
                tags=tuple(c.get("tags", [])),
                followup_to=c.get("followup_to"),
            )
            for c in raw.get("cases", [])
        )
        bench = cls(
            name=raw.get("name", Path(path).stem),
            video_url=raw["video_url"],
            cases=cases,
            notes=raw.get("notes", ""),
        )
        problems = bench.validate()
        if problems:
            raise ConfigurationError(
                f"Benchmark {path} is invalid:\n" + "\n".join(f"  • {p}" for p in problems)
            )
        return bench

    def validate(self) -> list[str]:
        problems: list[str] = []
        seen: set[str] = set()
        ids = {c.id for c in self.cases}
        for case in self.cases:
            if case.id in seen:
                problems.append(f"duplicate case id {case.id!r}")
            seen.add(case.id)
            if case.answerable and not case.relevant_spans:
                problems.append(
                    f"{case.id}: answerable cases need relevant_spans "
                    "(otherwise recall is undefined)"
                )
            if not case.answerable and case.relevant_spans:
                problems.append(
                    f"{case.id}: unanswerable cases must not have relevant_spans"
                )
            if case.followup_to and case.followup_to not in ids:
                problems.append(f"{case.id}: followup_to {case.followup_to!r} does not exist")
        if not self.cases:
            problems.append("benchmark has no cases")
        return problems

    def conversation_order(self) -> list[EvalCase]:
        """Cases ordered so every follow-up comes after its antecedent."""
        by_id = {c.id: c for c in self.cases}
        ordered: list[EvalCase] = []
        placed: set[str] = set()

        def place(case: EvalCase) -> None:
            if case.id in placed:
                return
            if case.followup_to:
                place(by_id[case.followup_to])
            ordered.append(case)
            placed.add(case.id)

        for case in self.cases:
            place(case)
        return ordered