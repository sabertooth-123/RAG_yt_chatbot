import pytest

from ytchat.database.repository import Repository
from ytchat.evaluation.dataset import Benchmark, EvalCase, TimeSpan
from ytchat.evaluation.generation_metrics import LLMJudge, _parse_score
from ytchat.evaluation.report import Report, comparison_table
from ytchat.evaluation.runner import EvalRunner, best_threshold
from ytchat.generation.llm import ScriptedLLM


@pytest.fixture
def benchmark(metadata):
    return Benchmark(
        name="fixture",
        video_url=f"https://youtu.be/{metadata.video_id}",
        cases=(
            EvalCase(id="attn", question="What is attention?",
                     expected_answer="Focusing on relevant tokens.",
                     relevant_spans=(TimeSpan(6.5, 11.0),), tags=("semantic",)),
            EvalCase(id="pos", question="What are positional encodings?",
                     relevant_spans=(TimeSpan(31.0, 36.0),), tags=("lexical",)),
            EvalCase(id="unans", question="What is the capital of Peru?",
                     answerable=False),
        ),
    )


def test_judge_score_parsing() -> None:
    assert _parse_score('{"score": 0.8, "reason": "ok"}')[0] == pytest.approx(0.8)
    assert _parse_score('Here you go: {"score": 1.0, "reason": "good"}')[0] == 1.0
    assert _parse_score("score: 0.5")[0] == pytest.approx(0.5)
    assert _parse_score("total nonsense")[0] == 0.0
    assert _parse_score('{"score": 7}')[0] == 1.0, "scores must be clamped to [0,1]"


def test_refusal_is_not_penalised_for_faithfulness(index) -> None:
    from ytchat.models import Answer

    judge = LLMJudge(ScriptedLLM(['{"score": 0.0, "reason": "n/a"}'] * 4))
    refusal = Answer(text="I cannot find this information in the video.",
                     citations=(), refused=True)
    scores = judge.score("q?", refusal, [])
    assert scores.faithfulness == 1.0, "a refusal invents nothing, so it is faithful"


def test_retrieval_only_evaluation_makes_no_llm_calls(
    benchmark, settings, providers, embedder, metadata
) -> None:
    transcripts, _ = providers
    settings.ensure_dirs()
    with Repository(settings.db_path) as repo:
        from ytchat.pipeline import ensure_indexed

        ensure_indexed(metadata.video_id, settings, repo,
                       transcript_provider=transcripts,
                       metadata_provider=providers[1], embedder=embedder)
        llm = ScriptedLLM(["never used"])
        runner = EvalRunner(benchmark, settings, repo, llm=llm)
        result = runner.evaluate_retrieval("hybrid")

    assert len(result.cases) == 3
    assert llm.calls == [], "retrieval-only evaluation must not call the LLM"
    assert 0.0 <= result.retrieval.recall_at_k <= 1.0


def test_threshold_calibration_costs_no_llm_calls(
    benchmark, settings, providers, embedder, metadata
) -> None:
    transcripts, metas = providers
    settings.ensure_dirs()
    with Repository(settings.db_path) as repo:
        from ytchat.pipeline import ensure_indexed

        ensure_indexed(metadata.video_id, settings, repo,
                       transcript_provider=transcripts, metadata_provider=metas,
                       embedder=embedder)
        llm = ScriptedLLM(["never used"])
        runner = EvalRunner(benchmark, settings, repo, llm=llm)
        points = runner.calibrate_threshold("hybrid", thresholds=[0.0, 0.5, 1.0])

    assert llm.calls == []
    assert [p.threshold for p in points] == [0.0, 0.5, 1.0]
    # At threshold 0 nothing is refused; at 1.0 everything is.
    assert points[0].false_refusal_rate == 0.0
    assert points[-1].correct_refusal_rate == 1.0
    assert best_threshold(points) is not None


def test_report_renders_markdown(benchmark) -> None:
    from ytchat.evaluation.runner import RetrieverResult

    results = [RetrieverResult(retriever=n) for n in ("dense", "sparse", "hybrid")]
    table = comparison_table(results)
    assert "Recall@k" in table and "CitePrec" in table
    md = Report(benchmark=benchmark, results=results).to_markdown()
    assert "# Evaluation — fixture" in md
    assert "1 unanswerable" in md


def test_csv_export_has_one_row_per_case(benchmark, tmp_path) -> None:
    from ytchat.evaluation.retrieval_metrics import RetrievalScores
    from ytchat.evaluation.runner import CaseResult, RetrieverResult

    result = RetrieverResult(
        retriever="hybrid",
        cases=[CaseResult(case=c, retrieval=RetrievalScores()) for c in benchmark.cases],
    )
    path = tmp_path / "out.csv"
    Report(benchmark=benchmark, results=[result]).to_csv(path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1 + len(benchmark.cases)