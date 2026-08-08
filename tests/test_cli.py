from typer.testing import CliRunner

from ytchat.cli.app import app

runner = CliRunner()


def test_help_lists_the_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("chat", "ask", "cache"):
        assert cmd in result.stdout


def test_cache_stats_runs_on_an_empty_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("YTCHAT_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["cache", "stats"])
    assert result.exit_code == 0 and "videos" in result.stdout


def test_cache_clear_requires_a_video(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("YTCHAT_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["cache", "clear"])
    assert result.exit_code == 2, "clearing the whole cache must not be a one-word command"


def test_invalid_url_exits_with_the_typed_code(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("YTCHAT_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["chat", "not-a-youtube-url"])
    assert result.exit_code != 0