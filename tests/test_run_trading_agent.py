from __future__ import annotations

import tempfile
from pathlib import Path
import subprocess

from run_trading_agent import append_result, best_objective, extract_code_block, git_commit, run_cmd


def test_extract_code_block_python_fence() -> None:
    text = "hello\n```python\nprint('ok')\n```\n"
    out = extract_code_block(text)
    assert out is not None
    assert "print('ok')" in out


def test_run_cmd_timeout() -> None:
    result = run_cmd(["python3", "-c", "import time; time.sleep(1)"], timeout_sec=0)
    assert not result.ok
    assert result.code == 124


def test_run_cmd_timeout_decodes_byte_streams(monkeypatch) -> None:
    import run_trading_agent as agent

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["python", "strategy.py"],
            timeout=30,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr(agent.subprocess, "run", fake_run)

    result = run_cmd(["python", "strategy.py"], timeout_sec=30)

    assert not result.ok
    assert result.code == 124
    assert result.stdout == "partial stdout"
    assert "partial stderr" in result.stderr
    assert "TIMEOUT: exceeded 30s" in result.stderr


def test_append_result_and_best_objective() -> None:
    import run_trading_agent as agent

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "results.tsv"
        old = agent.RESULTS_PATH
        try:
            agent.RESULTS_PATH = path
            append_result(
                provider="openai",
                model="gpt-4.1-mini",
                iteration=1,
                tests_ok=True,
                metrics={"sharpe": 1.2, "cagr": 0.11, "max_drawdown": -0.12, "objective": 1.0},
                commit_hash="abc123",
                notes="pass",
            )
            append_result(
                provider="openai",
                model="gpt-4.1-mini",
                iteration=2,
                tests_ok=True,
                metrics={"sharpe": 1.5, "cagr": 0.15, "max_drawdown": -0.09, "objective": 1.6},
                commit_hash="def456",
                notes="pass",
            )
            assert best_objective(path) == 1.6
        finally:
            agent.RESULTS_PATH = old


def test_git_commit_ignores_metrics_json_when_file_is_gitignored(monkeypatch) -> None:
    import run_trading_agent as agent

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "metrics").mkdir()
        (root / ".gitignore").write_text("metrics/*.json\n!metrics/.gitkeep\n", encoding="utf-8")
        (root / "metrics" / ".gitkeep").write_text("", encoding="utf-8")
        (root / "strategy.py").write_text("print('v1')\n", encoding="utf-8")
        (root / "results.tsv").write_text("header\n", encoding="utf-8")
        (root / "metrics" / "latest_metrics.json").write_text("{\"objective\": 1.0}\n", encoding="utf-8")

        old_root = agent.ROOT
        try:
            agent.ROOT = root
            assert run_cmd(["git", "init"], timeout_sec=30).ok
            assert run_cmd(["git", "config", "user.name", "Test Bot"], timeout_sec=30).ok
            assert run_cmd(["git", "config", "user.email", "test@example.com"], timeout_sec=30).ok
            assert run_cmd(["git", "add", ".gitignore", "strategy.py", "results.tsv", "metrics/.gitkeep"], timeout_sec=30).ok
            assert run_cmd(["git", "commit", "-m", "initial"], timeout_sec=30).ok

            (root / "strategy.py").write_text("print('v2')\n", encoding="utf-8")
            (root / "results.tsv").write_text("header\nrow\n", encoding="utf-8")
            (root / "metrics" / "latest_metrics.json").write_text("{\"objective\": 2.0}\n", encoding="utf-8")

            commit_hash = git_commit("test commit")

            assert commit_hash
            status = run_cmd(["git", "status", "--short"], timeout_sec=30)
            assert status.ok
            assert "metrics/latest_metrics.json" not in status.stdout
        finally:
            agent.ROOT = old_root
