from __future__ import annotations

import tempfile
from pathlib import Path
import subprocess

from run_trading_agent import append_result, best_objective, extract_code_block, run_cmd


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
