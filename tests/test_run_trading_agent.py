from __future__ import annotations

import json
import tempfile
from pathlib import Path
import subprocess

from run_trading_agent import (
    EVOLVABLE_REGION_END,
    EVOLVABLE_REGION_START,
    IMMUTABLE_REGION_END,
    IMMUTABLE_REGION_START,
    append_result,
    best_objective,
    build_candidate_strategy,
    build_prompt,
    compute_stability_score,
    extract_code_block,
    git_commit,
    load_recent_results,
    run_cmd,
    run_verification_loop,
    stable_immutable_region_hash,
    validate_strategy_update,
)


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


def test_validate_strategy_update_rejects_immutable_changes() -> None:
    old = "\n".join(
        [
            IMMUTABLE_REGION_START,
            "header",
            EVOLVABLE_REGION_START,
            "raw = 1",
            EVOLVABLE_REGION_END,
            "footer",
            IMMUTABLE_REGION_END,
        ]
    )
    new = old.replace("footer", "footer changed", 1)

    ok, reason = validate_strategy_update(old, new)

    assert not ok
    assert reason == "immutable_research_surface_changed"


def test_validate_strategy_update_allows_evolvable_changes_only() -> None:
    old = "\n".join(
        [
            IMMUTABLE_REGION_START,
            "header",
            EVOLVABLE_REGION_START,
            "raw = 1",
            EVOLVABLE_REGION_END,
            "footer",
            IMMUTABLE_REGION_END,
        ]
    )
    new = old.replace("raw = 1", "raw = raw.clip(0.0, 1.0)")

    ok, reason = validate_strategy_update(old, new)

    assert ok
    assert reason == ""


def test_build_prompt_mentions_evolvable_region_only() -> None:
    prompt = build_prompt("policy", "strategy", {"objective": 1.0, "sharpe": 0.5, "cagr": 0.1, "max_drawdown": -0.1}, 7)
    assert EVOLVABLE_REGION_START in prompt
    assert EVOLVABLE_REGION_END in prompt
    assert "raw signal logic only" in prompt


def test_stable_immutable_region_hash_ignores_evolvable_body_changes() -> None:
    old = "\n".join(
        [
            IMMUTABLE_REGION_START,
            EVOLVABLE_REGION_START,
            "raw = 1",
            EVOLVABLE_REGION_END,
            "footer",
            IMMUTABLE_REGION_END,
        ]
    )
    new = old.replace("raw = 1", "raw = raw.clip(-1.0, 1.0)")

    assert stable_immutable_region_hash(old) == stable_immutable_region_hash(new)


def test_build_candidate_strategy_splices_evolvable_snippet_into_old_code() -> None:
    old = "\n".join(
        [
            IMMUTABLE_REGION_START,
            "header",
            EVOLVABLE_REGION_START,
            "def generate_raw_signal(df, params):",
            "    return 0.0",
            EVOLVABLE_REGION_END,
            "footer",
            IMMUTABLE_REGION_END,
        ]
    )
    snippet = "\n".join(
        [
            "def generate_raw_signal(df, params):",
            "    return 1.0",
        ]
    )

    built = build_candidate_strategy(old, snippet)

    assert "return 1.0" in built
    assert "header" in built
    assert "footer" in built


def test_build_candidate_strategy_splices_marked_snippet_without_immutable_region() -> None:
    old = "\n".join(
        [
            IMMUTABLE_REGION_START,
            "header",
            EVOLVABLE_REGION_START,
            "def generate_raw_signal(df, params):",
            "    return 0.0",
            EVOLVABLE_REGION_END,
            "footer",
            IMMUTABLE_REGION_END,
        ]
    )
    snippet = "\n".join(
        [
            EVOLVABLE_REGION_START,
            "def generate_raw_signal(df, params):",
            "    return -1.0",
            EVOLVABLE_REGION_END,
        ]
    )

    built = build_candidate_strategy(old, snippet)

    assert built.count(IMMUTABLE_REGION_START) == 1
    assert built.count(IMMUTABLE_REGION_END) == 1
    assert "return -1.0" in built
    assert "header" in built
    assert "footer" in built


def test_build_candidate_strategy_discards_generated_immutable_changes() -> None:
    old = "\n".join(
        [
            IMMUTABLE_REGION_START,
            "original header",
            EVOLVABLE_REGION_START,
            "def generate_raw_signal(df, params):",
            "    return 0.0",
            EVOLVABLE_REGION_END,
            "original footer",
            IMMUTABLE_REGION_END,
        ]
    )
    generated_full_file = "\n".join(
        [
            IMMUTABLE_REGION_START,
            "mutated header",
            EVOLVABLE_REGION_START,
            "def generate_raw_signal(df, params):",
            "    return df['close'].pct_change().fillna(0.0)",
            EVOLVABLE_REGION_END,
            "mutated footer",
            IMMUTABLE_REGION_END,
        ]
    )

    built = build_candidate_strategy(old, generated_full_file)

    assert "mutated header" not in built
    assert "mutated footer" not in built
    assert "original header" in built
    assert "original footer" in built
    assert "pct_change" in built


def test_load_recent_results_and_stability_score() -> None:
    import run_trading_agent as agent

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "results.tsv"
        path.write_text(
            "\n".join(
                [
                    "timestamp\tprovider\tmodel\titeration\ttest_pass\tsharpe\tcagr\tmax_drawdown\tobjective\tgit_commit\tnotes",
                    "t1\topenai\tm\t1\t1\t1\t1\t-0.1\t1.0\tabc\tcommitted",
                    "t2\topenai\tm\t2\t0\t1\t1\t-0.1\t1.0\t\timmutable_research_surface_changed",
                    "t3\topenai\tm\t3\t0\t1\t1\t-0.1\t1.0\t\tprepare_failed",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        rows = load_recent_results(path, limit=2)

    assert len(rows) == 2
    assert rows[0]["iteration"] == "2"
    score = compute_stability_score(rows, objective_improving=True)
    assert 0.0 <= score <= 1.0


def test_run_verification_loop_rejects_hash_failure(monkeypatch) -> None:
    import run_trading_agent as agent

    def fake_run_cmd(cmd, timeout_sec=30):
        return agent.CmdResult(ok=False, code=1, stdout="", stderr="hash mismatch", duration_sec=0.0)

    monkeypatch.setattr(agent, "run_cmd", fake_run_cmd)

    ok, metrics, note = run_verification_loop({"objective": 0.5, "sharpe": 0.2, "cagr": 0.1, "max_drawdown": -0.1})

    assert not ok
    assert note == "immutable_hash_failed"
    assert metrics["objective"] == 0.5


def test_run_verification_loop_accepts_three_finite_backtests(monkeypatch, tmp_path: Path) -> None:
    import run_trading_agent as agent

    metrics_path = tmp_path / "latest_metrics.json"
    payload = {
        "objective": 0.75,
        "test_metrics": {
            "sharpe": 0.7,
            "cagr": 0.1,
            "max_drawdown": -0.12,
            "max_abs_position": 0.9,
        },
    }
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    calls = {"strategy": 0}

    def fake_run_cmd(cmd, timeout_sec=30):
        if cmd[:3] == ["python", "-c", "import strategy; strategy.verify_expected_immutable_hash()"]:
            return agent.CmdResult(ok=True, code=0, stdout="", stderr="", duration_sec=0.0)
        if cmd[:2] == ["pytest", "-q"]:
            return agent.CmdResult(ok=True, code=0, stdout="ok", stderr="", duration_sec=0.0)
        if cmd[:2] == ["python", "strategy.py"]:
            calls["strategy"] += 1
            return agent.CmdResult(ok=True, code=0, stdout="ok", stderr="", duration_sec=0.0)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(agent, "run_cmd", fake_run_cmd)
    monkeypatch.setattr(agent, "METRICS_PATH", metrics_path)

    ok, metrics, note = run_verification_loop({"objective": 0.2, "sharpe": 0.1, "cagr": 0.05, "max_drawdown": -0.1})

    assert ok
    assert note == ""
    assert metrics["objective"] == 0.75
    assert calls["strategy"] == 3
