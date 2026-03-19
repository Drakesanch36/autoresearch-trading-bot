from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import strategy
from strategy import (
    DEFAULT_RISK_POLICY,
    EVOLVABLE_REGION_END,
    EVOLVABLE_REGION_START,
    EXPECTED_IMMUTABLE_HASH,
    IMMUTABLE_REGION_END,
    IMMUTABLE_REGION_START,
    REJECTED_SCORE,
    RiskPolicy,
    StrategyParams,
    average_holding_period,
    backtest,
    compute_immutable_region_hash_from_text,
    compute_max_drawdown,
    compute_target_leverage,
    compute_sharpe,
    evaluate_guardrails,
    generate_raw_signal,
    generate_signals,
    generate_time_series_folds,
    objective,
    optimize_params,
    parameter_grid,
    performance_metrics,
    run_strategy,
    summarize_validation_runs,
    verify_expected_immutable_hash,
    walk_forward_validate,
)


ROOT = Path(__file__).resolve().parent.parent


def make_prices(rows: int = 900, *, slope: float = 80.0) -> pd.DataFrame:
    idx = pd.date_range("2018-01-01", periods=rows, freq="D", tz="UTC")
    trend = np.linspace(100.0, 100.0 + slope, rows)
    cycle = np.sin(np.arange(rows) / 12.0) * 4.0
    noise = np.random.default_rng(7).normal(0.0, 0.4, rows).cumsum() * 0.05
    close = trend + cycle + noise

    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": np.full(rows, 1_500_000.0),
        },
        index=idx,
    )


def strategy_source() -> str:
    return (ROOT / "strategy.py").read_text(encoding="utf-8")


def extract_immutable_text(text: str) -> str:
    start = re.search(rf"^{re.escape(IMMUTABLE_REGION_START)}$", text, flags=re.MULTILINE)
    end = re.search(rf"^{re.escape(IMMUTABLE_REGION_END)}$", text, flags=re.MULTILINE)
    assert start is not None
    assert end is not None
    start_index = start.start()
    end_index = end.end()
    return text[start_index:end_index]


def stable_immutable_hash(text: str) -> str:
    immutable_text = extract_immutable_text(text)
    start = re.search(rf"^{re.escape(EVOLVABLE_REGION_START)}$", immutable_text, flags=re.MULTILINE)
    end = re.search(rf"^{re.escape(EVOLVABLE_REGION_END)}$", immutable_text, flags=re.MULTILINE)
    assert start is not None
    assert end is not None
    masked = immutable_text[: start.start()] + "<EVOLVABLE_REGION>" + immutable_text[end.end() :]
    return hashlib.sha256(masked.encode("utf-8")).hexdigest()


def test_expected_immutable_hash_matches_current_source() -> None:
    source = strategy_source()
    assert compute_immutable_region_hash_from_text(source) == EXPECTED_IMMUTABLE_HASH


def test_verify_expected_immutable_hash_raises_on_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(strategy, "compute_immutable_region_hash_from_text", lambda text: "bad-hash")

    with pytest.raises(RuntimeError, match="Immutable strategy kernel hash mismatch"):
        verify_expected_immutable_hash()


def test_compute_sharpe_known_value() -> None:
    rets = pd.Series(np.full(252, 0.001))
    assert compute_sharpe(rets) == 0.0

    rets2 = pd.Series([0.01, -0.005] * 126)
    assert compute_sharpe(rets2) != 0.0


def test_compute_max_drawdown() -> None:
    equity = pd.Series([1.0, 1.2, 1.1, 1.3, 0.9, 1.0])
    mdd = compute_max_drawdown(equity)
    assert np.isclose(mdd, -0.3076923077)


def test_generate_raw_signal_is_bounded() -> None:
    data = make_prices(400)
    raw = generate_raw_signal(data, StrategyParams(fast_window=15, slow_window=55))
    assert raw.index.equals(data.index)
    assert raw.dropna().between(-1.0, 1.0).all()


def test_final_exposure_is_determined_outside_evolvable_block() -> None:
    data = make_prices(400)
    params = StrategyParams(fast_window=15, slow_window=55)
    exposure = generate_signals(data, params)
    target_leverage = compute_target_leverage(data, DEFAULT_RISK_POLICY)
    raw_signal = generate_raw_signal(data, params)
    expected = (raw_signal * target_leverage).shift(1).fillna(0.0)
    pd.testing.assert_series_equal(exposure, expected)


def test_target_leverage_independent_of_raw_signal_shape() -> None:
    data = make_prices(400)
    base_leverage = compute_target_leverage(data, DEFAULT_RISK_POLICY)

    original = strategy.generate_raw_signal
    try:
        strategy.generate_raw_signal = lambda df, params: pd.Series(1.0, index=df.index)
        alt_exposure = generate_signals(data, StrategyParams())
    finally:
        strategy.generate_raw_signal = original

    expected = base_leverage.shift(1).fillna(0.0).rename(None)
    pd.testing.assert_series_equal(alt_exposure, expected)


def test_generate_signals_shifted() -> None:
    data = make_prices(400)
    sig = generate_signals(data, StrategyParams(fast_window=15, slow_window=55))
    assert sig.iloc[0] == 0.0
    assert sig.index.equals(data.index)


def test_average_holding_period_handles_no_positions() -> None:
    position = pd.Series([0.0, 0.0, 0.0])
    assert average_holding_period(position, 0.05) == 0.0


def test_backtest_output_and_drawdown_bounds() -> None:
    data = make_prices(500)
    out = backtest(data, StrategyParams())
    metrics = out["metrics"]
    required = {
        "total_return",
        "cagr",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "win_rate",
        "trades",
        "annualized_volatility",
        "max_abs_position",
        "mean_abs_exposure",
        "percent_active_days",
        "turnover",
        "average_holding_period",
        "guardrails_passed",
    }
    assert required.issubset(metrics.keys())
    assert metrics["max_drawdown"] <= 0.0
    assert isinstance(metrics["trades"], int)
    assert metrics["max_abs_position"] <= DEFAULT_RISK_POLICY.max_leverage + 1e-12


def test_near_flat_strategy_is_rejected_by_guardrails() -> None:
    data = make_prices(500)
    original = strategy.generate_raw_signal
    try:
        strategy.generate_raw_signal = lambda df, params: pd.Series(0.0, index=df.index)
        out = backtest(data, StrategyParams())
    finally:
        strategy.generate_raw_signal = original

    assert not out["guardrails"]["passed"]
    assert objective(out["metrics"]) == REJECTED_SCORE


def test_valid_simple_strategy_passes_guardrails() -> None:
    data = make_prices(700, slope=120.0)
    out = backtest(data, StrategyParams(fast_window=15, slow_window=55))
    assert out["guardrails"]["passed"]
    assert objective(out["metrics"]) > REJECTED_SCORE


def test_objective_rejects_non_finite_inputs() -> None:
    metrics = {"sharpe": float("nan"), "cagr": 0.1, "max_drawdown": -0.1, "guardrails_passed": True}
    assert objective(metrics) == REJECTED_SCORE


def test_parameter_grid_excludes_risk_controls() -> None:
    params = list(parameter_grid())
    assert params
    assert all(p.fast_window < p.slow_window for p in params)
    assert all(asdict_keys == {"fast_window", "slow_window"} for asdict_keys in (set(p.__dict__.keys()) for p in params))


def test_generate_time_series_folds_are_chronological() -> None:
    data = make_prices(1200)
    folds = generate_time_series_folds(data, train_size=300, valid_size=120, step_size=60, offsets=(0, 30))
    assert folds
    for fold in folds:
        train = fold["train"]
        valid = fold["valid"]
        assert train.index.max() < valid.index.min()
        assert train.index.is_monotonic_increasing
        assert valid.index.is_monotonic_increasing


def test_summarize_validation_runs_uses_robust_statistics() -> None:
    runs = [
        {"validation": {"max_drawdown": -0.10, "guardrails_passed": True}, "score": 0.8},
        {"validation": {"max_drawdown": -0.12, "guardrails_passed": True}, "score": 0.6},
        {"validation": {"max_drawdown": -0.08, "guardrails_passed": True}, "score": 0.9},
    ]
    summary = summarize_validation_runs(runs)
    assert np.isclose(summary["median_fold_score"], 0.8)
    assert np.isclose(summary["lower_quartile_fold_score"], 0.7)
    assert np.isclose(summary["selection_score"], 0.7)


def test_optimize_params_returns_robust_selection_summary() -> None:
    data = make_prices(900)
    train = data.iloc[:500]
    valid = data.iloc[500:800]
    params, metrics = optimize_params(train, valid)
    assert isinstance(params, StrategyParams)
    assert {"fold_count", "median_fold_score", "lower_quartile_fold_score", "selection_score", "guardrails_passed"} <= set(metrics)


def test_run_strategy_uses_test_as_lockbox_and_writes_summary(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data = make_prices(1300, slope=140.0)
    data.iloc[:700].to_csv(data_dir / "train.csv")
    data.iloc[700:1000].to_csv(data_dir / "valid.csv")
    data.iloc[1000:].to_csv(data_dir / "test.csv")

    out_path = tmp_path / "metrics" / "latest_metrics.json"
    summary = run_strategy(data_dir, out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path.exists()
    assert written["objective"] == summary["objective"]
    assert "selection_metrics" in summary
    assert "test_metrics" in summary
    assert "walk_forward" in summary
    assert summary["test_metrics"]["guardrails_passed"]


def test_walk_forward_returns_robust_fold_summary() -> None:
    data = make_prices(1400)
    wf = walk_forward_validate(data, train_size=300, valid_size=120, step_size=120)
    assert wf["fold_count"] > 0
    assert "median_fold_result" in wf
    assert "worst_fold_result" in wf


def test_strategy_source_contains_region_markers() -> None:
    text = strategy_source()
    assert EVOLVABLE_REGION_START in text
    assert EVOLVABLE_REGION_END in text
    assert IMMUTABLE_REGION_START in text
    assert IMMUTABLE_REGION_END in text


def test_stable_immutable_hash_matches_fixture() -> None:
    text = strategy_source()
    actual = stable_immutable_hash(text)
    fixture_path = ROOT / "tests" / "fixtures" / "strategy_immutable_region.sha256"
    expected = fixture_path.read_text(encoding="utf-8").strip()
    assert actual == expected
