from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from strategy import (
    EVOLVABLE_REGION_END,
    EVOLVABLE_REGION_START,
    StrategyParams,
    backtest,
    compute_max_drawdown,
    compute_sharpe,
    generate_raw_signal,
    generate_signals,
    walk_forward_validate,
)


def make_prices(rows: int = 900) -> pd.DataFrame:
    idx = pd.date_range("2018-01-01", periods=rows, freq="D", tz="UTC")
    trend = np.linspace(100.0, 180.0, rows)
    cycle = np.sin(np.arange(rows) / 15.0) * 1.5
    close = trend + cycle

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


def test_compute_sharpe_known_value() -> None:
    rets = pd.Series(np.full(252, 0.001))
    # zero std -> guarded to 0
    assert compute_sharpe(rets) == 0.0

    rets2 = pd.Series([0.01, -0.005] * 126)
    assert compute_sharpe(rets2) != 0.0


def test_generate_signals_shifted() -> None:
    data = make_prices(400)
    params = StrategyParams(fast_window=10, slow_window=30)
    sig = generate_signals(data, params)

    assert sig.iloc[0] == 0.0
    assert sig.index.equals(data.index)


def test_generate_raw_signal_is_bounded_binary_trend() -> None:
    data = make_prices(400)
    params = StrategyParams(fast_window=10, slow_window=30)
    raw = generate_raw_signal(data, params)

    assert raw.index.equals(data.index)
    assert set(raw.dropna().unique()).issubset({0.0, 1.0})


def test_strategy_source_contains_evolvable_markers() -> None:
    text = Path(__file__).resolve().parent.parent.joinpath("strategy.py").read_text(encoding="utf-8")
    assert EVOLVABLE_REGION_START in text
    assert EVOLVABLE_REGION_END in text


def test_backtest_output_and_drawdown_bounds() -> None:
    data = make_prices(500)
    out = backtest(data, StrategyParams())
    m = out["metrics"]

    required = {"total_return", "cagr", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate", "trades"}
    assert required.issubset(m.keys())
    assert m["max_drawdown"] <= 0.0
    assert isinstance(m["trades"], int)


def test_compute_max_drawdown() -> None:
    equity = pd.Series([1.0, 1.2, 1.1, 1.3, 0.9, 1.0])
    mdd = compute_max_drawdown(equity)
    assert np.isclose(mdd, -0.3076923077)


def test_walk_forward_returns_folds() -> None:
    data = make_prices(1200)
    wf = walk_forward_validate(data, train_size=300, valid_size=120, test_size=120)
    assert wf["fold_count"] > 0
    assert len(wf["folds"]) == wf["fold_count"]
    assert "avg_test_sharpe" in wf
