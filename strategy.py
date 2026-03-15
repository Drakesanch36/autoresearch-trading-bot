from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class StrategyParams:
    fast_window: int = 20
    slow_window: int = 80
    vol_window: int = 20
    vol_target: float = 0.15
    max_leverage: float = 1.5
    fee_bps: float = 2.0
    max_drawdown: float = 0.20


def read_frame(base_path: Path) -> pd.DataFrame:
    parquet_path = base_path.with_suffix(".parquet")
    csv_path = base_path.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)
    raise FileNotFoundError(f"Missing dataset file: {parquet_path} or {csv_path}")


def compute_sharpe(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    rets = returns.dropna()
    if rets.empty:
        return 0.0
    std = rets.std(ddof=0)
    if std <= 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * rets.mean() / std)


def compute_sortino(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    rets = returns.dropna()
    if rets.empty:
        return 0.0
    downside = rets[rets < 0]
    dd = downside.std(ddof=0)
    if np.isnan(dd) or dd <= 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * rets.mean() / dd)


def compute_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    hwm = equity.cummax()
    drawdown = equity / hwm - 1.0
    return float(drawdown.min())


def performance_metrics(returns: pd.Series) -> Dict[str, float]:
    rets = returns.fillna(0.0)
    if rets.empty:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
        }

    equity = (1.0 + rets).cumprod()
    years = max(len(rets) / TRADING_DAYS, 1e-9)
    total_return = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    sharpe = compute_sharpe(rets)
    sortino = compute_sortino(rets)
    max_dd = compute_max_drawdown(equity)
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0
    win_rate = float((rets > 0).mean())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
    }


def generate_signals(df: pd.DataFrame, params: StrategyParams) -> pd.Series:
    close = df["close"].astype(float)
    ret_1 = close.pct_change().fillna(0.0)

    fast = close.rolling(params.fast_window).mean()
    slow = close.rolling(params.slow_window).mean()
    trend = (fast > slow).astype(float)

    realized_vol = ret_1.rolling(params.vol_window).std() * np.sqrt(TRADING_DAYS)
    target_leverage = (params.vol_target / realized_vol.replace(0, np.nan)).clip(0, params.max_leverage)
    target_leverage = target_leverage.fillna(0.0)

    raw_position = trend * target_leverage

    # Shift by 1 bar to avoid lookahead.
    return raw_position.shift(1).fillna(0.0)


def apply_drawdown_kill_switch(returns: pd.Series, max_drawdown: float) -> pd.Series:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    dd = equity / equity.cummax() - 1.0
    breached = (dd < -abs(max_drawdown)).cummax()
    return returns.where(~breached, 0.0)


def backtest(df: pd.DataFrame, params: StrategyParams) -> Dict[str, object]:
    close = df["close"].astype(float)
    asset_ret = close.pct_change().fillna(0.0)
    position = generate_signals(df, params)

    gross_ret = position * asset_ret
    turnover = position.diff().abs().fillna(position.abs())
    fees = turnover * (params.fee_bps / 10_000.0)
    net_ret = gross_ret - fees
    net_ret = apply_drawdown_kill_switch(net_ret, params.max_drawdown)

    equity = (1.0 + net_ret).cumprod()
    metrics = performance_metrics(net_ret)
    metrics["trades"] = int((position.diff().abs() > 1e-12).sum())

    return {
        "returns": net_ret,
        "position": position,
        "equity": equity,
        "metrics": metrics,
        "params": asdict(params),
    }


def parameter_grid() -> Iterable[StrategyParams]:
    for fast in (10, 20, 30):
        for slow in (40, 60, 90):
            if fast >= slow:
                continue
            for vol_target in (0.10, 0.15, 0.20):
                yield StrategyParams(
                    fast_window=fast,
                    slow_window=slow,
                    vol_window=20,
                    vol_target=vol_target,
                    max_leverage=1.5,
                    fee_bps=2.0,
                    max_drawdown=0.20,
                )


def objective(metrics: Dict[str, float]) -> float:
    sharpe = float(metrics.get("sharpe", 0.0))
    cagr = float(metrics.get("cagr", 0.0))
    max_dd = abs(float(metrics.get("max_drawdown", 0.0)))
    penalty = max(0.0, max_dd - 0.20) * 5.0
    return sharpe + 0.5 * cagr - penalty


def optimize_params(train: pd.DataFrame, valid: pd.DataFrame) -> Tuple[StrategyParams, Dict[str, float]]:
    best_params = StrategyParams()
    best_metrics: Dict[str, float] = {"sharpe": -1e9, "cagr": -1e9, "max_drawdown": -1.0}
    best_score = -1e9

    for candidate in parameter_grid():
        train_metrics = backtest(train, candidate)["metrics"]
        valid_metrics = backtest(valid, candidate)["metrics"]

        combined = {
            "sharpe": 0.4 * train_metrics["sharpe"] + 0.6 * valid_metrics["sharpe"],
            "cagr": 0.4 * train_metrics["cagr"] + 0.6 * valid_metrics["cagr"],
            "max_drawdown": min(train_metrics["max_drawdown"], valid_metrics["max_drawdown"]),
        }
        score = objective(combined)
        if score > best_score:
            best_score = score
            best_params = candidate
            best_metrics = combined

    return best_params, best_metrics


def walk_forward_validate(
    df: pd.DataFrame,
    train_size: int = 252 * 2,
    valid_size: int = 126,
    test_size: int = 126,
) -> Dict[str, object]:
    folds: List[Dict[str, object]] = []
    start = 0

    while start + train_size + valid_size + test_size <= len(df):
        tr = df.iloc[start : start + train_size]
        va = df.iloc[start + train_size : start + train_size + valid_size]
        te = df.iloc[start + train_size + valid_size : start + train_size + valid_size + test_size]

        params, val_metrics = optimize_params(tr, va)
        test_run = backtest(te, params)

        folds.append(
            {
                "start": str(tr.index[0]),
                "end": str(te.index[-1]),
                "params": asdict(params),
                "validation": val_metrics,
                "test": test_run["metrics"],
            }
        )
        start += test_size

    if not folds:
        return {
            "fold_count": 0,
            "avg_test_sharpe": 0.0,
            "avg_test_cagr": 0.0,
            "worst_test_drawdown": 0.0,
            "folds": [],
        }

    sharpe_vals = [f["test"]["sharpe"] for f in folds]
    cagr_vals = [f["test"]["cagr"] for f in folds]
    mdd_vals = [f["test"]["max_drawdown"] for f in folds]

    return {
        "fold_count": len(folds),
        "avg_test_sharpe": float(np.mean(sharpe_vals)),
        "avg_test_cagr": float(np.mean(cagr_vals)),
        "worst_test_drawdown": float(np.min(mdd_vals)),
        "folds": folds,
    }


def run_strategy(data_dir: Path, output_path: Path) -> Dict[str, object]:
    train = read_frame(data_dir / "train")
    valid = read_frame(data_dir / "valid")
    test = read_frame(data_dir / "test")

    best_params, blend_metrics = optimize_params(train, valid)
    test_result = backtest(test, best_params)

    full = pd.concat([train, valid, test]).sort_index()
    wf = walk_forward_validate(full)

    summary = {
        "selection_metrics": blend_metrics,
        "best_params": asdict(best_params),
        "test_metrics": test_result["metrics"],
        "walk_forward": wf,
        "objective": objective(test_result["metrics"]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run vectorized backtest and walk-forward validation")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="metrics/latest_metrics.json")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_strategy(data_dir=Path(args.data_dir), output_path=Path(args.output))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
