from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252
REJECTED_SCORE = -1e9
EVOLVABLE_REGION_START = "# === EVOLVABLE REGION START ==="
EVOLVABLE_REGION_END = "# === EVOLVABLE REGION END ==="
IMMUTABLE_REGION_START = "# === IMMUTABLE REGION START ==="
IMMUTABLE_REGION_END = "# === IMMUTABLE REGION END ==="


@dataclass(frozen=True)
class StrategyParams:
    fast_window: int = 20
    slow_window: int = 80


@dataclass(frozen=True)
class RiskPolicy:
    vol_window: int = 20
    vol_target: float = 0.12
    max_leverage: float = 1.0
    fee_bps: float = 2.0
    max_drawdown: float = 0.20
    min_mean_abs_exposure: float = 0.05
    min_active_day_pct: float = 0.10
    min_annualized_volatility: float = 0.01
    min_trades: int = 4
    min_turnover: float = 2.0
    active_exposure_threshold: float = 0.05


DEFAULT_RISK_POLICY = RiskPolicy()

# === IMMUTABLE REGION START ===


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
            "annualized_volatility": 0.0,
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
    annualized_volatility = float(rets.std(ddof=0) * np.sqrt(TRADING_DAYS))

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "annualized_volatility": annualized_volatility,
    }


# === EVOLVABLE REGION START ===
def generate_raw_signal(df: pd.DataFrame, params: StrategyParams) -> pd.Series:
    close = df["close"].astype(float)
    fast = close.rolling(params.fast_window, min_periods=params.fast_window).mean()
    slow = close.rolling(params.slow_window, min_periods=params.slow_window).mean()
    signal = np.sign(fast - slow)
    return signal.fillna(0.0).clip(-1.0, 1.0)


# === EVOLVABLE REGION END ===


def compute_target_leverage(df: pd.DataFrame, risk_policy: RiskPolicy = DEFAULT_RISK_POLICY) -> pd.Series:
    close = df["close"].astype(float)
    ret_1 = close.pct_change().fillna(0.0)
    realized_vol = ret_1.rolling(risk_policy.vol_window, min_periods=risk_policy.vol_window).std()
    realized_vol = realized_vol * np.sqrt(TRADING_DAYS)
    target_leverage = (risk_policy.vol_target / realized_vol.replace(0, np.nan)).clip(0, risk_policy.max_leverage)
    return target_leverage.fillna(0.0)


def generate_signals(df: pd.DataFrame, params: StrategyParams, risk_policy: RiskPolicy = DEFAULT_RISK_POLICY) -> pd.Series:
    raw_signal = generate_raw_signal(df, params)
    target_leverage = compute_target_leverage(df, risk_policy)
    raw_position = (raw_signal * target_leverage).clip(-risk_policy.max_leverage, risk_policy.max_leverage)
    return raw_position.shift(1).fillna(0.0)


def apply_drawdown_kill_switch(returns: pd.Series, max_drawdown: float) -> pd.Series:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    dd = equity / equity.cummax() - 1.0
    breached = (dd < -abs(max_drawdown)).cummax()
    return returns.where(~breached, 0.0)


def average_holding_period(position: pd.Series, exposure_threshold: float) -> float:
    active = position.abs() >= exposure_threshold
    if not active.any():
        return 0.0

    runs: List[int] = []
    current = 0
    for flag in active:
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return float(np.mean(runs)) if runs else 0.0


def evaluate_guardrails(
    position: pd.Series,
    returns: pd.Series,
    turnover: pd.Series,
    metrics: Dict[str, float],
    risk_policy: RiskPolicy = DEFAULT_RISK_POLICY,
) -> Dict[str, object]:
    mean_abs_exposure = float(position.abs().mean())
    active_days = position.abs() >= risk_policy.active_exposure_threshold
    percent_active_days = float(active_days.mean())
    annualized_volatility = float(metrics.get("annualized_volatility", 0.0))
    total_turnover = float(turnover.sum())
    trade_count = int((position.diff().abs() > 1e-12).sum())
    avg_holding_period = average_holding_period(position, risk_policy.active_exposure_threshold)

    checks = {
        "mean_abs_exposure": mean_abs_exposure >= risk_policy.min_mean_abs_exposure,
        "percent_active_days": percent_active_days >= risk_policy.min_active_day_pct,
        "annualized_volatility": annualized_volatility >= risk_policy.min_annualized_volatility,
        "finite_objective_inputs": all(
            np.isfinite(
                [
                    float(metrics.get("sharpe", 0.0)),
                    float(metrics.get("cagr", 0.0)),
                    float(metrics.get("max_drawdown", 0.0)),
                ]
            )
        ),
        "minimum_trades_or_turnover": trade_count >= risk_policy.min_trades or total_turnover >= risk_policy.min_turnover,
    }

    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "mean_abs_exposure": mean_abs_exposure,
        "percent_active_days": percent_active_days,
        "annualized_volatility": annualized_volatility,
        "turnover": total_turnover,
        "trade_count": trade_count,
        "average_holding_period": avg_holding_period,
    }


def backtest(df: pd.DataFrame, params: StrategyParams, risk_policy: RiskPolicy = DEFAULT_RISK_POLICY) -> Dict[str, object]:
    close = df["close"].astype(float)
    asset_ret = close.pct_change().fillna(0.0)
    position = generate_signals(df, params, risk_policy)

    gross_ret = position * asset_ret
    turnover = position.diff().abs().fillna(position.abs())
    fees = turnover * (risk_policy.fee_bps / 10_000.0)
    net_ret = gross_ret - fees
    net_ret = apply_drawdown_kill_switch(net_ret, risk_policy.max_drawdown)

    equity = (1.0 + net_ret).cumprod()
    metrics = performance_metrics(net_ret)
    guardrails = evaluate_guardrails(position, net_ret, turnover, metrics, risk_policy)
    metrics.update(
        {
            "trades": guardrails["trade_count"],
            "turnover": guardrails["turnover"],
            "mean_abs_exposure": guardrails["mean_abs_exposure"],
            "percent_active_days": guardrails["percent_active_days"],
            "average_holding_period": guardrails["average_holding_period"],
            "guardrails_passed": guardrails["passed"],
        }
    )

    return {
        "returns": net_ret,
        "position": position,
        "equity": equity,
        "metrics": metrics,
        "params": asdict(params),
        "guardrails": guardrails,
    }


def parameter_grid() -> Iterable[StrategyParams]:
    for fast in (15, 20, 30):
        for slow in (55, 75, 95):
            if fast >= slow:
                continue
            yield StrategyParams(
                fast_window=fast,
                slow_window=slow,
            )


def objective(metrics: Dict[str, float]) -> float:
    if not bool(metrics.get("guardrails_passed", True)):
        return REJECTED_SCORE

    sharpe = float(metrics.get("sharpe", 0.0))
    cagr = float(metrics.get("cagr", 0.0))
    max_dd = abs(float(metrics.get("max_drawdown", 0.0)))
    if not np.isfinite([sharpe, cagr, max_dd]).all():
        return REJECTED_SCORE

    score = sharpe + 0.5 * cagr - max(0.0, max_dd - 0.20) * 5.0
    if not np.isfinite(score) or score < -100.0 or score > 100.0:
        return REJECTED_SCORE
    return float(score)


def generate_time_series_folds(
    df: pd.DataFrame,
    train_size: int = 252 * 2,
    valid_size: int = 126,
    step_size: int = 63,
    offsets: Tuple[int, ...] = (0, 63),
) -> List[Dict[str, object]]:
    folds: List[Dict[str, object]] = []
    seen: set[Tuple[pd.Timestamp, pd.Timestamp]] = set()

    for offset in offsets:
        start = offset
        while start + train_size + valid_size <= len(df):
            train = df.iloc[start : start + train_size]
            valid = df.iloc[start + train_size : start + train_size + valid_size]
            if train.empty or valid.empty:
                break
            key = (train.index[0], valid.index[-1])
            if key not in seen:
                folds.append(
                    {
                        "train": train,
                        "valid": valid,
                        "train_start": str(train.index[0]),
                        "train_end": str(train.index[-1]),
                        "valid_start": str(valid.index[0]),
                        "valid_end": str(valid.index[-1]),
                    }
                )
                seen.add(key)
            start += step_size

    return folds


def summarize_validation_runs(fold_runs: List[Dict[str, object]]) -> Dict[str, object]:
    if not fold_runs:
        return {
            "fold_count": 0,
            "median_fold_score": REJECTED_SCORE,
            "lower_quartile_fold_score": REJECTED_SCORE,
            "worst_fold_drawdown": -1.0,
            "selection_score": REJECTED_SCORE,
            "guardrails_passed": False,
            "folds": [],
        }

    scores = [float(run["score"]) for run in fold_runs]
    drawdowns = [float(run["validation"]["max_drawdown"]) for run in fold_runs]
    guardrails_passed = all(bool(run["validation"].get("guardrails_passed", False)) for run in fold_runs)
    median_fold_score = float(np.median(scores))
    lower_quartile_fold_score = float(np.quantile(scores, 0.25))
    worst_fold_drawdown = float(np.min(drawdowns))
    selection_score = REJECTED_SCORE if not guardrails_passed else min(median_fold_score, lower_quartile_fold_score)

    return {
        "fold_count": len(fold_runs),
        "median_fold_score": median_fold_score,
        "lower_quartile_fold_score": lower_quartile_fold_score,
        "worst_fold_drawdown": worst_fold_drawdown,
        "selection_score": selection_score,
        "guardrails_passed": guardrails_passed,
        "folds": fold_runs,
    }


def optimize_params(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    risk_policy: RiskPolicy = DEFAULT_RISK_POLICY,
) -> Tuple[StrategyParams, Dict[str, object]]:
    selection_df = pd.concat([train, valid]).sort_index()
    folds = generate_time_series_folds(selection_df)

    best_params = StrategyParams()
    best_summary = summarize_validation_runs([])
    best_score = REJECTED_SCORE

    for candidate in parameter_grid():
        fold_runs: List[Dict[str, object]] = []
        for fold in folds:
            validation_result = backtest(fold["valid"], candidate, risk_policy)
            validation_metrics = validation_result["metrics"]
            score = objective(validation_metrics)
            fold_runs.append(
                {
                    "train_start": fold["train_start"],
                    "train_end": fold["train_end"],
                    "valid_start": fold["valid_start"],
                    "valid_end": fold["valid_end"],
                    "validation": validation_metrics,
                    "score": score,
                }
            )

        summary = summarize_validation_runs(fold_runs)
        if summary["selection_score"] > best_score:
            best_score = float(summary["selection_score"])
            best_params = candidate
            best_summary = summary

    return best_params, best_summary


def walk_forward_validate(
    df: pd.DataFrame,
    train_size: int = 252 * 2,
    valid_size: int = 126,
    step_size: int = 63,
    risk_policy: RiskPolicy = DEFAULT_RISK_POLICY,
) -> Dict[str, object]:
    folds = generate_time_series_folds(df, train_size=train_size, valid_size=valid_size, step_size=step_size)
    summarized_folds: List[Dict[str, object]] = []

    for fold in folds:
        params, validation_summary = optimize_params(fold["train"], fold["valid"], risk_policy)
        test_result = backtest(fold["valid"], params, risk_policy)
        summarized_folds.append(
            {
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "valid_start": fold["valid_start"],
                "valid_end": fold["valid_end"],
                "params": asdict(params),
                "validation_summary": validation_summary,
                "test": test_result["metrics"],
            }
        )

    if not summarized_folds:
        return {
            "fold_count": 0,
            "median_fold_result": 0.0,
            "worst_fold_result": 0.0,
            "worst_test_drawdown": 0.0,
            "folds": [],
        }

    fold_results = [objective(fold["test"]) for fold in summarized_folds]
    worst_drawdowns = [float(fold["test"]["max_drawdown"]) for fold in summarized_folds]
    return {
        "fold_count": len(summarized_folds),
        "median_fold_result": float(np.median(fold_results)),
        "worst_fold_result": float(np.min(fold_results)),
        "worst_test_drawdown": float(np.min(worst_drawdowns)),
        "folds": summarized_folds,
    }


def run_strategy(data_dir: Path, output_path: Path, risk_policy: RiskPolicy = DEFAULT_RISK_POLICY) -> Dict[str, object]:
    train = read_frame(data_dir / "train")
    valid = read_frame(data_dir / "valid")
    test = read_frame(data_dir / "test")

    best_params, selection_metrics = optimize_params(train, valid, risk_policy)
    lockbox_result = backtest(test, best_params, risk_policy)

    full = pd.concat([train, valid, test]).sort_index()
    wf = walk_forward_validate(full, risk_policy=risk_policy)

    summary = {
        "selection_metrics": selection_metrics,
        "best_params": asdict(best_params),
        "test_metrics": lockbox_result["metrics"],
        "walk_forward": wf,
        "objective": objective(lockbox_result["metrics"]),
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


# === IMMUTABLE REGION END ===
