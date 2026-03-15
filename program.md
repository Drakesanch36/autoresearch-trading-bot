Every edit to strategy.py MUST keep all tests in tests/test_strategy.py passing. If a change breaks any test, reject it immediately.

# Autonomous Trading Research Program

## Mission
Evolve `strategy.py` into a robust, out-of-sample profitable strategy using only changes to `strategy.py`.

## Hard Constraints
1. `strategy.py` is the only file the agent may modify.
2. Never introduce lookahead bias.
3. Preserve walk-forward validation integrity.
4. Keep drawdown controlled (`max_drawdown` should stay above -0.25).
5. The objective favors high Sharpe, stable CAGR, and low drawdown.

## Mandatory Validation Per Iteration
1. Run `python prepare.py --symbol SPY --source yfinance --timeframe 1d --start 2016-01-01 --out-dir data`
2. Run `python strategy.py --data-dir data --output metrics/latest_metrics.json`
3. Run `pytest tests/ -q`

If any command fails, discard the proposal.

## Optimization Targets
1. Improve out-of-sample Sharpe ratio.
2. Keep max drawdown better than previous best.
3. Avoid fragile overfit behavior by preferring parameter robustness over single-split gains.

## Allowed Strategy Levers
1. Signal design (trend, mean-reversion, volatility scaling)
2. Position sizing and leverage controls
3. Fee and turnover-aware logic
4. Better internal parameter search across train/valid

## Disallowed Behaviors
1. Editing `prepare.py`, `program.md`, or any test files.
2. Disabling tests or reducing test assertions.
3. Using future data at decision time.
4. Removing risk controls.

## Commit Rule
Only commit if:
1. `pytest tests/ -q` passes.
2. The new objective score beats the prior best in `results.tsv`.
