from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

try:
    import ccxt
except Exception:  # pragma: no cover
    ccxt = None

REQUIRED_OHLCV = ["open", "high", "low", "close", "volume"]


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common OHLCV column naming and enforce datetime index."""
    work = df.copy()

    if isinstance(work.columns, pd.MultiIndex):
        work.columns = [c[0] if isinstance(c, tuple) else c for c in work.columns]

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "close",
        "Volume": "volume",
        "Datetime": "timestamp",
        "Date": "timestamp",
        "Timestamp": "timestamp",
    }
    work = work.rename(columns=rename_map)
    work.columns = [str(c).lower() for c in work.columns]
    # yfinance can expose both Adj Close and Close, which collide after rename.
    work = work.loc[:, ~pd.Index(work.columns).duplicated(keep="last")]

    if "timestamp" in work.columns:
        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
        work = work.set_index("timestamp")

    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index, utc=True)

    work = work.sort_index()
    return work


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    work = normalize_ohlcv_columns(df)
    missing = [c for c in REQUIRED_OHLCV if c not in work.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    work = work[REQUIRED_OHLCV].replace([np.inf, -np.inf], np.nan).dropna()
    work = work[(work["high"] >= work["low"]) & (work["volume"] >= 0)]
    return work


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute non-leaky features and one-step-ahead target return."""
    work = df.copy()
    close = work["close"]

    work["ret_1"] = close.pct_change()
    work["ret_5"] = close.pct_change(5)
    work["vol_20"] = work["ret_1"].rolling(20).std()
    work["ma_10"] = close.rolling(10).mean()
    work["ma_30"] = close.rolling(30).mean()
    work["ma_ratio"] = work["ma_10"] / work["ma_30"]
    roll_mean = close.rolling(20).mean()
    roll_std = close.rolling(20).std()
    work["zscore_20"] = (close - roll_mean) / roll_std

    # Target is intentionally forward-looking, features above are not.
    work["target_ret_1d"] = close.pct_change().shift(-1)

    return work.dropna().copy()


def split_walk_forward(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    valid_frac: float = 0.2,
) -> Dict[str, pd.DataFrame]:
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be in (0, 1)")
    if not 0 < valid_frac < 1:
        raise ValueError("valid_frac must be in (0, 1)")
    if train_frac + valid_frac >= 1:
        raise ValueError("train_frac + valid_frac must be < 1")

    n = len(df)
    if n < 100:
        raise ValueError("Need at least 100 rows for stable splits")

    train_end = int(n * train_frac)
    valid_end = train_end + int(n * valid_frac)

    train = df.iloc[:train_end].copy()
    valid = df.iloc[train_end:valid_end].copy()
    test = df.iloc[valid_end:].copy()

    if train.empty or valid.empty or test.empty:
        raise ValueError("One of train/valid/test splits is empty")

    return {"train": train, "valid": valid, "test": test}


def fetch_yfinance(
    symbol: str,
    start: str,
    end: Optional[str],
    interval: str,
) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed")

    raw = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"No yfinance data returned for {symbol}")
    return clean_ohlcv(raw)


def _timeframe_ms(timeframe: str) -> int:
    mapping = {
        "1m": 60_000,
        "5m": 300_000,
        "15m": 900_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }
    if timeframe not in mapping:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return mapping[timeframe]


def fetch_ccxt(
    symbol: str,
    exchange_name: str,
    start: str,
    end: Optional[str],
    timeframe: str,
    limit: int = 1000,
) -> pd.DataFrame:
    if ccxt is None:
        raise RuntimeError("ccxt is not installed")

    exchange_cls = getattr(ccxt, exchange_name)
    exchange = exchange_cls({"enableRateLimit": True})

    since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else None
    step_ms = _timeframe_ms(timeframe)

    rows = []
    while True:
        chunk = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not chunk:
            break
        rows.extend(chunk)
        last_ts = chunk[-1][0]
        since = last_ts + step_ms
        if end_ms and since >= end_ms:
            break
        if len(chunk) < limit:
            break

    if not rows:
        raise RuntimeError(f"No ccxt data returned for {symbol}")

    raw = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], unit="ms", utc=True)
    raw = raw.set_index("timestamp")

    if end:
        raw = raw[raw.index <= pd.Timestamp(end, tz="UTC")]

    return clean_ohlcv(raw)


def save_frame(df: pd.DataFrame, parquet_path: Path) -> Path:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(parquet_path)
        return parquet_path
    except Exception:
        csv_path = parquet_path.with_suffix(".csv")
        df.to_csv(csv_path)
        return csv_path


def save_splits(
    splits: Dict[str, pd.DataFrame],
    out_dir: Path,
    metadata: Dict[str, object],
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: Dict[str, str] = {}

    for name, frame in splits.items():
        path = out_dir / f"{name}.parquet"
        saved = save_frame(frame, path)
        saved_paths[name] = str(saved)

    meta_path = out_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    saved_paths["metadata"] = str(meta_path)
    return saved_paths


def build_dataset(
    symbol: str,
    source: str,
    timeframe: str,
    start: str,
    end: Optional[str],
    exchange: str,
) -> pd.DataFrame:
    if source == "yfinance":
        data = fetch_yfinance(symbol=symbol, start=start, end=end, interval=timeframe)
    elif source == "ccxt":
        data = fetch_ccxt(
            symbol=symbol,
            exchange_name=exchange,
            start=start,
            end=end,
            timeframe=timeframe,
        )
    else:
        raise ValueError(f"Unsupported source: {source}")

    return compute_features(data)


def run_prepare(
    symbol: str,
    source: str,
    timeframe: str,
    start: str,
    end: Optional[str],
    exchange: str,
    out_dir: Path,
    train_frac: float,
    valid_frac: float,
) -> Dict[str, str]:
    dataset = build_dataset(
        symbol=symbol,
        source=source,
        timeframe=timeframe,
        start=start,
        end=end,
        exchange=exchange,
    )
    splits = split_walk_forward(dataset, train_frac=train_frac, valid_frac=valid_frac)

    metadata = {
        "symbol": symbol,
        "source": source,
        "timeframe": timeframe,
        "exchange": exchange,
        "start": start,
        "end": end,
        "rows": int(len(dataset)),
        "train_rows": int(len(splits["train"])),
        "valid_rows": int(len(splits["valid"])),
        "test_rows": int(len(splits["test"])),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    paths = save_splits(splits=splits, out_dir=out_dir, metadata=metadata)
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare market data for autonomous strategy search")
    parser.add_argument("--symbol", default="SPY", help="Ticker (e.g. SPY or BTC/USDT)")
    parser.add_argument("--source", choices=["yfinance", "ccxt"], default="yfinance")
    parser.add_argument("--exchange", default="binance", help="ccxt exchange name")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--valid-frac", type=float, default=0.2)
    return parser


def main() -> None:
    args = _parser().parse_args()
    out_dir = Path(args.out_dir)
    paths = run_prepare(
        symbol=args.symbol,
        source=args.source,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        exchange=args.exchange,
        out_dir=out_dir,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
    )
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
