from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import clean_ohlcv, compute_features, save_splits, split_walk_forward


def make_ohlcv(rows: int = 240) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=rows, freq="D", tz="UTC")
    close = np.linspace(100.0, 150.0, rows)
    out = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.linspace(1_000_000, 2_000_000, rows),
        },
        index=idx,
    )
    return out


def test_clean_ohlcv_normalizes_columns() -> None:
    raw = make_ohlcv(120)
    clean = clean_ohlcv(raw)
    assert list(clean.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(clean.index, pd.DatetimeIndex)
    assert clean.index.is_monotonic_increasing


def test_compute_features_has_target_and_no_nans() -> None:
    clean = clean_ohlcv(make_ohlcv(300))
    feat = compute_features(clean)
    assert "target_ret_1d" in feat.columns
    assert not feat.isna().any().any()


def test_compute_features_no_lookahead_for_ma10() -> None:
    clean = clean_ohlcv(make_ohlcv(200))
    feat = compute_features(clean)

    sample_idx = feat.index[10]
    src = clean.loc[:sample_idx, "close"]
    expected = src.tail(10).mean()
    observed = feat.loc[sample_idx, "ma_10"]
    assert np.isclose(observed, expected)


def test_split_walk_forward_chronological_and_disjoint() -> None:
    feat = compute_features(clean_ohlcv(make_ohlcv(400)))
    splits = split_walk_forward(feat, train_frac=0.6, valid_frac=0.2)

    train = splits["train"]
    valid = splits["valid"]
    test = splits["test"]

    assert train.index.max() < valid.index.min()
    assert valid.index.max() < test.index.min()
    assert len(train) + len(valid) + len(test) == len(feat)


def test_save_splits_writes_metadata(tmp_path: Path) -> None:
    feat = compute_features(clean_ohlcv(make_ohlcv(320)))
    splits = split_walk_forward(feat)

    metadata = {
        "symbol": "SPY",
        "rows": len(feat),
    }
    saved = save_splits(splits, tmp_path, metadata)

    assert Path(saved["metadata"]).exists()
    with Path(saved["metadata"]).open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["symbol"] == "SPY"
    assert loaded["rows"] == len(feat)

    for name in ("train", "valid", "test"):
        assert Path(saved[name]).exists()
