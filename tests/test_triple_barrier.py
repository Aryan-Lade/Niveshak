import numpy as np
import pandas as pd
import pytest

from stock_engine.labels.triple_barrier import (
    compute_sample_weights,
    compute_triple_barrier_labels,
    compute_volatility,
)


def _make_toy_stock(n_bars: int = 30, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_bars)
    # Drift upward slightly
    prices = [100.0]
    for _ in range(n_bars - 1):
        prices.append(prices[-1] * (1.0 + rng.normal(0.001, 0.015)))

    rows = []
    for d, p in zip(dates, prices):
        rows.append({
            "date": d,
            "ticker": "TEST.NS",
            "open": p * 0.998,
            "high": p * 1.015,
            "low": p * 0.985,
            "close": p,
            "volume": 10000,
        })
    return pd.DataFrame(rows)


def test_triple_barrier_upper_hit():
    # Construct a price path where upper barrier is hit on day 2
    dates = pd.bdate_range("2023-01-01", periods=10)
    # Day 0: close=100.0 (vol ~ 0.02)
    # Day 1: open=100.0, high=101.0, low=99.0, close=100.0 (entry = 100.0)
    # Day 2: open=100.0, high=110.0 (touches upper barrier 103), low=99.0, close=108.0
    df = pd.DataFrame({
        "date": dates,
        "ticker": ["ABC.NS"] * 10,
        "open": [100.0] * 10,
        "high": [101.0, 101.0, 110.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0],
        "low": [99.0] * 10,
        "close": [100.0, 100.0, 108.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    })

    res = compute_triple_barrier_labels(df, k_up=1.5, k_dn=1.0, horizon=5, holdout_start=None)
    # Check day 0's label: entry is day 1 open (100.0), day 2 touches upper barrier
    row_t0 = res.loc[(dates[0], "ABC.NS")]
    assert row_t0["hit"] == 1
    assert row_t0["label"] == 1
    assert row_t0["exit_date"] == dates[2]


def test_triple_barrier_lower_hit():
    dates = pd.bdate_range("2023-01-01", periods=10)
    # Day 2 crashes down to 90
    df = pd.DataFrame({
        "date": dates,
        "ticker": ["ABC.NS"] * 10,
        "open": [100.0] * 10,
        "high": [101.0] * 10,
        "low": [99.0, 99.0, 90.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
        "close": [100.0, 100.0, 92.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    })

    res = compute_triple_barrier_labels(df, k_up=1.5, k_dn=1.0, horizon=5, holdout_start=None)
    row_t0 = res.loc[(dates[0], "ABC.NS")]
    assert row_t0["hit"] == -1
    assert row_t0["label"] == 0
    assert row_t0["exit_date"] == dates[2]


def test_triple_barrier_conservative_same_bar():
    dates = pd.bdate_range("2023-01-01", periods=10)
    # Day 2 has both extreme high (115) and extreme low (85)
    df = pd.DataFrame({
        "date": dates,
        "ticker": ["ABC.NS"] * 10,
        "open": [100.0] * 10,
        "high": [101.0, 101.0, 115.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0],
        "low": [99.0, 99.0, 85.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
        "close": [100.0] * 10,
    })

    res = compute_triple_barrier_labels(df, k_up=1.5, k_dn=1.0, horizon=5, holdout_start=None)
    row_t0 = res.loc[(dates[0], "ABC.NS")]
    # Conservative rule mandates lower barrier touch first
    assert row_t0["hit"] == -1
    assert row_t0["label"] == 0


def test_triple_barrier_vertical_barrier_reached():
    dates = pd.bdate_range("2023-01-01", periods=10)
    # Price stays flat in a narrow range [99.5, 100.5]
    df = pd.DataFrame({
        "date": dates,
        "ticker": ["ABC.NS"] * 10,
        "open": [100.0] * 10,
        "high": [100.5] * 10,
        "low": [99.5] * 10,
        "close": [100.0] * 10,
    })

    res = compute_triple_barrier_labels(df, k_up=2.0, k_dn=2.0, horizon=5, holdout_start=None)
    row_t0 = res.loc[(dates[0], "ABC.NS")]
    # Neither touched, vertical horizon reached
    assert row_t0["hit"] == 0
    assert row_t0["exit_date"] == dates[5]


def test_truncation_no_lookahead():
    """
    Truncation test: removing all data after exit date t1 must not alter
    the label or hit at date t.
    """
    df = _make_toy_stock(n_bars=25, seed=123)
    res_full = compute_triple_barrier_labels(df, k_up=1.5, k_dn=1.0, horizon=5, holdout_start=None)

    # Pick an event in the first half
    date_t = res_full.index.get_level_values("date")[3]
    t1 = res_full.loc[(date_t, "TEST.NS")]["t1"]

    # Truncate dataset after t1
    df_truncated = df[df["date"] <= t1].copy()
    res_trunc = compute_triple_barrier_labels(df_truncated, k_up=1.5, k_dn=1.0, horizon=5, holdout_start=None)

    row_full = res_full.loc[(date_t, "TEST.NS")]
    row_trunc = res_trunc.loc[(date_t, "TEST.NS")]

    assert row_full["label"] == row_trunc["label"]
    assert row_full["hit"] == row_trunc["hit"]
    assert pytest.approx(row_full["ret"], abs=1e-6) == row_trunc["ret"]


def test_sample_weights_positive():
    df = _make_toy_stock(n_bars=20, seed=77)
    res = compute_triple_barrier_labels(df, k_up=1.5, k_dn=1.0, horizon=5, holdout_start=None)
    weights = res["sample_weight"]
    assert (weights > 0).all()
    assert (weights <= 1.0).all()
