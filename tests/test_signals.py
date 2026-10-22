import pandas as pd
import numpy as np
import pytest
from stock_engine.models.signals import generate_signals, compute_threshold


def test_generate_signals_selects_top_k():
    data = {
        "date": ["2023-01-01", "2023-01-01", "2023-01-01", "2023-01-01"],
        "ticker": ["AAA", "BBB", "CCC", "DDD"],
        "prob": [0.9, 0.8, 0.7, 0.6],
    }
    df = pd.DataFrame(data)
    threshold = 0.75
    max_positions = 2
    signals = generate_signals(df, threshold, max_positions)
    assert set(signals.ticker) == {"AAA", "BBB"}
    assert len(signals) == 2


def test_compute_threshold_uses_quantile():
    probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    q = 0.6
    thr = compute_threshold(probs, q)
    assert thr == 0.7
