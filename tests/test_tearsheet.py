from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from stock_engine.backtest.tearsheet import (
    compute_cornish_fisher_var,
    compute_cvar,
    compute_drawdown_series,
    compute_monthly_returns_table,
    generate_tearsheet_markdown,
    generate_tearsheet_metrics,
)


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-01", periods=252)
    daily_rets = rng.normal(0.0008, 0.012, size=len(dates))
    return pd.Series(daily_rets, index=dates)


def test_cornish_fisher_var_positive(sample_returns):
    var_95 = compute_cornish_fisher_var(sample_returns, alpha=0.05)
    var_99 = compute_cornish_fisher_var(sample_returns, alpha=0.01)

    assert var_95 > 0.0
    assert var_99 > var_95  # 99% VaR must exceed 95% VaR


def test_cvar_exceeds_var(sample_returns):
    var_95 = compute_cornish_fisher_var(sample_returns, alpha=0.05)
    cvar_95 = compute_cvar(sample_returns, alpha=0.05)

    assert cvar_95 > 0.0
    assert cvar_95 >= var_95 * 0.85


def test_drawdown_series_monotonic_peak():
    dates = pd.bdate_range("2023-01-01", periods=5)
    # Starts up, then drops
    rets = pd.Series([0.10, -0.05, -0.05, 0.02, -0.01], index=dates)
    dd_series, max_dd, max_dur = compute_drawdown_series(rets)

    assert (dd_series <= 0.0).all()
    assert max_dd > 0.0
    assert max_dur >= 1


def test_monthly_returns_table_structure(sample_returns):
    table = compute_monthly_returns_table(sample_returns)
    assert not table.empty
    assert "Year" in table.columns
    assert 2023 in table.index


def test_generate_tearsheet_markdown(sample_returns, tmp_path):
    out_file = tmp_path / "tearsheet.md"
    md = generate_tearsheet_markdown(sample_returns, output_path=out_file)

    assert out_file.exists()
    assert "# Quantitative Performance Tearsheet" in md
    assert "Cornish-Fisher VaR" in md
    assert "CAGR" in md
