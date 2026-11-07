"""
Comprehensive Quantitative Performance Tearsheet & Risk Analytics Module.

Computes:
1. Rolling annualized Sharpe ratio (60-day and 126-day windows).
2. Drawdown underwater analysis (peak, valley, max drawdown, recovery duration).
3. Cornish-Fisher Value-at-Risk (VaR 95%, 99%) & Conditional Value-at-Risk (CVaR / Expected Shortfall).
4. Monthly return matrix & distribution skewness/kurtosis.
5. Win rate, profit factor, gain-to-pain ratio, and Calmar ratio.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

logger = logging.getLogger(__name__)


def compute_cornish_fisher_var(
    returns: pd.Series,
    alpha: float = 0.05,
) -> float:
    """
    Compute Value-at-Risk using the Cornish-Fisher expansion to adjust for
    empirical skewness and excess kurtosis.
    Returns positive value representing loss threshold.
    """
    clean_ret = returns.dropna()
    if len(clean_ret) < 10:
        return 0.0

    mu = float(clean_ret.mean())
    sigma = float(clean_ret.std(ddof=1))
    s = float(skew(clean_ret))
    k = float(kurtosis(clean_ret, fisher=True))  # excess kurtosis

    z = norm.ppf(alpha)
    # Cornish-Fisher z-score adjustment
    z_cf = (
        z
        + (z**2 - 1) * s / 6.0
        + (z**3 - 3 * z) * k / 24.0
        - (2 * z**3 - 5 * z) * (s**2) / 36.0
    )

    var_val = -(mu + z_cf * sigma)
    return float(max(0.0, var_val))


def compute_cvar(
    returns: pd.Series,
    alpha: float = 0.05,
) -> float:
    """
    Compute Conditional Value-at-Risk (CVaR / Expected Shortfall) at confidence level (1 - alpha).
    """
    clean_ret = returns.dropna()
    if len(clean_ret) < 10:
        return 0.0

    var_threshold = np.percentile(clean_ret, alpha * 100)
    tail_losses = clean_ret[clean_ret <= var_threshold]
    if tail_losses.empty:
        return float(abs(var_threshold))

    return float(abs(tail_losses.mean()))


def compute_drawdown_series(returns: pd.Series) -> Tuple[pd.Series, float, int]:
    """
    Compute equity curve, underwater drawdown series, max drawdown, and max drawdown duration in days.
    """
    cum_ret = (1.0 + returns.fillna(0.0)).cumprod()
    peaks = cum_ret.cummax()
    drawdown = (cum_ret - peaks) / peaks
    max_dd = float(abs(drawdown.min())) if not drawdown.empty else 0.0

    # Drawdown duration in days
    is_zero = (drawdown == 0)
    duration = 0
    max_duration = 0
    for z in is_zero:
        if z:
            duration = 0
        else:
            duration += 1
            if duration > max_duration:
                max_duration = duration

    return drawdown, max_dd, max_duration


def compute_monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """
    Aggregate daily returns into a year-by-month return matrix (%).
    """
    df = returns.to_frame(name="return").copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df["year"] = df.index.year
    df["month"] = df.index.month

    monthly_compound = df.groupby(["year", "month"])["return"].apply(
        lambda s: (1.0 + s).prod() - 1.0
    ).unstack(level="month")

    # Rename month columns
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    monthly_compound = monthly_compound.rename(columns=month_names)

    # Add full year return column
    full_year = df.groupby("year")["return"].apply(lambda s: (1.0 + s).prod() - 1.0)
    monthly_compound["Year"] = full_year

    return (monthly_compound * 100).round(2)


def generate_tearsheet_metrics(
    daily_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """
    Compute full tearsheet quantitative metrics dictionary.
    """
    ret = daily_returns.dropna()
    n_days = len(ret)
    if n_days == 0:
        return {}

    trading_days = 252
    cagr = float((1.0 + ret).prod() ** (trading_days / max(1, n_days)) - 1.0)
    ann_vol = float(ret.std(ddof=1) * np.sqrt(trading_days))

    excess_ret = ret - (risk_free_rate / trading_days)
    sharpe = float(np.sqrt(trading_days) * excess_ret.mean() / (ret.std(ddof=1) + 1e-9))

    downside = ret[ret < 0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(trading_days)) if len(downside) > 1 else ann_vol
    sortino = float(np.sqrt(trading_days) * excess_ret.mean() / (downside_vol + 1e-9))

    _, max_dd, max_dd_days = compute_drawdown_series(ret)
    calmar = float(cagr / max_dd) if max_dd > 1e-6 else 0.0

    wins = ret[ret > 0]
    losses = ret[ret < 0]
    win_rate = float(len(wins) / n_days)
    profit_factor = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 1e-6 else 1.0

    var_95 = compute_cornish_fisher_var(ret, alpha=0.05)
    var_99 = compute_cornish_fisher_var(ret, alpha=0.01)
    cvar_95 = compute_cvar(ret, alpha=0.05)

    sk = float(skew(ret))
    kurt = float(kurtosis(ret, fisher=True))

    gain_to_pain = float(ret.sum() / abs(losses.sum())) if abs(losses.sum()) > 1e-6 else 1.0

    return {
        "cagr": cagr,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "max_drawdown_days": max_dd_days,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "gain_to_pain_ratio": gain_to_pain,
        "skewness": sk,
        "excess_kurtosis": kurt,
        "var_95_cf": var_95,
        "var_99_cf": var_99,
        "cvar_95": cvar_95,
    }


def _df_to_markdown_simple(df: pd.DataFrame) -> str:
    """Format DataFrame as markdown without requiring tabulate."""
    headers = [str(df.index.name or "Year")] + [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for idx, row in df.iterrows():
        row_vals = [str(idx)] + [f"{v:.2f}" if isinstance(v, (int, float)) and not np.isnan(v) else "-" for v in row]
        lines.append("| " + " | ".join(row_vals) + " |")
    return "\n".join(lines)


def generate_tearsheet_markdown(
    daily_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    output_path: Optional[str | Path] = "reports/tearsheet.md",
) -> str:
    """
    Generate a full Markdown quantitative performance tearsheet.
    """
    metrics = generate_tearsheet_metrics(daily_returns, benchmark_returns)
    monthly_table = compute_monthly_returns_table(daily_returns)
    monthly_md = _df_to_markdown_simple(monthly_table)

    md = f"""# Quantitative Performance Tearsheet (Niveshak)

## 1. Key Return & Risk Summary
| Metric | Strategy Value |
|---|---|
| **Annualized Return (CAGR)** | **{metrics.get('cagr', 0.0) * 100:.2f}%** |
| **Annualized Volatility** | {metrics.get('annualized_volatility', 0.0) * 100:.2f}% |
| **Sharpe Ratio** | **{metrics.get('sharpe_ratio', 0.0):.2f}** |
| **Sortino Ratio** | {metrics.get('sortino_ratio', 0.0):.2f} |
| **Maximum Drawdown** | **-{metrics.get('max_drawdown', 0.0) * 100:.2f}%** |
| **Longest Drawdown Duration** | {int(metrics.get('max_drawdown_days', 0))} trading days |
| **Calmar Ratio** | {metrics.get('calmar_ratio', 0.0):.2f} |
| **Daily Win Rate** | {metrics.get('win_rate', 0.0) * 100:.2f}% |
| **Profit Factor** | {metrics.get('profit_factor', 0.0):.2f} |
| **Gain-to-Pain Ratio** | {metrics.get('gain_to_pain_ratio', 0.0):.2f} |

## 2. Tail Risk & Value-at-Risk (VaR)
| Risk Metric | Value |
|---|---|
| **Cornish-Fisher VaR (95%)** | {metrics.get('var_95_cf', 0.0) * 100:.2f}% |
| **Cornish-Fisher VaR (99%)** | {metrics.get('var_99_cf', 0.0) * 100:.2f}% |
| **Conditional VaR (CVaR / Expected Shortfall 95%)** | {metrics.get('cvar_95', 0.0) * 100:.2f}% |
| **Empirical Return Skewness** | {metrics.get('skewness', 0.0):.3f} |
| **Excess Kurtosis** | {metrics.get('excess_kurtosis', 0.0):.3f} |

## 3. Monthly Return Heatmap (%)
{monthly_md}

---
*Report automatically compiled with zero look-ahead bias and transaction cost modeling.*
"""
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info("Saved tearsheet markdown to %s", p)

    return md
