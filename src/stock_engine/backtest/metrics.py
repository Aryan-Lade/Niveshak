from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def cagr(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]
    start_date = equity_curve.index[0]
    end_date = equity_curve.index[-1]
    years = (end_date - start_date).days / 365.25
    if years <= 0:
        return 0.0
    return (end_value / start_value) ** (1 / years) - 1


def annualized_volatility(daily_returns: pd.Series) -> float:
    if len(daily_returns) == 0:
        return 0.0
    return daily_returns.std() * np.sqrt(252)


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    if len(daily_returns) == 0:
        return 0.0
    excess_returns = daily_returns - risk_free_rate / periods_per_year
    std = excess_returns.std()
    if std == 0:
        return 0.0
    return excess_returns.mean() / std * np.sqrt(periods_per_year)


def sortino_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    if len(daily_returns) == 0:
        return 0.0
    excess_returns = daily_returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return np.inf
    downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)
    if downside_deviation == 0:
        return 0.0
    return excess_returns.mean() * np.sqrt(periods_per_year) / downside_deviation


def max_drawdown(equity_curve: pd.Series) -> Tuple[float, int]:
    if len(equity_curve) < 2:
        return 0.0, 0
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = drawdown.min()
    end_idx = drawdown.idxmin()
    start_idx = equity_curve.loc[:end_idx][equity_curve.loc[:end_idx] == running_max.loc[:end_idx]].index[-1]
    duration = (end_idx - start_idx).days
    return abs(max_dd), duration


def calmar_ratio(equity_curve: pd.Series) -> float:
    mdd, _ = max_drawdown(equity_curve)
    if mdd == 0:
        return np.inf
    return cagr(equity_curve) / mdd


def win_rate(trade_log: list) -> float:
    if not trade_log:
        return 0.0
    profits = [t['pnl'] for t in trade_log if not pd.isna(t.get('pnl'))]
    if not profits:
        return 0.0
    wins = sum(1 for p in profits if p > 0)
    return wins / len(profits)


def profit_factor(trade_log: list) -> float:
    if not trade_log:
        return 0.0
    profits = [t['pnl'] for t in trade_log if not pd.isna(t.get('pnl'))]
    if not profits:
        return 0.0
    gross_profits = sum(p for p in profits if p > 0)
    gross_losses = abs(sum(p for p in profits if p < 0))
    if gross_losses == 0:
        return np.inf
    return gross_profits / gross_losses


def average_holding_days(trade_log: list) -> float:
    if not trade_log:
        return 0.0
    holding_periods = []
    for t in trade_log:
        if pd.isna(t.get('exit_date')) or pd.isna(t.get('entry_date')):
            continue
        holding_periods.append((t['exit_date'] - t['entry_date']).days)
    if not holding_periods:
        return 0.0
    return float(np.mean(holding_periods))


def annual_turnover(
    trade_log: list,
    initial_cash: float,
    total_days: float
) -> float:
    if not trade_log or total_days <= 0:
        return 0.0
    years = total_days / 365.25
    total_turnover = 0.0
    for t in trade_log:
        if pd.isna(t.get('entry_price')) or pd.isna(t.get('exit_price')):
            continue
        total_turnover += t['entry_price'] * t['quantity'] + t['exit_price'] * t['quantity']
    return total_turnover / (initial_cash * years)


def average_exposure(exposure_series: pd.Series) -> float:
    if len(exposure_series) == 0:
        return 0.0
    return float(exposure_series.mean())


def bootstrap_sharpe_ci(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float]:
    if len(daily_returns) == 0:
        return 0.0, 0.0
    sharpe_estimates = []
    for _ in range(n_bootstrap):
        sample = daily_returns.sample(n=len(daily_returns), replace=True)
        sharpe = sharpe_ratio(sample, risk_free_rate)
        sharpe_estimates.append(sharpe)
    lower = float(np.percentile(sharpe_estimates, (1 - confidence) * 100 / 2))
    upper = float(np.percentile(sharpe_estimates, 100 - (1 - confidence) * 100 / 2))
    return lower, upper


def calculate_all_metrics(
    backtest_result: dict,
    risk_free_rate: float = 0.0,
    initial_cash: float = 1_000_000.0
) -> dict:
    equity_curve = backtest_result['equity_curve']
    daily_returns = backtest_result['daily_returns']
    trade_log = backtest_result['trade_log']
    exposure_series = backtest_result['exposure_series']

    total_days = (equity_curve.index[-1] - equity_curve.index[0]).days

    metrics = {
        'CAGR': cagr(equity_curve),
        'Annualized Volatility': annualized_volatility(daily_returns),
        'Sharpe Ratio': sharpe_ratio(daily_returns, risk_free_rate),
        'Sortino Ratio': sortino_ratio(daily_returns, risk_free_rate),
        'Max Drawdown': max_drawdown(equity_curve)[0],
        'Max Drawdown Duration': max_drawdown(equity_curve)[1],
        'Calmar Ratio': calmar_ratio(equity_curve),
        'Win Rate': win_rate(trade_log),
        'Profit Factor': profit_factor(trade_log),
        'Average Holding Days': average_holding_days(trade_log),
        'Annual Turnover': annual_turnover(trade_log, initial_cash, total_days),
        'Average Exposure': average_exposure(exposure_series),
    }
    sharpe_ci = bootstrap_sharpe_ci(daily_returns, risk_free_rate)
    metrics['Sharpe CI Lower'] = sharpe_ci[0]
    metrics['Sharpe CI Upper'] = sharpe_ci[1]

    return metrics