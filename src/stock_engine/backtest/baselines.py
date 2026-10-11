"""
Baseline strategies for the Stock Intelligence Engine.
Implements and evaluates 5 baseline strategies to establish performance benchmarks.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
import tempfile
import os
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.stock_engine.backtest.engine import Backtester, run_backtest
    from src.stock_engine.backtest.costs import CostModel
    from src.stock_engine.backtest.metrics import calculate_all_metrics
except ImportError:
    from stock_engine.backtest.engine import Backtester, run_backtest
    from stock_engine.backtest.costs import CostModel
    from stock_engine.backtest.metrics import calculate_all_metrics

logger = logging.getLogger(__name__)

CONFIG_DIR = Path("configs") if Path("configs").exists() else Path("stock-engine/configs")
DATA_STORE_DIR = Path("data_store")
REPORTS_DIR = Path("reports")


def load_configs() -> Tuple[Dict, Dict, Dict, Dict]:
    config_dir = Path("stock-engine/configs") if Path("stock-engine/configs").exists() else Path("configs")
    with open(config_dir / "universe.yaml", 'r') as f:
        universe_config = yaml.safe_load(f)
    with open(config_dir / "data.yaml", 'r') as f:
        data_config = yaml.safe_load(f)
    costs_path = Path("configs/costs.yaml") if Path("configs/costs.yaml").exists() else (config_dir / "costs.yaml")
    with open(costs_path, 'r') as f:
        costs_config = yaml.safe_load(f)
    with open(config_dir / "risk.yaml", 'r') as f:
        risk_config = yaml.safe_load(f)

    return universe_config, data_config, costs_config, risk_config


def load_price_data(universe: list) -> Dict[str, pd.DataFrame]:
    prices = {}
    raw_prices_dir = DATA_STORE_DIR / "raw" / "prices"

    for ticker in universe:
        price_file = raw_prices_dir / f"{ticker}.parquet"
        if not price_file.exists():
            logger.warning(f"Price data not found for {ticker} at {price_file}")
            continue

        try:
            df = pd.read_parquet(price_file)

            new_columns = []
            for col in df.columns:
                if isinstance(col, str) and col.startswith("('") and col.endswith("')"):
                    try:
                        import ast
                        parsed = ast.literal_eval(col)
                        new_col = parsed[0] if parsed else col
                    except Exception:
                        new_col = col
                else:
                    new_col = col
                new_columns.append(new_col)
            df.columns = new_columns

            column_mapping = {}
            for col in df.columns:
                if col == 'date':
                    column_mapping[col] = 'date'
                elif col == 'ticker':
                    column_mapping[col] = 'ticker'
                elif 'open' in str(col).lower():
                    column_mapping[col] = 'open'
                elif 'high' in str(col).lower():
                    column_mapping[col] = 'high'
                elif 'low' in str(col).lower():
                    column_mapping[col] = 'low'
                elif 'close' in str(col).lower():
                    column_mapping[col] = 'close'
                elif 'volume' in str(col).lower():
                    column_mapping[col] = 'volume'

            if column_mapping:
                df = df.rename(columns=column_mapping)

            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()

            required_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in required_cols if col in df.columns]
            if len(available_cols) == len(required_cols):
                df = df[required_cols]
                prices[ticker] = df
                logger.info(f"Loaded price data for {ticker}: {len(df)} rows")
            else:
                logger.warning(f"Missing required columns for {ticker}. Available: {df.columns.tolist()}")

        except Exception as e:
            logger.error(f"Error loading price data for {ticker}: {e}")

    return prices


def create_nifty_buy_hold_signals(prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    logger.info("Creating NIFTY 50 buy-and-hold signals...")

    if not prices:
        raise ValueError("No price data available")

    first_ticker = list(prices.keys())[0]
    date_range = prices[first_ticker].index

    signals = []
    if len(date_range) > 0:
        first_date = date_range[0]
        signals.append({
            'date': first_date,
            'ticker': 'NIFTY_50',
            'score': 1.0,
            'exit_flag': False
        })

    signals_df = pd.DataFrame(signals)
    logger.info(f"Created NIFTY buy-and-hold signals: {len(signals_df)} signals")
    return signals_df


def create_equal_weight_signals(prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    logger.info("Creating equal-weight monthly rebalance signals...")

    if not prices:
        raise ValueError("No price data available")

    first_ticker = list(prices.keys())[0]
    date_range = prices[first_ticker].index

    signals = []
    monthly_dates = {}
    for date in date_range:
        month_key = date.strftime('%Y-%m')
        if month_key not in monthly_dates:
            monthly_dates[month_key] = []
        monthly_dates[month_key].append(date)

    for month_key, dates_in_month in monthly_dates.items():
        if not dates_in_month:
            continue

        rebalance_date = dates_in_month[0]
        for ticker in prices.keys():
            signals.append({
                'date': rebalance_date,
                'ticker': ticker,
                'score': 1.0,
                'exit_flag': False
            })

    signals_df = pd.DataFrame(signals)
    logger.info(f"Created equal-weight signals: {len(signals_df)} signals")
    return signals_df


def create_sma_crossover_signals(prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    logger.info("Creating SMA(20/50) crossover signals...")

    signals = []

    for ticker, df in prices.items():
        if len(df) < 50:
            continue

        close_prices = df['close']
        sma_20 = close_prices.rolling(window=20, min_periods=20).mean()
        sma_50 = close_prices.rolling(window=50, min_periods=50).mean()

        position_open = False

        for i in range(len(df)):
            if pd.isna(sma_20.iloc[i]) or pd.isna(sma_50.iloc[i]):
                continue

            date = df.index[i]

            if (not position_open and
                i > 0 and
                not pd.isna(sma_20.iloc[i-1]) and
                not pd.isna(sma_50.iloc[i-1]) and
                sma_20.iloc[i-1] <= sma_50.iloc[i-1] and
                sma_20.iloc[i] > sma_50.iloc[i]):
                signals.append({
                    'date': date,
                    'ticker': ticker,
                    'score': 1.0,
                    'exit_flag': False
                })
                position_open = True

            elif (position_open and
                  i > 0 and
                  not pd.isna(sma_20.iloc[i-1]) and
                  not pd.isna(sma_50.iloc[i-1]) and
                  sma_20.iloc[i-1] >= sma_50.iloc[i-1] and
                  sma_20.iloc[i] < sma_50.iloc[i]):
                position_open = False

    signals_df = pd.DataFrame(signals)
    logger.info(f"Created SMA crossover signals: {len(signals_df)} signals")
    return signals_df


def create_momentum_signals(prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    logger.info("Creating cross-sectional momentum signals...")

    if not prices:
        raise ValueError("No price data available")

    first_ticker = list(prices.keys())[0]
    date_range = prices[first_ticker].index

    signals = []
    monthly_dates = {}
    for date in date_range:
        month_key = date.strftime('%Y-%m')
        if month_key not in monthly_dates:
            monthly_dates[month_key] = []
        monthly_dates[month_key].append(date)

    for month_key, dates_in_month in monthly_dates.items():
        if not dates_in_month:
            continue

        rebalance_date = dates_in_month[0]
        momentum_scores = {}

        for ticker, df in prices.items():
            if rebalance_date not in df.index:
                continue

            try:
                idx = df.index.get_loc(rebalance_date)
            except KeyError:
                continue

            if idx < 252:
                continue

            if idx - 252 < 0 or idx - 21 < 0:
                continue

            price_12m_ago = df.iloc[idx - 252]['close']
            price_1m_ago = df.iloc[idx - 21]['close']

            if price_12m_ago <= 0:
                continue

            momentum = (price_1m_ago - price_12m_ago) / price_12m_ago
            momentum_scores[ticker] = momentum

        if momentum_scores:
            sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
            top_5 = [ticker for ticker, _ in sorted_stocks[:5]]

            for ticker in top_5:
                signals.append({
                    'date': rebalance_date,
                    'ticker': ticker,
                    'score': 1.0,
                    'exit_flag': False
                })

    signals_df = pd.DataFrame(signals)
    logger.info(f"Created momentum signals: {len(signals_df)} signals")
    return signals_df


def create_rsi_mean_reversion_signals(prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    logger.info("Creating RSI(14) < 30 mean-reversion signals...")

    signals = []

    for ticker, df in prices.items():
        if len(df) < 14:
            logger.warning(f"Insufficient data for {ticker}: {len(df)} rows")
            continue

        close_prices = df['close']
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        for i in range(len(rsi)):
            if pd.isna(rsi.iloc[i]):
                continue

            if rsi.iloc[i] < 30:
                signals.append({
                    'date': df.index[i],
                    'ticker': ticker,
                    'score': 1.0,
                    'exit_flag': False
                })

    signals_df = pd.DataFrame(signals)
    logger.info(f"Created RSI mean-reversion signals: {len(signals_df)} signals")
    return signals_df


def run_strategy(
    name: str,
    prices: Dict[str, pd.DataFrame],
    signals: pd.DataFrame,
    cost_model: CostModel,
    params: Dict
) -> Dict:
    logger.info(f"Running strategy: {name}")

    try:
        result = run_backtest(
            prices=prices,
            signals=signals,
            cost_model=cost_model,
            params=params,
            initial_cash=1_000_000.0
        )

        metrics = calculate_all_metrics({
            'equity_curve': result.equity_curve,
            'daily_returns': result.daily_returns,
            'trade_log': result.trade_log,
            'exposure_series': result.exposure_series
        }, initial_cash=1_000_000.0)

        logger.info(f"Strategy {name} completed. CAGR: {metrics.get('CAGR', 0):.2%}")

        return {
            'name': name,
            'equity_curve': result.equity_curve,
            'metrics': metrics,
            'trade_log': result.trade_log,
            'success': True
        }

    except Exception as e:
        logger.error(f"Error running strategy {name}: {e}")
        return {
            'name': name,
            'error': str(e),
            'success': False
        }


def save_equity_curve_plot(strategies_results: list, save_path: Path):
    logger.info("Creating equity curve comparison plot...")

    plt.figure(figsize=(12, 8))

    for result in strategies_results:
        if not result['success'] or 'equity_curve' not in result:
            continue

        equity_curve = result['equity_curve']
        normalized = equity_curve / equity_curve.iloc[0] * 100
        plt.plot(normalized.index, normalized.values, label=result['name'], linewidth=2)

    plt.title('Baseline Strategies Equity Curve Comparison (Normalized to 100)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Equity curve plot saved to {save_path}")


def generate_baselines_report(strategies_results: list, save_path: Path):
    logger.info("Generating baselines report...")

    table_data = []

    for result in strategies_results:
        if not result['success']:
            continue

        metrics = result['metrics']
        row = {
            'Strategy': result['name'],
            'CAGR': f"{metrics.get('CAGR', 0):.2%}",
            'Sharpe Ratio': f"{metrics.get('Sharpe Ratio', 0):.2f}",
            'Sharpe CI Lower': f"{metrics.get('Sharpe CI Lower', 0):.2f}",
            'Sharpe CI Upper': f"{metrics.get('Sharpe CI Upper', 0):.2f}",
            'Max Drawdown': f"{metrics.get('Max Drawdown', 0):.2%}",
            'Annual Turnover': f"{metrics.get('Annual Turnover', 0):.2f}",
        }
        table_data.append(row)

    table_data.sort(key=lambda x: float(x['Sharpe Ratio']), reverse=True)

    if table_data:
        headers = table_data[0].keys()
        markdown_lines = [
            "# Baselines Performance Report\n",
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|"
        ]

        for row in table_data:
            markdown_lines.append("| " + " | ".join([str(row[h]) for h in headers]) + " |")

        strongest = table_data[0] if table_data else None
        if strongest:
            markdown_lines.extend([
                "",
                f"**Strongest Baseline by Sharpe Ratio:** {strongest['Strategy']} ",
                f"(Sharpe: {strongest['Sharpe Ratio']})"
            ])

        markdown_content = "\n".join(markdown_lines)
    else:
        markdown_content = "# Baselines Performance Report\n\nNo successful strategies to report."

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(markdown_content)

    logger.info(f"Baselines report saved to {save_path}")


def main():
    logger.info("Starting baseline strategies evaluation...")

    universe_config, data_config, costs_config, risk_config = load_configs()
    universe = universe_config['universe']

    logger.info(f"Loaded universe with {len(universe)} stocks")
    logger.info(f"Data range: {data_config['data']['start_date']} to {data_config['data']['holdout_start']}")

    prices = load_price_data(universe)

    if not prices:
        logger.error("No price data loaded. Exiting.")
        return

    logger.info(f"Loaded price data for {len(prices)} stocks")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(costs_config, f)
        costs_config_path = f.name

    try:
        cost_model = CostModel(costs_config_path)
    finally:
        os.unlink(costs_config_path)

    base_params = {
        'max_positions': 20,
        'target_weight_per_position': 0.05,
        'holding_horizon': 5,
        'atr_stop': None
    }

    strategies_results = []

    logger.info("\n=== Running NIFTY 50 Buy-and-Hold ===")
    nifty_signals = create_nifty_buy_hold_signals(prices)
    nifty_params = base_params.copy()
    nifty_params['max_positions'] = 1
    nifty_params['target_weight_per_position'] = 1.0
    nifty_result = run_strategy(
        "NIFTY 50 Buy-and-Hold",
        prices,
        nifty_signals,
        cost_model,
        nifty_params
    )
    strategies_results.append(nifty_result)

    logger.info("\n=== Running Equal-Weight Monthly Rebalance ===")
    equal_weight_signals = create_equal_weight_signals(prices)
    equal_weight_params = base_params.copy()
    equal_weight_result = run_strategy(
        "Equal-Weight Monthly Rebalance",
        prices,
        equal_weight_signals,
        cost_model,
        equal_weight_params
    )
    strategies_results.append(equal_weight_result)

    logger.info("\n=== Running SMA(20/50) Crossover ===")
    sma_signals = create_sma_crossover_signals(prices)
    sma_params = base_params.copy()
    sma_result = run_strategy(
        "SMA(20/50) Crossover",
        prices,
        sma_signals,
        cost_model,
        sma_params
    )
    strategies_results.append(sma_result)

    logger.info("\n=== Running Cross-Sectional Momentum ===")
    momentum_signals = create_momentum_signals(prices)
    momentum_params = base_params.copy()
    momentum_params['max_positions'] = 5
    momentum_params['target_weight_per_position'] = 0.2
    momentum_result = run_strategy(
        "Cross-Sectional Momentum (Top 5)",
        prices,
        momentum_signals,
        cost_model,
        momentum_params
    )
    strategies_results.append(momentum_result)

    logger.info("\n=== Running RSI(14) < 30 Mean-Reversion ===")
    rsi_signals = create_rsi_mean_reversion_signals(prices)
    rsi_params = base_params.copy()
    rsi_params['holding_horizon'] = 5
    rsi_result = run_strategy(
        "RSI(14) < 30 Mean-Reversion (5-day hold)",
        prices,
        rsi_signals,
        cost_model,
        rsi_params
    )
    strategies_results.append(rsi_result)

    logger.info("\n=== Generating Reports ===")

    plot_save_path = REPORTS_DIR / "baselines_equity_curve.png"
    save_equity_curve_plot(strategies_results, plot_save_path)

    report_save_path = REPORTS_DIR / "baselines.md"
    generate_baselines_report(strategies_results, report_save_path)

    logger.info("\n=== BASELINES EVALUATION COMPLETE ===")
    successful_strategies = [r for r in strategies_results if r['success']]

    if successful_strategies:
        print("\nPerformance Summary:")
        print("-" * 80)
        print(f"{'Strategy':<35} {'CAGR':<8} {'Sharpe':<8} {'MaxDD':<8} {'Turnover':<8}")
        print("-" * 80)
        for result in successful_strategies:
            metrics = result['metrics']
            print(f"{result['name']:<35} {metrics.get('CAGR', 0):<7.2%} {metrics.get('Sharpe Ratio', 0):<7.2f} "
                  f"{metrics.get('Max Drawdown', 0):<7.2%} {metrics.get('Annual Turnover', 0):<7.2f}")

        strongest = max(successful_strategies,
                       key=lambda x: x['metrics'].get('Sharpe Ratio', 0))
        print("-" * 80)
        print(f"Strongest baseline by Sharpe ratio: {strongest['name']} "
              f"(Sharpe: {strongest['metrics'].get('Sharpe Ratio', 0):.2f})")
        print(f"Check {report_save_path} for detailed report.")
        print(f"Check {plot_save_path} for equity curve comparison.")
    else:
        logger.error("No strategies completed successfully")


if __name__ == "__main__":
    main()