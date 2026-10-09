from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.stock_engine.backtest.costs import CostModel, OrderCost


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    quantity: int
    entry_costs: OrderCost
    stop_price: Optional[float] = None


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    trade_log: List[dict]
    exposure_series: pd.Series


class Backtester:
    def __init__(
        self,
        prices: Dict[str, pd.DataFrame],
        signals: pd.DataFrame,
        cost_model: CostModel,
        params: Dict,
    ):
        self.prices = prices
        self.signals = signals.copy()
        self.cost_model = cost_model
        self.params = params
        self.max_positions = params.get('max_positions', 10)
        self.target_weight = params.get('target_weight_per_position', 0.1)
        self.holding_horizon = params.get('holding_horizon', 10)
        self.atr_stop = params.get('atr_stop', None)

        self.cash: float = 0.0
        self.initial_cash: float = 0.0
        self.positions: Dict[str, Position] = {}
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.trade_log: List[dict] = []
        self.exposure_series: List[Tuple[pd.Timestamp, float]] = []

        if self.atr_stop is not None:
            self.atr_data: Dict[str, pd.Series] = self._compute_atr()
        else:
            self.atr_data = {}

    def _compute_atr(self) -> Dict[str, pd.Series]:
        atr_dict = {}
        for ticker, df in self.prices.items():
            high = df['high']
            low = df['low']
            close = df['close']
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14, min_periods=1).mean()
            atr_dict[ticker] = atr
        return atr_dict

    def _get_position_market_value(self, date: pd.Timestamp) -> float:
        total = 0.0
        for ticker, pos in self.positions.items():
            close_price = self._get_close_price(ticker, date)
            if close_price is None:
                continue
            total += pos.quantity * close_price
        return total

    def _get_last_processed_date(self) -> pd.Timestamp:
        if not self.equity_curve:
            return pd.Timestamp.min
        return self.equity_curve[-1][0]

    def run_backtest(self, initial_cash: float = 1_000_000.0) -> BacktestResult:
        self.cash = initial_cash
        self.initial_cash = initial_cash

        all_dates = set()
        for df in self.prices.values():
            all_dates.update(df.index)
        self.trading_days = sorted(all_dates)

        self.signals = self.signals.sort_values('date')

        for i, current_date in enumerate(self.trading_days):
            self._process_exits_at_open(current_date)
            self._process_entries_at_open(current_date)
            if i != len(self.trading_days) - 1:
                self._update_equity_and_exposure_at_close(current_date)

        if self.trading_days:
            last_date = self.trading_days[-1]
            self._close_all_positions(last_date, reason='end_of_backtest')
            self._update_equity_and_exposure_at_close(last_date)

        equity_df = pd.DataFrame(self.equity_curve, columns=['date', 'equity']).set_index('date')
        exposure_df = pd.DataFrame(self.exposure_series, columns=['date', 'exposure']).set_index('date')
        daily_returns = equity_df['equity'].pct_change().fillna(0.0)

        return BacktestResult(
            equity_curve=equity_df['equity'],
            daily_returns=daily_returns,
            trade_log=self.trade_log,
            exposure_series=exposure_df['exposure']
        )

    def _process_exits_at_open(self, current_date: pd.Timestamp):
        for ticker, pos in list(self.positions.items()):
            exit_price = None
            exit_reason = None

            if self._has_exit_signal(ticker, current_date):
                exit_price = self._get_open_price(ticker, current_date)
                exit_reason = 'signal'
            elif (current_date - pos.entry_date).days >= self.holding_horizon:
                exit_price = self._get_open_price(ticker, current_date)
                exit_reason = 'horizon'
            elif self.atr_stop is not None and pos.stop_price is not None:
                open_price = self._get_open_price(ticker, current_date)
                if open_price is not None and open_price < pos.stop_price:
                    exit_price = open_price
                    exit_reason = 'stop_loss_gap'

            if exit_price is not None:
                self._exit_position(ticker, current_date, exit_price, exit_reason)

    def _process_stop_loss_during_day(self, current_date: pd.Timestamp):
        for ticker, pos in list(self.positions.items()):
            if self.atr_stop is None or pos.stop_price is None:
                continue
            low_price = self._get_low_price(ticker, current_date)
            if low_price is not None and low_price <= pos.stop_price:
                open_price = self._get_open_price(ticker, current_date)
                if open_price is not None and open_price < pos.stop_price:
                    exit_price = open_price
                    exit_reason = 'stop_loss_gap'
                else:
                    exit_price = pos.stop_price
                    exit_reason = 'stop_loss'
                self._exit_position(ticker, current_date, exit_price, exit_reason)

    def _process_entries_at_open(self, current_date: pd.Timestamp):
        prev_date = self._get_last_processed_date()
        day_signals = self.signals[self.signals['date'] == prev_date].sort_values('score', ascending=False)
        open_positions_count = len(self.positions)

        for _, signal in day_signals.iterrows():
            if signal.get('score', 0.0) <= 0:
                continue
            ticker = signal['ticker']
            if ticker in self.positions:
                continue
            if open_positions_count >= self.max_positions:
                break

            open_price = self._get_open_price(ticker, current_date)
            if open_price is None or open_price <= 0:
                continue

            market_value_prev = self._get_position_market_value(prev_date)
            current_equity = self.cash + market_value_prev
            target_equity = current_equity * self.target_weight

            max_shares = int(target_equity // open_price)
            if max_shares <= 0:
                continue

            order_cost = self.cost_model.order_cost('buy', open_price, max_shares)
            total_cost = open_price * max_shares + order_cost.total
            if total_cost > self.cash:
                for shares in range(max_shares, 0, -1):
                    order_cost = self.cost_model.order_cost('buy', open_price, shares)
                    total_cost = open_price * shares + order_cost.total
                    if total_cost <= self.cash:
                        max_shares = shares
                        break
                else:
                    continue

            self._enter_position(ticker, current_date, open_price, max_shares, signal.get('score', 0.0))
            open_positions_count += 1
            if open_positions_count >= self.max_positions:
                break

    def _enter_position(
        self,
        ticker: str,
        entry_date: pd.Timestamp,
        entry_price: float,
        quantity: int,
        signal_score: float
    ):
        order_cost = self.cost_model.order_cost('buy', entry_price, quantity)
        total_cost = entry_price * quantity + order_cost.total

        if total_cost > self.cash:
            return

        self.cash -= total_cost

        stop_price = None
        if self.atr_stop is not None and ticker in self.atr_data:
            atr_series = self.atr_data[ticker]
            atr_value = None
            if entry_date in atr_series.index:
                atr_value = atr_series.loc[entry_date]
            else:
                mask = atr_series.index < entry_date
                if mask.any():
                    atr_value = atr_series[mask].iloc[-1]
            if atr_value is not None and not pd.isna(atr_value):
                stop_price = entry_price - self.atr_stop * atr_value

        position = Position(
            ticker=ticker,
            entry_date=entry_date,
            entry_price=entry_price,
            quantity=quantity,
            entry_costs=order_cost,
            stop_price=stop_price
        )
        self.positions[ticker] = position

        self.trade_log.append({
            'ticker': ticker,
            'entry_date': entry_date,
            'exit_date': pd.NaT,
            'entry_price': entry_price,
            'exit_price': np.nan,
            'quantity': quantity,
            'entry_costs': order_cost,
            'exit_costs': None,
            'pnl': np.nan,
            'reason': 'entry',
            'signal_score': signal_score
        })

    def _exit_position(
        self,
        ticker: str,
        exit_date: pd.Timestamp,
        exit_price: float,
        reason: str
    ):
        if ticker not in self.positions:
            return

        pos = self.positions[ticker]
        order_cost = self.cost_model.order_cost('sell', exit_price, pos.quantity)
        total_revenue = exit_price * pos.quantity - order_cost.total

        self.cash += total_revenue

        entry_trade = None
        for trade in reversed(self.trade_log):
            if trade['ticker'] == ticker and pd.isna(trade['exit_date']):
                entry_trade = trade
                break
        if entry_trade is not None:
            pnl = (exit_price - pos.entry_price) * pos.quantity - (pos.entry_costs.total + order_cost.total)
            entry_trade.update({
                'exit_date': exit_date,
                'exit_price': exit_price,
                'exit_costs': order_cost,
                'pnl': pnl,
                'reason': reason
            })

        del self.positions[ticker]

    def _close_all_positions(self, current_date: pd.Timestamp, reason: str):
        for ticker, pos in list(self.positions.items()):
            close_price = self._get_close_price(ticker, current_date)
            if close_price is None:
                continue
            self._exit_position(ticker, current_date, close_price, reason)

    def _update_equity_and_exposure_at_close(self, current_date: pd.Timestamp):
        market_value = self._get_position_market_value(current_date)
        equity = self.cash + market_value
        self.equity_curve.append((current_date, equity))
        self.exposure_series.append((current_date, market_value))

    def _get_open_price(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        if ticker not in self.prices:
            return None
        df = self.prices[ticker]
        if date not in df.index:
            return None
        return df.loc[date, 'open']

    def _get_high_price(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        if ticker not in self.prices:
            return None
        df = self.prices[ticker]
        if date not in df.index:
            return None
        return df.loc[date, 'high']

    def _get_low_price(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        if ticker not in self.prices:
            return None
        df = self.prices[ticker]
        if date not in df.index:
            return None
        return df.loc[date, 'low']

    def _get_close_price(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        if ticker not in self.prices:
            return None
        df = self.prices[ticker]
        if date not in df.index:
            return None
        return df.loc[date, 'close']

    def _has_exit_signal(self, ticker: str, current_date: pd.Timestamp) -> bool:
        prev_date = self._get_last_processed_date()
        day_signals = self.signals[(self.signals['date'] == prev_date) & (self.signals['ticker'] == ticker)]
        if day_signals.empty:
            return False
        if 'exit_flag' in day_signals.columns:
            return bool(day_signals['exit_flag'].iloc[0])
        return False


def run_backtest(
    prices: Dict[str, pd.DataFrame],
    signals: pd.DataFrame,
    cost_model: CostModel,
    params: Dict,
    initial_cash: float = 1_000_000.0
) -> BacktestResult:
    backtester = Backtester(prices, signals, cost_model, params)
    return backtester.run_backtest(initial_cash)