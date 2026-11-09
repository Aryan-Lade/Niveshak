"""
Portfolio Rebalance Optimizer and Target Weight Execution Planner.

Computes target-weight delta orders:
1. Translates continuous model signal scores into discrete target weights.
2. Applies position constraints (e.g. 20% max position limit, cash reserves).
3. Produces executable trade delta orders with integer shares and estimated cost breakdown.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from stock_engine.backtest.costs import CostModel, OrderCost

logger = logging.getLogger(__name__)


@dataclass
class RebalanceOrder:
    ticker: str
    action: str              # "BUY", "SELL", "HOLD"
    current_shares: int
    target_shares: int
    delta_shares: int
    estimated_price: float
    trade_value: float
    estimated_cost: OrderCost


@dataclass
class RebalancePlan:
    nav: float
    cash_available: float
    cash_after_rebalance: float
    orders: List[RebalanceOrder]
    total_estimated_costs_inr: float
    turnover_inr: float


class PortfolioRebalancer:
    """
    Computes optimal portfolio rebalancing orders to transition current holdings
    to target signal weights under integer lot and cost constraints.
    """

    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        max_position_weight: float = 0.20,
        min_trade_value_inr: float = 5000.0,
    ):
        self.cost_model = cost_model or CostModel()
        self.max_position_weight = max_position_weight
        self.min_trade_value_inr = min_trade_value_inr

    def plan_rebalance(
        self,
        current_positions: Dict[str, int],       # {ticker: shares}
        market_prices: Dict[str, float],          # {ticker: price}
        target_weights: Dict[str, float],         # {ticker: target_weight_0_to_1}
        cash: float,
    ) -> RebalancePlan:
        """
        Generate rebalance plan from current holdings to target weights.
        """
        # Calculate current total NAV
        invested_val = sum(
            current_positions.get(t, 0) * market_prices.get(t, 0.0)
            for t in set(list(current_positions.keys()) + list(market_prices.keys()))
        )
        total_nav = cash + invested_val

        # Normalize and cap target weights
        total_w = sum(max(0.0, w) for w in target_weights.values())
        norm_weights = {}
        for t, w in target_weights.items():
            w_pos = max(0.0, w)
            if total_w > 1.0:
                scaled = min(self.max_position_weight, w_pos / total_w)
            else:
                scaled = min(self.max_position_weight, w_pos)
            norm_weights[t] = scaled

        all_tickers = sorted(set(list(current_positions.keys()) + list(norm_weights.keys())))
        rebalance_orders = []
        total_cost = 0.0
        turnover = 0.0
        cash_projected = cash

        # Sells first to free cash
        sell_orders = []
        buy_orders = []

        for ticker in all_tickers:
            price = market_prices.get(ticker, 0.0)
            if price <= 0:
                continue

            curr_qty = current_positions.get(ticker, 0)
            target_w = norm_weights.get(ticker, 0.0)
            target_val = total_nav * target_w
            target_qty = int(target_val / price)

            delta = target_qty - curr_qty
            trade_val = abs(delta) * price

            if abs(delta) == 0 or trade_val < self.min_trade_value_inr:
                action = "HOLD"
                cost = self.cost_model.order_cost(side="BUY", price=price, quantity=0)
                rebalance_orders.append(RebalanceOrder(
                    ticker=ticker,
                    action=action,
                    current_shares=curr_qty,
                    target_shares=curr_qty,
                    delta_shares=0,
                    estimated_price=price,
                    trade_value=0.0,
                    estimated_cost=cost,
                ))
            elif delta < 0:
                # Sell
                qty_to_sell = abs(delta)
                cost = self.cost_model.order_cost(side="SELL", price=price, quantity=qty_to_sell)
                sell_orders.append(RebalanceOrder(
                    ticker=ticker,
                    action="SELL",
                    current_shares=curr_qty,
                    target_shares=target_qty,
                    delta_shares=-qty_to_sell,
                    estimated_price=price,
                    trade_value=trade_val,
                    estimated_cost=cost,
                ))
                cash_projected += (trade_val - cost.total)
                total_cost += cost.total
                turnover += trade_val
            else:
                # Buy
                qty_to_buy = delta
                cost = self.cost_model.order_cost(side="BUY", price=price, quantity=qty_to_buy)
                buy_orders.append(RebalanceOrder(
                    ticker=ticker,
                    action="BUY",
                    current_shares=curr_qty,
                    target_shares=target_qty,
                    delta_shares=qty_to_buy,
                    estimated_price=price,
                    trade_value=trade_val,
                    estimated_cost=cost,
                ))

        # Filter buys if projected cash is insufficient
        for bo in buy_orders:
            needed = bo.trade_value + bo.estimated_cost.total
            if needed <= cash_projected:
                rebalance_orders.append(bo)
                cash_projected -= needed
                total_cost += bo.estimated_cost.total
                turnover += bo.trade_value
            else:
                # Scale down buy quantity to fit available cash
                affordable_shares = int((cash_projected - 100.0) / bo.estimated_price)
                if affordable_shares > 0 and (affordable_shares * bo.estimated_price) >= self.min_trade_value_inr:
                    cost = self.cost_model.order_cost(side="BUY", price=bo.estimated_price, quantity=affordable_shares)
                    trade_val = affordable_shares * bo.estimated_price
                    scaled_order = RebalanceOrder(
                        ticker=bo.ticker,
                        action="BUY",
                        current_shares=bo.current_shares,
                        target_shares=bo.current_shares + affordable_shares,
                        delta_shares=affordable_shares,
                        estimated_price=bo.estimated_price,
                        trade_value=trade_val,
                        estimated_cost=cost,
                    )
                    rebalance_orders.append(scaled_order)
                    cash_projected -= (trade_val + cost.total)
                    total_cost += cost.total
                    turnover += trade_val

        final_orders = sell_orders + [o for o in rebalance_orders if o.action != "SELL"]

        return RebalancePlan(
            nav=round(total_nav, 2),
            cash_available=round(cash, 2),
            cash_after_rebalance=round(cash_projected, 2),
            orders=final_orders,
            total_estimated_costs_inr=round(total_cost, 2),
            turnover_inr=round(turnover, 2),
        )
