"""
Automated Daily Trading Runner (run_daily pipeline).

End-to-end daily cycle:
1. Fill queued orders at Market Open using actual open prices (from yfinance, delayed).
2. Snapshot mark-to-market equity curve.
3. Check Kill Switch status.
4. Generate signals using calibrated ML models and market features at Market Close.
5. Evaluate prospective orders through the deterministic Risk Engine.
6. Queue approved orders into PaperBroker (or GrowwBroker) for execution at next open.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from stock_engine.execution.broker import BrokerAdapter, Order
from stock_engine.execution.paper import PaperBroker
from stock_engine.risk.engine import OrderRequest, PortfolioState, RiskEngine
from stock_engine.risk.kill_switch import KillSwitch

logger = logging.getLogger(__name__)


class DailyTradingRunner:
    """
    Orchestrates the complete daily quant trading lifecycle.
    """

    def __init__(
        self,
        broker: Optional[BrokerAdapter] = None,
        risk_engine: Optional[RiskEngine] = None,
        kill_switch: Optional[KillSwitch] = None,
    ) -> None:
        self.broker = broker or PaperBroker()
        self.kill_switch = kill_switch or KillSwitch()
        self.risk_engine = risk_engine or RiskEngine(kill_switch=self.kill_switch)

    def run_daily(
        self,
        as_of: pd.Timestamp | str,
        open_prices: Optional[Dict[str, float]] = None,
        close_prices: Optional[Dict[str, float]] = None,
        signals_df: Optional[pd.DataFrame] = None,
        market_stats: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full daily quant workflow for trading date `as_of`.

        Parameters:
        - as_of: current trading date (YYYY-MM-DD)
        - open_prices: ticker -> day's open price (for filling prior pending orders)
        - close_prices: ticker -> day's close price (for mark-to-market and signals)
        - signals_df: precomputed signals DataFrame [ticker, calibrated_prob, score]
        - market_stats: ticker -> {atr, avg_daily_traded_value, vix_zscore, rag_sentiment, rag_confidence}
        """
        date_str = pd.Timestamp(as_of).strftime("%Y-%m-%d")
        logger.info(f"========== Starting Daily Trading Cycle: {date_str} ==========")

        fills: List[Order] = []
        # Step 1: Fill prior pending orders at today's Market Open
        if open_prices and isinstance(self.broker, PaperBroker):
            fills = self.broker.fill_pending_orders(open_prices=open_prices, fill_date=date_str)
            logger.info(f"Filled {len(fills)} pending orders at open of {date_str}")

        # Step 2: Mark-to-market equity snapshot
        ref_prices = close_prices or open_prices or {}
        if isinstance(self.broker, PaperBroker):
            total_equity = self.broker.snapshot_equity(as_of_date=date_str, current_prices=ref_prices)
        else:
            total_equity = self.broker.get_cash()

        current_cash = self.broker.get_cash()
        positions = self.broker.get_positions()
        pos_values = {t: p.quantity * ref_prices.get(t, p.avg_entry_price) for t, p in positions.items()}

        # Step 3: Check Kill Switch status
        is_halted = self.kill_switch.is_tripped()
        if is_halted:
            ks_status = self.kill_switch.status()
            logger.warning(f"Trading cycle halted by kill switch: {ks_status.get('reason')}")
            return {
                "date": date_str,
                "fills": [f.to_dict() for f in fills],
                "orders_queued": [],
                "total_equity": total_equity,
                "cash": current_cash,
                "status": "HALTED_BY_KILL_SWITCH",
            }

        # Step 4: Construct PortfolioState for Risk Engine
        portfolio = PortfolioState(
            total_equity=total_equity,
            cash=current_cash,
            start_of_day_equity=total_equity,  # Reset per daily cycle
            peak_equity=total_equity,
            open_positions=pos_values,
        )

        orders_queued: List[Dict[str, Any]] = []

        # Step 5: Evaluate Signals and prospective orders
        if signals_df is not None and not signals_df.empty:
            stats = market_stats or {}

            # Sort signals descending by calibrated probability
            candidates = signals_df.sort_values(
                by="calibrated_prob" if "calibrated_prob" in signals_df.columns else "score",
                ascending=False,
            )

            for _, row in candidates.iterrows():
                ticker = str(row["ticker"])
                intended_price = float(ref_prices.get(ticker, 0.0))
                if intended_price <= 0:
                    continue

                t_stats = stats.get(ticker, {})
                atr = float(t_stats.get("atr", 0.02 * intended_price))
                avg_val = float(t_stats.get("avg_daily_traded_value", 100_000_000.0))
                vix_z = t_stats.get("vix_zscore")
                rag_sent = t_stats.get("rag_sentiment")
                rag_conf = t_stats.get("rag_confidence")

                order_req = OrderRequest(
                    ticker=ticker,
                    side="BUY",
                    intended_price=intended_price,
                    last_close=intended_price,
                    atr=atr,
                    avg_daily_traded_value=avg_val,
                    vix_zscore=vix_z,
                    rag_sentiment=rag_sent,
                    rag_confidence=rag_conf,
                )

                risk_decision = self.risk_engine.evaluate_order(order_req, portfolio)

                if risk_decision.approved and risk_decision.approved_qty > 0:
                    client_order_id = f"ord_{ticker}_{date_str}_{int(datetime.datetime.now().timestamp())}"
                    placed_order = self.broker.place_order(
                        client_order_id=client_order_id,
                        ticker=ticker,
                        side="BUY",
                        quantity=risk_decision.approved_qty,
                        order_type="LIMIT",
                        price=intended_price,
                    )
                    orders_queued.append({
                        "order": placed_order.to_dict(),
                        "decision": risk_decision.to_dict(),
                    })
                    # Reduce theoretical cash for next candidates in same cycle
                    portfolio.cash -= risk_decision.approved_qty * intended_price

        logger.info(f"Completed Daily Trading Cycle: {date_str}. Fills: {len(fills)}, Orders queued: {len(orders_queued)}")
        return {
            "date": date_str,
            "fills": [f.to_dict() for f in fills],
            "orders_queued": orders_queued,
            "total_equity": total_equity,
            "cash": self.broker.get_cash(),
            "status": "COMPLETED",
        }

