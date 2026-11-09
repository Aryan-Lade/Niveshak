"""
Real-Time Market Tick Feed and Order Book Depth Simulator for NSE Equities.

Provides:
- Realistic tick-level price simulation with bid-ask spread and 5-level market depth.
- Async generator for real-time WebSocket streaming to frontend and paper broker.
- Deterministic random seed option for simulation testing.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from dataclasses import asdict, dataclass
from typing import AsyncGenerator, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MarketDepthLevel:
    price: float
    quantity: int
    orders: int


@dataclass
class MarketTick:
    ticker: str
    ltp: float                  # Last Traded Price
    open_price: float
    high_price: float
    low_price: float
    close_prev: float
    volume: int
    change_pct: float
    bids: List[MarketDepthLevel]
    asks: List[MarketDepthLevel]
    timestamp: str


class MarketTickSimulator:
    """
    Simulates high-fidelity tick updates and 5-level market depth for testing
    real-time paper execution and live UI dashboards.
    """

    def __init__(
        self,
        ticker: str,
        base_price: float = 2500.0,
        volatility: float = 0.015,
        tick_interval_sec: float = 0.5,
        seed: Optional[int] = 42,
    ):
        self.ticker = ticker
        self.base_price = base_price
        self.current_price = base_price
        self.open_price = base_price
        self.high_price = base_price
        self.low_price = base_price
        self.prev_close = base_price
        self.volume = 100000
        self.volatility = volatility
        self.tick_interval_sec = tick_interval_sec
        self.rng = np.random.default_rng(seed)

    def next_tick(self) -> MarketTick:
        """Generate the next simulated tick."""
        # Small Brownian increment
        dt = 1.0 / (252.0 * 375.0 * 120.0)
        drift = 0.0001 * dt
        shock = self.rng.normal(0, self.volatility * np.sqrt(dt))
        ret = drift + shock
        self.current_price = round(float(self.current_price * (1.0 + ret)), 2)

        self.high_price = max(self.high_price, self.current_price)
        self.low_price = min(self.low_price, self.current_price)
        self.volume += int(self.rng.integers(10, 500))

        change_pct = round(((self.current_price - self.prev_close) / self.prev_close) * 100.0, 2)

        # Build 5-level market depth
        tick_size = 0.05
        spread = tick_size * self.rng.integers(1, 4)
        best_bid = round(self.current_price - spread / 2.0, 2)
        best_ask = round(self.current_price + spread / 2.0, 2)

        bids = []
        asks = []
        for level in range(5):
            b_price = round(best_bid - level * tick_size, 2)
            a_price = round(best_ask + level * tick_size, 2)
            b_qty = int(self.rng.integers(50, 1500) * (5 - level))
            a_qty = int(self.rng.integers(50, 1500) * (5 - level))
            bids.append(MarketDepthLevel(price=b_price, quantity=b_qty, orders=int(self.rng.integers(1, 10))))
            asks.append(MarketDepthLevel(price=a_price, quantity=a_qty, orders=int(self.rng.integers(1, 10))))

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return MarketTick(
            ticker=self.ticker,
            ltp=self.current_price,
            open_price=self.open_price,
            high_price=self.high_price,
            low_price=self.low_price,
            close_prev=self.prev_close,
            volume=self.volume,
            change_pct=change_pct,
            bids=bids,
            asks=asks,
            timestamp=now_iso,
        )

    async def stream_ticks(self, max_ticks: int = 100) -> AsyncGenerator[Dict[str, any], None]:
        """Async generator yielding ticks as dictionaries."""
        for _ in range(max_ticks):
            tick = self.next_tick()
            yield asdict(tick)
            await asyncio.sleep(self.tick_interval_sec)
