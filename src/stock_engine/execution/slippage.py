"""
Almgren-Chriss Market Impact and Dynamic Execution Slippage Model.

Implements the institutional Square-Root Law of Market Impact:
- Temporary Impact: slippage = eta * sigma * sqrt(OrderSize / AverageDailyVolume)
- Permanent Impact: price_drift = gamma * (OrderSize / AverageDailyVolume)
- Adverse execution: Buy orders execute higher than mid/decision price;
  Sell orders execute lower than mid/decision price.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionFill:
    side: Literal["BUY", "SELL"]
    decision_price: float
    fill_price: float
    quantity: int
    slippage_bps: float
    impact_cost_inr: float
    temporary_impact: float
    permanent_impact: float


class AlmgrenChrissSlippageModel:
    """
    Nonlinear institutional execution model adapting slippage to current order size,
    average daily volume (ADV), and market volatility.
    """

    def __init__(
        self,
        eta: float = 0.142,        # Temporary impact coefficient
        gamma: float = 0.314,      # Permanent impact coefficient
        base_spread_bps: float = 5.0, # Minimum bid-ask half-spread in bps
        max_slippage_bps: float = 150.0, # Safety circuit cap on slippage
    ):
        self.eta = eta
        self.gamma = gamma
        self.base_spread_bps = base_spread_bps
        self.max_slippage_bps = max_slippage_bps

    def compute_slippage_bps(
        self,
        quantity: int,
        daily_volume: float,
        daily_volatility: float = 0.015,
    ) -> float:
        """
        Calculate total execution slippage in basis points (bps).
        """
        if daily_volume <= 0 or quantity <= 0:
            return float(self.base_spread_bps)

        participation_rate = float(quantity / daily_volume)

        # Square-root temporary impact
        temp_impact_bps = self.eta * daily_volatility * np.sqrt(participation_rate) * 10000.0

        # Permanent impact
        perm_impact_bps = self.gamma * daily_volatility * participation_rate * 10000.0

        total_slippage = float(self.base_spread_bps + temp_impact_bps + 0.5 * perm_impact_bps)
        return float(min(self.max_slippage_bps, max(self.base_spread_bps, total_slippage)))

    def simulate_fill(
        self,
        side: Literal["BUY", "SELL"],
        decision_price: float,
        quantity: int,
        daily_volume: float = 1_000_000.0,
        daily_volatility: float = 0.015,
    ) -> ExecutionFill:
        """
        Simulate an order fill with adverse execution price.
        """
        if decision_price <= 0:
            raise ValueError(f"Decision price must be positive, got {decision_price}")
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        slippage_bps = self.compute_slippage_bps(
            quantity=quantity,
            daily_volume=daily_volume,
            daily_volatility=daily_volatility,
        )

        slippage_factor = slippage_bps / 10000.0

        participation = quantity / max(1.0, daily_volume)
        temp_impact = self.eta * daily_volatility * np.sqrt(participation) * decision_price
        perm_impact = self.gamma * daily_volatility * participation * decision_price

        if side.upper() == "BUY":
            fill_price = float(decision_price * (1.0 + slippage_factor))
        elif side.upper() == "SELL":
            fill_price = float(decision_price * (1.0 - slippage_factor))
        else:
            raise ValueError(f"Unsupported side: {side}")

        impact_cost = float(abs(fill_price - decision_price) * quantity)

        return ExecutionFill(
            side=side.upper(),
            decision_price=decision_price,
            fill_price=fill_price,
            quantity=quantity,
            slippage_bps=slippage_bps,
            impact_cost_inr=impact_cost,
            temporary_impact=temp_impact,
            permanent_impact=perm_impact,
        )
