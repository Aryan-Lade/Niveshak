"""
Abstract Broker Adapter Interface for Order and Portfolio Management.

Standard interface implemented by PaperBroker (Day 24) and GrowwBroker (Day 28).
Guarantees idempotency via unique client_order_id.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Order:
    """Standardized order entity across all broker implementations."""
    client_order_id: str
    ticker: str
    side: str  # "BUY" or "SELL"
    quantity: int
    order_type: str = "LIMIT"  # "LIMIT", "MARKET"
    price: Optional[float] = None
    status: str = "PENDING"  # "PENDING", "FILLED", "CANCELLED", "REJECTED"
    created_at: str = ""
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    total_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "ticker": self.ticker,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "price": self.price,
            "status": self.status,
            "created_at": self.created_at,
            "filled_qty": self.filled_qty,
            "avg_fill_price": self.avg_fill_price,
            "total_cost": self.total_cost,
        }


@dataclass
class Position:
    """Current open intraday/swing trading position."""
    ticker: str
    quantity: int
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "quantity": self.quantity,
            "avg_entry_price": self.avg_entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
        }


@dataclass
class Holding:
    """Dematerialized cash/delivery equity holding (CNC)."""
    ticker: str
    quantity: int
    avg_cost: float
    market_value: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "quantity": self.quantity,
            "avg_cost": self.avg_cost,
            "market_value": self.market_value,
        }


class BrokerAdapter(ABC):
    """Abstract interface defining required broker capabilities."""

    @abstractmethod
    def place_order(
        self,
        client_order_id: str,
        ticker: str,
        side: str,
        quantity: int,
        order_type: str = "LIMIT",
        price: Optional[float] = None,
    ) -> Order:
        """Submit a new order. Must be idempotent on client_order_id."""
        pass

    @abstractmethod
    def cancel_order(self, client_order_id: str) -> bool:
        """Cancel an open pending order."""
        pass

    @abstractmethod
    def get_order_status(self, client_order_id: str) -> Optional[Order]:
        """Fetch current order details and fill status."""
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """Fetch current open positions."""
        pass

    @abstractmethod
    def get_holdings(self) -> List[Holding]:
        """Fetch settled equity holdings."""
        pass

    @abstractmethod
    def get_cash(self) -> float:
        """Fetch available unencumbered cash balance in INR."""
        pass

