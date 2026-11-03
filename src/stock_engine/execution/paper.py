"""
SQLite-backed PaperBroker simulating next-open execution with transaction costs and slippage.

Features:
- Full order lifecycle: PENDING -> FILLED / CANCELLED / REJECTED
- Idempotent order submission on client_order_id
- Real statutory Indian equity cost model (STT, GST, SEBI, Stamp duty, DP charges)
- Cash balance protection: cash NEVER goes negative
- Fully restart-safe and persistent across process restarts
"""

from __future__ import annotations

import datetime
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from stock_engine.backtest.costs import CostModel, OrderCost
from stock_engine.execution.broker import BrokerAdapter, Holding, Order, Position

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data_store/trading.db")
INITIAL_CASH_INR = 1_000_000.0  # 10 Lakh INR paper trading seed capital


class PaperBroker(BrokerAdapter):
    """
    Simulated broker for paper trading execution on Indian equities.
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        cost_model: Optional[CostModel] = None,
        initial_cash: float = INITIAL_CASH_INR,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cost_model = cost_model or CostModel()
        self.initial_cash = initial_cash
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    client_order_id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    order_type TEXT NOT NULL,
                    price REAL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    filled_qty INTEGER DEFAULT 0,
                    avg_fill_price REAL DEFAULT 0.0,
                    total_cost REAL DEFAULT 0.0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fills (
                    fill_id TEXT PRIMARY KEY,
                    client_order_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    cost REAL NOT NULL,
                    fill_time TEXT NOT NULL,
                    FOREIGN KEY(client_order_id) REFERENCES orders(client_order_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    ticker TEXT PRIMARY KEY,
                    quantity INTEGER NOT NULL,
                    avg_entry_price REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cash_ledger (
                    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    delta REAL NOT NULL,
                    balance REAL NOT NULL,
                    description TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE NOT NULL,
                    cash REAL NOT NULL,
                    positions_value REAL NOT NULL,
                    total_equity REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    score REAL,
                    calibrated_prob REAL,
                    action TEXT NOT NULL
                )
            """)

            # Seed initial cash ledger if brand new DB
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM cash_ledger")
            if cur.fetchone()[0] == 0:
                now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO cash_ledger (timestamp, delta, balance, description) VALUES (?, ?, ?, ?)",
                    (now_iso, self.initial_cash, self.initial_cash, "Initial capital allocation"),
                )
            conn.commit()

    # -------------------------------------------------------------------------
    # BrokerAdapter Implementation
    # -------------------------------------------------------------------------

    def place_order(
        self,
        client_order_id: str,
        ticker: str,
        side: str,
        quantity: int,
        order_type: str = "LIMIT",
        price: Optional[float] = None,
    ) -> Order:
        """
        Place an order into PENDING state.
        Guarantees idempotency: submitting the same client_order_id returns existing record.
        """
        side_upper = side.upper()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,))
            row = cur.fetchone()
            if row:
                logger.info(f"Idempotent place_order hit for {client_order_id}. Returning existing order.")
                return Order(
                    client_order_id=row["client_order_id"],
                    ticker=row["ticker"],
                    side=row["side"],
                    quantity=row["quantity"],
                    order_type=row["order_type"],
                    price=row["price"],
                    status=row["status"],
                    created_at=row["created_at"],
                    filled_qty=row["filled_qty"],
                    avg_fill_price=row["avg_fill_price"],
                    total_cost=row["total_cost"],
                )

            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO orders (client_order_id, ticker, side, quantity, order_type, price, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?)
                """,
                (client_order_id, ticker, side_upper, quantity, order_type.upper(), price, now_iso),
            )
            conn.commit()

        return Order(
            client_order_id=client_order_id,
            ticker=ticker,
            side=side_upper,
            quantity=quantity,
            order_type=order_type.upper(),
            price=price,
            status="PENDING",
            created_at=now_iso,
        )

    def cancel_order(self, client_order_id: str) -> bool:
        """Cancel a pending order."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT status FROM orders WHERE client_order_id = ?", (client_order_id,))
            row = cur.fetchone()
            if not row or row["status"] != "PENDING":
                return False

            conn.execute(
                "UPDATE orders SET status = 'CANCELLED' WHERE client_order_id = ?",
                (client_order_id,),
            )
            conn.commit()
            return True

    def get_order_status(self, client_order_id: str) -> Optional[Order]:
        """Fetch current order details."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,))
            row = cur.fetchone()
            if not row:
                return None
            return Order(
                client_order_id=row["client_order_id"],
                ticker=row["ticker"],
                side=row["side"],
                quantity=row["quantity"],
                order_type=row["order_type"],
                price=row["price"],
                status=row["status"],
                created_at=row["created_at"],
                filled_qty=row["filled_qty"],
                avg_fill_price=row["avg_fill_price"],
                total_cost=row["total_cost"],
            )

    def get_positions(self) -> Dict[str, Position]:
        """Fetch active open positions with current valuations."""
        positions: Dict[str, Position] = {}
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM positions WHERE quantity > 0")
            for row in cur.fetchall():
                positions[row["ticker"]] = Position(
                    ticker=row["ticker"],
                    quantity=row["quantity"],
                    avg_entry_price=row["avg_entry_price"],
                    current_price=row["avg_entry_price"],
                    unrealized_pnl=0.0,
                )
        return positions

    def get_holdings(self) -> List[Holding]:
        """Fetch settled equity holdings."""
        holdings: List[Holding] = []
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM positions WHERE quantity > 0")
            for row in cur.fetchall():
                market_val = row["quantity"] * row["avg_entry_price"]
                holdings.append(
                    Holding(
                        ticker=row["ticker"],
                        quantity=row["quantity"],
                        avg_cost=row["avg_entry_price"],
                        market_value=market_val,
                    )
                )
        return holdings

    def get_cash(self) -> float:
        """Fetch current cash balance."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT balance FROM cash_ledger ORDER BY entry_id DESC LIMIT 1")
            row = cur.fetchone()
            return float(row["balance"]) if row else 0.0

    # -------------------------------------------------------------------------
    # Next-Open Execution Simulation
    # -------------------------------------------------------------------------

    def fill_pending_orders(
        self,
        open_prices: Dict[str, float],
        fill_date: str,
    ) -> List[Order]:
        """
        Simulate next-day market open executions.
        Fills orders at open price with slippage and realistic transaction costs.
        Enforces zero negative cash guarantee.
        """
        filled_orders: List[Order] = []
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM orders WHERE status = 'PENDING'")
            pending = cur.fetchall()

            current_cash = self.get_cash()

            for ord_row in pending:
                oid = ord_row["client_order_id"]
                ticker = ord_row["ticker"]
                side = ord_row["side"]
                qty = ord_row["quantity"]

                if ticker not in open_prices or open_prices[ticker] <= 0:
                    logger.warning(f"No open price for {ticker}. Order {oid} remains PENDING.")
                    continue

                raw_open = open_prices[ticker]
                cost_side = "buy" if side == "BUY" else "sell"
                cost_obj: OrderCost = self.cost_model.order_cost(cost_side, raw_open, qty)

                # Slippage applied to fill price
                fill_price = raw_open + cost_obj.slippage if side == "BUY" else raw_open - cost_obj.slippage
                order_total_cost = cost_obj.total

                if side == "BUY":
                    total_required = (fill_price * qty) + order_total_cost
                    if total_required > current_cash:
                        # Cannot fill: cash cannot go negative
                        logger.warning(
                            f"Order {oid} rejected: required INR {total_required:,.2f} > cash INR {current_cash:,.2f}"
                        )
                        conn.execute("UPDATE orders SET status = 'REJECTED' WHERE client_order_id = ?", (oid,))
                        continue

                    # Deduct cash
                    current_cash -= total_required
                    conn.execute(
                        "INSERT INTO cash_ledger (timestamp, delta, balance, description) VALUES (?, ?, ?, ?)",
                        (fill_date, -total_required, current_cash, f"Buy fill {qty} {ticker} @ {fill_price:.2f}"),
                    )

                    # Update position
                    cur.execute("SELECT quantity, avg_entry_price FROM positions WHERE ticker = ?", (ticker,))
                    pos_row = cur.fetchone()
                    if pos_row:
                        old_qty = pos_row["quantity"]
                        old_avg = pos_row["avg_entry_price"]
                        new_qty = old_qty + qty
                        new_avg = ((old_qty * old_avg) + (qty * fill_price)) / new_qty
                        conn.execute(
                            "UPDATE positions SET quantity = ?, avg_entry_price = ?, updated_at = ? WHERE ticker = ?",
                            (new_qty, new_avg, fill_date, ticker),
                        )
                    else:
                        conn.execute(
                            "INSERT INTO positions (ticker, quantity, avg_entry_price, updated_at) VALUES (?, ?, ?, ?)",
                            (ticker, qty, fill_price, fill_date),
                        )

                elif side == "SELL":
                    # Check position
                    cur.execute("SELECT quantity, avg_entry_price FROM positions WHERE ticker = ?", (ticker,))
                    pos_row = cur.fetchone()
                    if not pos_row or pos_row["quantity"] < qty:
                        conn.execute("UPDATE orders SET status = 'REJECTED' WHERE client_order_id = ?", (oid,))
                        continue

                    gross_proceeds = fill_price * qty
                    net_proceeds = gross_proceeds - order_total_cost
                    current_cash += net_proceeds
                    conn.execute(
                        "INSERT INTO cash_ledger (timestamp, delta, balance, description) VALUES (?, ?, ?, ?)",
                        (fill_date, net_proceeds, current_cash, f"Sell fill {qty} {ticker} @ {fill_price:.2f}"),
                    )

                    new_qty = pos_row["quantity"] - qty
                    if new_qty > 0:
                        conn.execute(
                            "UPDATE positions SET quantity = ?, updated_at = ? WHERE ticker = ?",
                            (new_qty, fill_date, ticker),
                        )
                    else:
                        conn.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))

                # Record fill and update order
                fill_id = f"fill_{oid}_{int(datetime.datetime.now().timestamp())}"
                conn.execute(
                    """
                    INSERT INTO fills (fill_id, client_order_id, ticker, side, quantity, price, cost, fill_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (fill_id, oid, ticker, side, qty, fill_price, order_total_cost, fill_date),
                )
                conn.execute(
                    """
                    UPDATE orders
                    SET status = 'FILLED', filled_qty = ?, avg_fill_price = ?, total_cost = ?
                    WHERE client_order_id = ?
                    """,
                    (qty, fill_price, order_total_cost, oid),
                )

                filled_orders.append(
                    Order(
                        client_order_id=oid,
                        ticker=ticker,
                        side=side,
                        quantity=qty,
                        price=raw_open,
                        status="FILLED",
                        created_at=ord_row["created_at"],
                        filled_qty=qty,
                        avg_fill_price=fill_price,
                        total_cost=order_total_cost,
                    )
                )

            conn.commit()

        return filled_orders

    def snapshot_equity(
        self,
        as_of_date: str,
        current_prices: Dict[str, float],
    ) -> float:
        """
        Record end-of-day equity mark-to-market snapshot in SQLite.
        """
        cash = self.get_cash()
        positions = self.get_positions()
        pos_val = sum(pos.quantity * current_prices.get(ticker, pos.avg_entry_price) for ticker, pos in positions.items())
        total_equity = cash + pos_val

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO equity_snapshots (date, cash, positions_value, total_equity)
                VALUES (?, ?, ?, ?)
                """,
                (as_of_date, cash, pos_val, total_equity),
            )
            conn.commit()

        logger.info(f"[{as_of_date}] Equity Snapshot: Cash INR {cash:,.2f}, Positions INR {pos_val:,.2f}, Total INR {total_equity:,.2f}")
        return total_equity

