"""
Unit and integration tests for Day 24 BrokerAdapter, PaperBroker, and DailyTradingRunner.

Tests:
1. Idempotency on client_order_id (no duplicate orders).
2. Fills apply realistic Indian transaction costs and slippage.
3. Cash never goes negative under any circumstance.
4. Crash-and-recover restart safety (no duplicate fills on reload).
5. Multi-day simulated run producing consistent orders, fills, and equity.
"""

import sqlite3
import pandas as pd
import pytest

from stock_engine.execution.paper import PaperBroker
from stock_engine.execution.runner import DailyTradingRunner
from stock_engine.risk.engine import RiskEngine
from stock_engine.risk.kill_switch import KillSwitch


def test_order_idempotency_prevents_duplicate_orders(tmp_path):
    """
    Submitting the same client_order_id multiple times must return the existing
    order and never insert duplicate records into SQLite.
    """
    db_file = tmp_path / "paper_test.db"
    broker = PaperBroker(db_path=db_file)

    # First submission
    ord1 = broker.place_order(
        client_order_id="CLIENT_ORD_1001",
        ticker="RELIANCE.NS",
        side="BUY",
        quantity=50,
        price=2500.0,
    )
    assert ord1.client_order_id == "CLIENT_ORD_1001"
    assert ord1.status == "PENDING"

    # Duplicate submission with same ID
    ord2 = broker.place_order(
        client_order_id="CLIENT_ORD_1001",
        ticker="RELIANCE.NS",
        side="BUY",
        quantity=100,  # Different params ignored on idempotent hit
        price=2600.0,
    )
    assert ord2.client_order_id == "CLIENT_ORD_1001"
    assert ord2.quantity == 50  # Must retain original quantity

    # Verify database has strictly 1 record
    with sqlite3.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM orders WHERE client_order_id = 'CLIENT_ORD_1001'")
        assert cur.fetchone()[0] == 1


def test_costs_and_slippage_applied_on_fill(tmp_path):
    """Fills must incorporate slippage on price and deduct transaction costs."""
    db_file = tmp_path / "paper_test.db"
    broker = PaperBroker(db_path=db_file, initial_cash=500_000.0)

    broker.place_order("ORD_BUY_1", "TCS.NS", "BUY", 10, price=3500.0)

    fills = broker.fill_pending_orders(open_prices={"TCS.NS": 3500.0}, fill_date="2024-05-10")
    assert len(fills) == 1
    f = fills[0]

    assert f.status == "FILLED"
    assert f.filled_qty == 10
    # Slippage increases fill price on buy
    assert f.avg_fill_price >= 3500.0
    # Indian statutory charges + brokerage applied
    assert f.total_cost > 0.0

    # Cash must equal initial - (fill_price * qty) - total_cost
    expected_cash = 500_000.0 - (f.avg_fill_price * 10) - f.total_cost
    assert pytest.approx(broker.get_cash(), abs=0.01) == expected_cash


def test_cash_never_goes_negative(tmp_path):
    """An order requiring more cash than available balance is rejected to prevent negative cash."""
    db_file = tmp_path / "paper_test.db"
    broker = PaperBroker(db_path=db_file, initial_cash=10_000.0)

    # Order requires 100 * 2000 = 200,000 INR (far above 10,000 cash)
    broker.place_order("ORD_OVERDRAFT", "INFY.NS", "BUY", 100, price=2000.0)

    fills = broker.fill_pending_orders(open_prices={"INFY.NS": 2000.0}, fill_date="2024-05-10")
    assert len(fills) == 0

    ord_status = broker.get_order_status("ORD_OVERDRAFT")
    assert ord_status.status == "REJECTED"
    assert broker.get_cash() == 10_000.0  # Cash completely untouched
    assert broker.get_cash() >= 0.0


def test_crash_and_recover_restart_safety(tmp_path):
    """A new broker instance created after a simulated crash retains all state and does not duplicate fills."""
    db_file = tmp_path / "paper_test.db"

    # Instance 1: Place and fill order
    broker1 = PaperBroker(db_path=db_file, initial_cash=1_000_000.0)
    broker1.place_order("CRASH_TEST_ORD", "SBIN.NS", "BUY", 20, price=750.0)
    fills1 = broker1.fill_pending_orders(open_prices={"SBIN.NS": 750.0}, fill_date="2024-05-10")
    assert len(fills1) == 1
    cash1 = broker1.get_cash()

    # Instance 2: Brand new instance pointing to same SQLite database
    broker2 = PaperBroker(db_path=db_file)
    assert pytest.approx(broker2.get_cash(), abs=0.01) == cash1

    # Attempting to fill again on same or new day must NOT duplicate fill
    fills2 = broker2.fill_pending_orders(open_prices={"SBIN.NS": 750.0}, fill_date="2024-05-11")
    assert len(fills2) == 0  # No pending orders left to fill
    assert pytest.approx(broker2.get_cash(), abs=0.01) == cash1

    positions = broker2.get_positions()
    assert "SBIN.NS" in positions
    assert positions["SBIN.NS"].quantity == 20


def test_simulated_5_day_trading_run(tmp_path):
    """
    Simulate a full 5-day cycle:
    Day 1: Queues orders at close.
    Day 2: Fills Day 1 orders at open, snapshots equity, queues next orders.
    Days 3-5: Continues cycle, verifying consistent equity marks.
    """
    db_file = tmp_path / "trading_5d.db"
    broker = PaperBroker(db_path=db_file, initial_cash=1_000_000.0)
    kill_switch = KillSwitch(db_path=db_file)
    risk_engine = RiskEngine(db_path=db_file, kill_switch=kill_switch)
    runner = DailyTradingRunner(broker=broker, risk_engine=risk_engine, kill_switch=kill_switch)

    prices = {
        "2024-05-01": {"open": {"INFY.NS": 1400.0}, "close": {"INFY.NS": 1420.0}},
        "2024-05-02": {"open": {"INFY.NS": 1425.0}, "close": {"INFY.NS": 1450.0}},
        "2024-05-03": {"open": {"INFY.NS": 1445.0}, "close": {"INFY.NS": 1430.0}},
        "2024-05-04": {"open": {"INFY.NS": 1435.0}, "close": {"INFY.NS": 1460.0}},
        "2024-05-05": {"open": {"INFY.NS": 1465.0}, "close": {"INFY.NS": 1480.0}},
    }

    signals = pd.DataFrame([
        {"ticker": "INFY.NS", "score": 0.85, "calibrated_prob": 0.62}
    ])

    for day in sorted(prices.keys()):
        p_info = prices[day]
        res = runner.run_daily(
            as_of=day,
            open_prices=p_info["open"],
            close_prices=p_info["close"],
            signals_df=signals,
        )
        assert res["status"] == "COMPLETED"
        assert res["total_equity"] > 0

    # By Day 5, we should have accumulated positions and fills
    positions = broker.get_positions()
    assert "INFY.NS" in positions
    assert positions["INFY.NS"].quantity > 0
    assert broker.get_cash() > 0.0

    # Equity snapshots should contain all 5 days
    with sqlite3.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM equity_snapshots")
        assert cur.fetchone()[0] == 5

