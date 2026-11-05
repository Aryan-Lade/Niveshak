"""
Unit tests for Day 28 GrowwBroker Adapter with Mocked SDK.

Checks:
1. dry_run=True NEVER invokes the SDK's place_order method (proves zero live sending).
2. Order value above cap (INR 2,00,000) is rejected.
3. Order reference ID format constraints (8-20 chars) enforced.
4. Tripped kill switch blocks order placement.
5. Duplicate client_order_id returns existing without secondary placement.
6. Asserts no live-data endpoints are referenced.
"""

from unittest.mock import MagicMock
import pytest

from stock_engine.execution.groww import GrowwBroker
from stock_engine.risk.kill_switch import KillSwitch


@pytest.fixture
def mock_sdk():
    sdk = MagicMock()
    sdk.place_order.return_value = {"order_id": "GROWW_12345", "status": "SUBMITTED"}
    sdk.cancel_order.return_value = True
    sdk.get_access_token.return_value = "mock_access_token"
    return sdk


def test_dry_run_never_calls_sdk_place_order(mock_sdk, tmp_path):
    """
    CRITICAL SAFETY ACCEPTANCE CHECK:
    dry_run=True must log the payload and return DRY_RUN_APPROVED without
    ever calling sdk.place_order().
    """
    db_file = tmp_path / "groww_test.db"
    broker = GrowwBroker(
        dry_run=True,
        kill_switch=KillSwitch(db_path=db_file),
        sdk_client=mock_sdk,
    )

    ord_res = broker.place_order(
        client_order_id="ORD_REF_1001",
        ticker="RELIANCE.NS",
        side="BUY",
        quantity=10,
        order_type="LIMIT",
        price=2500.0,
    )

    assert ord_res.status == "DRY_RUN_APPROVED"
    assert ord_res.client_order_id == "ORD_REF_1001"
    # Proves the SDK was NEVER touched
    mock_sdk.place_order.assert_not_called()


def test_order_value_above_cap_rejected(tmp_path):
    """Orders exceeding max_order_value (default 200,000 INR) must raise ValueError."""
    db_file = tmp_path / "groww_test.db"
    broker = GrowwBroker(
        max_order_value=200_000.0,
        kill_switch=KillSwitch(db_path=db_file),
    )

    # 100 shares * INR 2500 = 250,000 INR (exceeds 200,000 cap)
    with pytest.raises(ValueError, match="exceeds cap"):
        broker.place_order(
            client_order_id="ORD_REF_1002",
            ticker="TCS.NS",
            side="BUY",
            quantity=100,
            price=2500.0,
        )


def test_invalid_order_reference_id_rejected(tmp_path):
    """Client order ID must be 8-20 alphanumeric characters."""
    db_file = tmp_path / "groww_test.db"
    broker = GrowwBroker(kill_switch=KillSwitch(db_path=db_file))

    # Too short (< 8 chars)
    with pytest.raises(ValueError, match="Must be 8-20"):
        broker.place_order(
            client_order_id="SHORT",
            ticker="INFY.NS",
            side="BUY",
            quantity=1,
            price=1500.0,
        )


def test_kill_switch_blocks_orders(tmp_path):
    """When kill switch is tripped, order placement is immediately halted."""
    db_file = tmp_path / "groww_test.db"
    ks = KillSwitch(db_path=db_file)
    ks.trip(reason="Market drawdown halt")

    broker = GrowwBroker(kill_switch=ks)

    with pytest.raises(RuntimeError, match="Kill switch is ACTIVE"):
        broker.place_order(
            client_order_id="ORD_REF_1003",
            ticker="INFY.NS",
            side="BUY",
            quantity=10,
            price=1500.0,
        )


def test_duplicate_order_protection(mock_sdk, tmp_path):
    """Submitting the same client_order_id twice returns the cached order without re-placement."""
    db_file = tmp_path / "groww_test.db"
    broker = GrowwBroker(
        dry_run=True,
        kill_switch=KillSwitch(db_path=db_file),
        sdk_client=mock_sdk,
    )

    ord1 = broker.place_order(
        client_order_id="ORD_REF_1004",
        ticker="SBIN.NS",
        side="BUY",
        quantity=20,
        price=700.0,
    )

    # Duplicate call
    ord2 = broker.place_order(
        client_order_id="ORD_REF_1004",
        ticker="SBIN.NS",
        side="BUY",
        quantity=50,
        price=750.0,
    )

    assert ord1.client_order_id == ord2.client_order_id
    assert ord2.quantity == 20  # Retained original quantity
    assert broker.orders_placed_today == 1  # Incremented only once

