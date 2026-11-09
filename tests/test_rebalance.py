import pytest

from stock_engine.execution.rebalance import PortfolioRebalancer


@pytest.fixture
def rebalancer():
    return PortfolioRebalancer(max_position_weight=0.20, min_trade_value_inr=1000.0)


def test_rebalance_sell_then_buy(rebalancer):
    current_positions = {"TCS.NS": 50}       # 50 shares @ 3000 = 150k
    market_prices = {"TCS.NS": 3000.0, "INFY.NS": 1500.0}
    target_weights = {"TCS.NS": 0.05, "INFY.NS": 0.15}
    cash = 50_000.0  # Total NAV = 200k

    plan = rebalancer.plan_rebalance(
        current_positions=current_positions,
        market_prices=market_prices,
        target_weights=target_weights,
        cash=cash,
    )

    assert plan.nav == 200_000.0
    assert plan.cash_after_rebalance > 0
    # Must have a sell order for TCS and buy order for INFY
    actions = {o.ticker: o.action for o in plan.orders}
    assert actions.get("TCS.NS") == "SELL"
    assert actions.get("INFY.NS") == "BUY"


def test_rebalance_cash_never_negative(rebalancer):
    current_positions = {}
    market_prices = {"RELIANCE.NS": 2500.0, "HDFCBANK.NS": 1600.0}
    target_weights = {"RELIANCE.NS": 0.5, "HDFCBANK.NS": 0.5}
    cash = 10_000.0

    plan = rebalancer.plan_rebalance(
        current_positions=current_positions,
        market_prices=market_prices,
        target_weights=target_weights,
        cash=cash,
    )

    assert plan.cash_after_rebalance >= 0.0
    for order in plan.orders:
        if order.action == "BUY":
            assert order.trade_value <= cash


def test_rebalance_hold_no_trade_below_threshold(rebalancer):
    current_positions = {"TCS.NS": 10}
    market_prices = {"TCS.NS": 3000.0}
    target_weights = {"TCS.NS": 0.10}
    cash = 270_000.0  # Total NAV = 300k, 10% = 30k -> exactly 10 shares!

    plan = rebalancer.plan_rebalance(
        current_positions=current_positions,
        market_prices=market_prices,
        target_weights=target_weights,
        cash=cash,
    )

    tcs_order = next(o for o in plan.orders if o.ticker == "TCS.NS")
    assert tcs_order.action == "HOLD"
    assert tcs_order.delta_shares == 0


def test_api_rebalance_plan_endpoint(monkeypatch):
    from fastapi.testclient import TestClient
    from stock_engine.api.main import app

    monkeypatch.setenv("API_KEY", "test-secret-123")
    client = TestClient(app)

    payload = {
        "target_weights": {"TCS.NS": 0.15, "INFY.NS": 0.10},
        "market_prices": {"TCS.NS": 3500.0, "INFY.NS": 1400.0},
    }
    # Unauthenticated should fail
    unauth_resp = client.post("/portfolio/rebalance-plan", json=payload)
    assert unauth_resp.status_code == 401

    # Authenticated should succeed
    auth_resp = client.post(
        "/portfolio/rebalance-plan",
        json=payload,
        headers={"X-API-Key": "test-secret-123"},
    )
    assert auth_resp.status_code == 200
    data = auth_resp.json()
    assert "nav" in data
    assert "cash_available" in data
    assert "orders" in data
    assert isinstance(data["orders"], list)

