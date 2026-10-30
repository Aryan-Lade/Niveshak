"""
Table-driven unit tests for Day 23 Risk Engine and Persisted Kill Switch.

Checks:
- Table-driven tests for every rule (passing and failing cases)
- Boundary value analysis for daily loss and circuit breaker
- Kill switch SQLite persistence across restarts
- Sizing never exceeds available cash
- Human-readable reasons in all risk decisions
"""

import pytest

from stock_engine.risk.engine import (
    OrderRequest,
    PortfolioState,
    RiskDecision,
    RiskEngine,
    calculate_position_size,
    check_daily_loss_limit,
    check_drawdown_circuit_breaker,
    check_liquidity_filter,
    check_news_risk_flag,
    check_price_sanity,
    check_vix_regime,
)
from stock_engine.risk.kill_switch import KillSwitch


# =========================================================================
# Table-Driven Rule Tests (Passing & Failing for EVERY rule)
# =========================================================================

@pytest.mark.parametrize(
    "avg_val, min_req, expected_pass",
    [
        (100_000_000, 50_000_000, True),   # Pass: 10 Cr > 5 Cr
        (50_000_000, 50_000_000, True),    # Pass: exact boundary
        (49_999_999, 50_000_000, False),   # Fail: just below boundary
        (10_000_000, 50_000_000, False),   # Fail: illiquid stock
    ],
)
def test_liquidity_filter_table(avg_val, min_req, expected_pass):
    passed, reason = check_liquidity_filter(avg_val, min_req)
    assert passed == expected_pass
    if not passed:
        assert reason is not None
        assert "Average daily traded value" in reason


@pytest.mark.parametrize(
    "price, last_close, band_pct, expected_pass",
    [
        (102.0, 100.0, 0.05, True),   # Pass: +2% within 5% band
        (96.0, 100.0, 0.05, True),    # Pass: -4% within 5% band
        (105.0, 100.0, 0.05, True),   # Pass: exact +5% boundary
        (105.5, 100.0, 0.05, False),  # Fail: +5.5% outside band
        (94.5, 100.0, 0.05, False),   # Fail: -5.5% outside band
    ],
)
def test_price_sanity_band_table(price, last_close, band_pct, expected_pass):
    passed, reason = check_price_sanity(price, last_close, band_pct)
    assert passed == expected_pass
    if not passed:
        assert reason is not None
        assert "deviates" in reason


@pytest.mark.parametrize(
    "vix_z, max_z, expected_pass",
    [
        (1.2, 2.5, True),    # Pass: normal regime
        (2.5, 2.5, True),    # Pass: exact boundary
        (2.51, 2.5, False),  # Fail: high volatility spike
        (3.8, 2.5, False),   # Fail: panic regime
        (None, 2.5, True),   # Pass: missing VIX skips filter
    ],
)
def test_vix_regime_table(vix_z, max_z, expected_pass):
    passed, reason = check_vix_regime(vix_z, max_z)
    assert passed == expected_pass
    if not passed:
        assert "VIX z-score" in reason


@pytest.mark.parametrize(
    "daily_pnl_pct, limit_pct, expected_pass",
    [
        (0.01, -0.02, True),     # Pass: profitable day
        (-0.015, -0.02, True),   # Pass: small loss within limit
        (-0.02, -0.02, False),   # Fail: hit -2% limit
        (-0.035, -0.02, False),  # Fail: severe daily drop
    ],
)
def test_daily_loss_limit_table(daily_pnl_pct, limit_pct, expected_pass):
    passed, reason = check_daily_loss_limit(daily_pnl_pct, limit_pct)
    assert passed == expected_pass
    if not passed:
        assert "Daily PnL" in reason


@pytest.mark.parametrize(
    "dd_pct, breaker_pct, expected_pass",
    [
        (-0.05, -0.10, True),    # Pass: 5% drawdown within 10% limit
        (-0.099, -0.10, True),   # Pass: just before trip
        (-0.10, -0.10, False),   # Fail: hits circuit breaker
        (-0.15, -0.10, False),   # Fail: 15% drawdown
    ],
)
def test_drawdown_circuit_breaker_table(dd_pct, breaker_pct, expected_pass):
    passed, reason = check_drawdown_circuit_breaker(dd_pct, breaker_pct)
    assert passed == expected_pass
    if not passed:
        assert "Portfolio drawdown" in reason


@pytest.mark.parametrize(
    "sent, conf, auto_exit, expected_flag, expected_action",
    [
        (-0.85, 0.90, False, True, "REVIEW"),      # Pass: Flagged for review
        (-0.85, 0.90, True, True, "AUTO_EXIT"),     # Pass: Auto-exit configured
        (-0.50, 0.90, False, False, "EXECUTE"),     # Pass: Sentiment not severe
        (-0.85, 0.50, False, False, "EXECUTE"),     # Pass: Confidence too low
    ],
)
def test_news_risk_flag_table(sent, conf, auto_exit, expected_flag, expected_action):
    flagged, action, reason = check_news_risk_flag(
        ticker="RELIANCE.NS",
        rag_sentiment=sent,
        rag_confidence=conf,
        sentiment_thresh=-0.70,
        confidence_thresh=0.70,
        auto_exit=auto_exit,
    )
    assert flagged == expected_flag
    assert action == expected_action


# =========================================================================
# Position Sizing & Cash Constraints
# =========================================================================

def test_position_sizing_never_exceeds_cash():
    """Verify approved shares never exceed available cash balance."""
    equity = 1_000_000.0
    cash = 15_000.0  # Only enough for ~15 shares at 1000 INR
    price = 1_000.0
    atr = 20.0

    qty, reasons = calculate_position_size(
        equity=equity,
        cash=cash,
        price=price,
        atr=atr,
        risk_per_trade_pct=0.01,  # 10,000 risk -> raw shares = 10000 / (2*20) = 250
        atr_multiplier=2.0,
        max_weight_per_stock=0.20, # 200,000 max -> 200 shares
    )

    # Approved quantity must be strictly capped by cash (15 shares)
    assert qty == 15
    assert qty * price <= cash
    assert any("cash cap: 15" in r for r in reasons)


# =========================================================================
# Kill Switch Persistence & Recovery Across Simulated Restart
# =========================================================================

def test_kill_switch_persists_across_restart(tmp_path):
    """
    Test that kill switch state survives process termination and reload from SQLite.
    """
    db_file = tmp_path / "trading_test.db"

    # Step 1: Initialize first instance and trip it
    ks1 = KillSwitch(db_path=db_file)
    assert not ks1.is_tripped()
    ks1.trip(reason="Daily max drawdown breaker tripped during market rout")
    assert ks1.is_tripped()

    # Step 2: Simulate process crash/restart with brand new instance pointing to same DB
    ks2 = KillSwitch(db_path=db_file)
    assert ks2.is_tripped(), "Kill switch state failed to persist across simulated restart!"
    status = ks2.status()
    assert "Daily max drawdown" in status["reason"]

    # Step 3: Attempt reset without confirm=True
    reset_failed = ks2.reset(confirm=False)
    assert not reset_failed
    assert ks2.is_tripped()

    # Step 4: Valid reset with confirm=True
    reset_ok = ks2.reset(confirm=True)
    assert reset_ok
    assert not ks2.is_tripped()

    # Step 5: Verify third fresh instance sees cleared state
    ks3 = KillSwitch(db_path=db_file)
    assert not ks3.is_tripped()


# =========================================================================
# Integrated RiskEngine Evaluation Tests
# =========================================================================

def test_risk_engine_full_evaluation_flow(tmp_path):
    """Integrated test asserting human-readable reasons in every risk decision."""
    db_file = tmp_path / "risk_test.db"
    engine = RiskEngine(db_path=db_file)

    portfolio = PortfolioState(
        total_equity=1_000_000.0,
        cash=500_000.0,
        start_of_day_equity=1_000_000.0,
        peak_equity=1_000_000.0,
        open_positions={},
    )

    # Happy path: valid order
    order_good = OrderRequest(
        ticker="INFY.NS",
        side="BUY",
        intended_price=1500.0,
        last_close=1500.0,
        atr=30.0,
        avg_daily_traded_value=200_000_000.0,
        vix_zscore=0.5,
    )
    decision = engine.evaluate_order(order_good, portfolio)
    assert decision.approved is True
    assert decision.approved_qty > 0
    assert len(decision.reasons) > 0
    assert any("Sized" in r for r in decision.reasons)

    # Rejection: Max positions reached
    portfolio_full = PortfolioState(
        total_equity=1_000_000.0,
        cash=500_000.0,
        start_of_day_equity=1_000_000.0,
        peak_equity=1_000_000.0,
        open_positions={"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0},  # 5 positions
    )
    decision_full = engine.evaluate_order(order_good, portfolio_full)
    assert decision_full.approved is False
    assert decision_full.approved_qty == 0
    assert any("Max positions reached: 5/5" in r for r in decision_full.reasons)
