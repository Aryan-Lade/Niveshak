"""
Unit tests for OrderBlock Engine, Sector Flows, OI Buildups, and Top 10 Buy/Sell Selectors.
"""

import pytest
from stock_engine.analytics.orderblock_engine import OrderBlockEngine


@pytest.fixture
def engine():
    return OrderBlockEngine(random_seed=42)


def test_sector_money_flows(engine):
    sectors = engine.get_sector_money_flows()
    assert len(sectors) >= 15
    assert any(s.sector == "IT" for s in sectors)
    assert any(s.sector == "AUTO" for s in sectors)
    # Check change_pct and inflow
    it_sector = next(s for s in sectors if s.sector == "IT")
    assert it_sector.change_pct > 0
    assert it_sector.inflow_cr_inr > 0


def test_top_10_buy_stocks(engine):
    buys = engine.get_top_10_buy_stocks()
    assert len(buys) == 10
    # All top buy stocks should have positive price change or bullish buildup
    for stock in buys:
        assert stock.change_pct > 0
        assert stock.buildup_status in ["LONG BUILD-UP", "SHORT COVERING"]
        assert stock.rating_stars >= 3
        assert len(stock.order_block_zone) == 2


def test_least_10_sell_stocks(engine):
    sells = engine.get_least_10_sell_stocks()
    assert len(sells) == 10
    # Worst performer should be ANANDRATHI
    assert sells[0].ticker == "ANANDRATHI"
    for stock in sells:
        assert stock.change_pct < 0 or "SHORT BUILD-UP" in stock.buildup_status or "LONG UNWINDING" in stock.buildup_status


def test_market_breadth_and_calm_zone(engine):
    breadth = engine.get_market_breadth()
    assert breadth["bullish_count"] == 172
    assert breadth["bearish_count"] == 41
    assert breadth["india_vix"] > 0
    assert breadth["calm_index"] > 50


def test_option_pulse(engine):
    pulse = engine.get_option_pulse("NIFTY")
    assert pulse["underlying"] == "NIFTY"
    assert pulse["pcr"] > 0
    assert len(pulse["strikes"]) == 11
    assert pulse["max_pain_strike"] > 20000


def test_rsi_screener_and_academy(engine):
    rsi_res = engine.get_rsi_screener_results()
    assert "overbought_above_65" in rsi_res
    assert "oversold_below_35" in rsi_res

    academy = engine.get_fo_academy_curriculum()
    assert len(academy) == 4
    assert academy[0]["module_id"] == "MOD_01"


def test_stock_detail_and_tradingview_candles(engine):
    detail = engine.get_stock_detail("TVSMOTOR", timeframe="Daily")
    assert detail["ticker"] == "TVSMOTOR"
    assert detail["price"] > 2000
    assert detail["confidence_score"] >= 80
    assert len(detail["candles"]) >= 40
    # Candlestick keys
    candle = detail["candles"][0]
    assert "time" in candle and "open" in candle and "high" in candle and "low" in candle and "close" in candle and "volume" in candle
    # Markers & AI explanation
    assert len(detail["markers"]) == 3
    assert "TVSMOTOR" in detail["ai_explanation"]
    assert "SMC" in detail["ai_explanation"] or "institutional" in detail["ai_explanation"]
    assert "support_1" in detail and "resistance_1" in detail


def test_realtime_alerts_and_watchlist(engine):
    alerts = engine.get_realtime_alerts()
    assert len(alerts) >= 3
    assert any(a["ticker"] == "TVSMOTOR" for a in alerts)
    assert any(a["sentiment"] == "bullish" for a in alerts)

    watchlist = engine.get_watchlist()
    assert len(watchlist) >= 5
    assert any(w["ticker"] == "TVSMOTOR" for w in watchlist)


