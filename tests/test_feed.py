import pytest
from fastapi.testclient import TestClient

from stock_engine.api.main import app
from stock_engine.execution.feed import MarketTickSimulator


def test_tick_simulator_next_tick():
    sim = MarketTickSimulator(ticker="RELIANCE.NS", base_price=2500.0, seed=42)
    tick = sim.next_tick()

    assert tick.ticker == "RELIANCE.NS"
    assert tick.ltp > 0
    assert len(tick.bids) == 5
    assert len(tick.asks) == 5
    # Best ask must be >= best bid
    assert tick.asks[0].price >= tick.bids[0].price
    assert tick.timestamp != ""


@pytest.mark.asyncio
async def test_tick_simulator_stream():
    sim = MarketTickSimulator(ticker="TCS.NS", base_price=3500.0, tick_interval_sec=0.01, seed=123)
    ticks = []
    async for tick in sim.stream_ticks(max_ticks=5):
        ticks.append(tick)

    assert len(ticks) == 5
    assert ticks[0]["ticker"] == "TCS.NS"
    assert "bids" in ticks[0]
    assert "asks" in ticks[0]


def test_websocket_ticks_endpoint():
    client = TestClient(app)
    with client.websocket_connect("/ws/ticks/INFY.NS") as websocket:
        data = websocket.receive_json()
        assert data["ticker"] == "INFY.NS"
        assert "ltp" in data
        assert len(data["bids"]) == 5
