"""
Unit and integration tests for Niveshak Intraday Copilot and Broker Chart Overlay Engine.
"""

import pytest
from fastapi.testclient import TestClient

from stock_engine.api.main import app
from stock_engine.copilot.intraday import IntradayCopilotScanner


@pytest.fixture
def client():
    return TestClient(app)


def test_copilot_scanner_prediction():
    scanner = IntradayCopilotScanner()
    pred = scanner.analyze_ticker("RELIANCE.NS")

    assert pred.ticker == "RELIANCE.NS"
    assert pred.direction in ["UP", "DOWN", "NEUTRAL"]
    assert 50.0 <= pred.confidence_pct <= 99.0
    assert pred.entry_price > 0
    assert pred.target_1 > 0
    assert pred.stop_loss > 0
    assert pred.risk_reward_ratio >= 0
    assert len(pred.broker_compatibility) >= 3
    assert "Groww" in pred.broker_compatibility
    assert "TradingView" in pred.broker_compatibility


def test_copilot_api_endpoint(client):
    resp = client.get("/api/v1/copilot/overlay/TCS.NS")
    assert resp.status_code == 200
    data = resp.json()

    assert data["ticker"] == "TCS.NS"
    assert data["direction"] in ["UP", "DOWN", "NEUTRAL"]
    assert "target_1" in data
    assert "stop_loss" in data
    assert "signals" in data
    assert "vwap" in data["signals"]
    assert "orb_high" in data["signals"]


def test_copilot_api_endpoint_short_ticker(client):
    # Short symbol without .NS should resolve automatically
    resp = client.get("/api/v1/copilot/overlay/INFY")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "INFY.NS"

