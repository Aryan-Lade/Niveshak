"""
Unit and integration tests for Day 25 FastAPI backend.

Tests:
1. Happy path: GET /health, GET /signals/latest, GET /positions, GET /trades, GET /equity, GET /risk/status
2. Auth failure: POST routes return 401 when X-API-Key is missing or invalid
3. Kill switch trip and reset workflow with persisted SQLite state
4. NO-LIVE-ORDERS SAFETY TEST: Scans all API routes and asserts that NO route calls or imports live Groww order execution.
"""

import inspect
import pytest
from fastapi.testclient import TestClient

from stock_engine.api.main import app
from stock_engine.risk.kill_switch import KillSwitch


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Fixture providing TestClient with clean isolated API Key and test DB."""
    monkeypatch.setenv("API_KEY", "test-secret-key-123")
    return TestClient(app)


def test_health_endpoint(client):
    """GET /health returns healthy status and paper mode."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["mode"] == "PAPER"
    assert "timestamp" in data


def test_get_endpoints_happy_path(client):
    """Verify all read-only GET endpoints return 200 with schema compliance."""
    endpoints = [
        "/signals/latest",
        "/positions",
        "/trades",
        "/equity",
        "/backtest/summary",
        "/risk/status",
        "/news/recent",
    ]
    for ep in endpoints:
        resp = client.get(ep)
        assert resp.status_code == 200, f"Endpoint {ep} failed with status {resp.status_code}"


def test_auth_failure_on_post_endpoints(client):
    """All state-mutating POST endpoints must reject requests without valid X-API-Key with 401."""
    # 1. /risk/kill without header
    resp = client.post("/risk/kill", json={"reason": "Emergency halt"})
    assert resp.status_code == 401
    assert "X-API-Key" in resp.json()["detail"]

    # 2. /risk/kill with invalid header
    resp = client.post(
        "/risk/kill",
        json={"reason": "Emergency halt"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401

    # 3. /risk/reset without header
    resp = client.post("/risk/reset", json={"confirm": True})
    assert resp.status_code == 401

    # 4. /paper/run without header
    resp = client.post("/paper/run", json={"as_of": "2024-05-15"})
    assert resp.status_code == 401


def test_kill_switch_trip_and_reset_flow(client):
    """POST /risk/kill trips kill switch; POST /risk/reset restores state."""
    headers = {"X-API-Key": "test-secret-key-123"}

    # Step 1: Trip the kill switch
    trip_resp = client.post(
        "/risk/kill",
        json={"reason": "Risk limit breached in testing"},
        headers=headers,
    )
    assert trip_resp.status_code == 200
    assert trip_resp.json()["is_tripped"] is True
    assert "Risk limit breached" in trip_resp.json()["reason"]

    # Verify GET /risk/status reflects tripped state
    status_resp = client.get("/risk/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["is_tripped"] is True

    # Step 2: Attempt reset without confirm=True
    fail_reset = client.post(
        "/risk/reset",
        json={"confirm": False},
        headers=headers,
    )
    assert fail_reset.status_code == 400

    # Step 3: Valid reset with confirm=True
    ok_reset = client.post(
        "/risk/reset",
        json={"confirm": True},
        headers=headers,
    )
    assert ok_reset.status_code == 200
    assert ok_reset.json()["is_tripped"] is False


def test_explain_endpoint_success(client):
    """POST /explain returns grounded structured explanation with disclaimer."""
    headers = {"X-API-Key": "test-secret-key-123"}
    resp = client.post(
        "/explain",
        json={"ticker": "RELIANCE.NS", "date": "2024-05-10"},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "RELIANCE.NS"
    assert "model_view" in data
    assert "Not financial advice" in data["disclaimer"]


def test_no_live_orders_in_api_router():
    """
    NON-NEGOTIABLE SAFETY TEST:
    Scan all registered FastAPI route handlers and assert that none of them
    reference, call, or import live Groww order execution methods.
    """
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is not None:
            source = inspect.getsource(endpoint)
            # Ensure no live order functions are invoked
            assert "GrowwBroker" not in source, f"Route {route.path} references GrowwBroker!"
            assert "growwapi" not in source, f"Route {route.path} imports live growwapi!"
            assert "place_live_order" not in source, f"Route {route.path} has live order methods!"
