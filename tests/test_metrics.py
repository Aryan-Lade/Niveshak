from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from stock_engine.api.main import app
from stock_engine.monitoring.metrics import check_system_health, generate_prometheus_metrics


@pytest.fixture
def client():
    return TestClient(app)


def test_generate_prometheus_metrics_format():
    text = generate_prometheus_metrics(
        portfolio_nav=1_050_000.0,
        cash_balance=800_000.0,
        active_drawdown_pct=0.035,
        kill_switch_tripped=False,
        trade_count=12,
    )
    assert "niveshak_portfolio_nav_inr 1050000.00" in text
    assert "niveshak_cash_balance_inr 800000.00" in text
    assert "niveshak_active_drawdown_pct 0.0350" in text
    assert "niveshak_kill_switch_state 0" in text
    assert "niveshak_trades_total 12" in text


def test_check_system_health(tmp_path):
    db_file = tmp_path / "test_trading.db"
    health = check_system_health(trading_db_path=db_file)
    assert health["status"] in ["healthy", "degraded"]
    assert "database" in health["checks"]
    assert health["checks"]["database"]["ok"] is True
    assert "disk_storage" in health["checks"]


def test_api_healthz_endpoint(client):
    res = client.get("/healthz")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "checks" in data


def test_api_metrics_endpoint(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "niveshak_portfolio_nav_inr" in res.text
    assert "niveshak_kill_switch_state" in res.text
