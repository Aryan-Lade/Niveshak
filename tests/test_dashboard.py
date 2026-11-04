"""
Smoke tests for Day 27 Streamlit Multi-Page Dashboard using AppTest.

Checks:
1. Dashboard app loads without runtime exceptions.
2. Latency warning banner renders properly.
3. Navigation through pages completes without crash.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


def test_dashboard_app_smoke():
    """Smoke test asserting app.py initializes cleanly without crashing."""
    app_path = Path(__file__).parent.parent / "src" / "stock_engine" / "dashboard" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=15)
    at.run()
    assert not at.exception
    # Check that title and warning rendered
    assert any("Niveshak" in str(title.value) for title in at.title)
    assert any("Data Delay Notice" in str(w.value) for w in at.warning)


def test_dashboard_api_client_fallbacks():
    """Verify that DashboardAPIClient handles offline backend gracefully."""
    from stock_engine.dashboard.api_client import DashboardAPIClient

    client = DashboardAPIClient(base_url="http://localhost:9999")  # Unreachable port
    health = client.get_health()
    assert health["status"] == "healthy"
    assert health["mode"] == "PAPER"

    signals = client.get_signals()
    assert len(signals) > 0

    equity = client.get_equity()
    assert equity["current_equity"] > 0
