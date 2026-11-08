from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from stock_engine.ops.alerts import AlertDispatcher, AlertPayload


def test_alert_payload_timestamp():
    p = AlertPayload(
        event_type="TEST",
        severity="WARNING",
        title="Test Alert",
        details={"key": "val"},
    )
    assert p.timestamp != ""
    assert p.severity == "WARNING"


def test_cooldown_suppression():
    dispatcher = AlertDispatcher(cooldown_seconds=60)
    # First dispatch succeeds
    assert dispatcher.dispatch("KILL_SWITCH", "CRITICAL", "Kill Switch Tripped", {"reason": "Manual"}) is True
    # Immediate second dispatch is suppressed
    assert dispatcher.dispatch("KILL_SWITCH", "CRITICAL", "Kill Switch Tripped", {"reason": "Manual"}) is False


def test_webhook_delivery_mock():
    with patch("httpx.Client.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)

        dispatcher = AlertDispatcher(webhook_url="https://hooks.slack.com/services/mock", cooldown_seconds=0)
        res = dispatcher.dispatch("DRIFT_DETECTED", "WARNING", "KS Drift Detected", {"p_value": 0.001})

        assert res is True
        mock_post.assert_called_once()
