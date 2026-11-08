"""
Operational Risk Alert & Incident Notification Dispatcher.

Features:
- Dispatches formatted alert payloads on critical trading risks:
  * Kill Switch activation
  * Circuit breaker / Drawdown breach
  * Data drift detection
- Built-in cooldown mechanism preventing alert floods.
- Supports generic HTTP webhooks (Slack, Discord, custom monitoring).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import httpx
import yaml

logger = logging.getLogger(__name__)


@dataclass
class AlertPayload:
    event_type: str
    severity: str  # "INFO", "WARNING", "CRITICAL"
    title: str
    details: Dict[str, any]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class AlertDispatcher:
    """
    Manages operational notifications with rate limiting and multi-channel routing.
    """

    def __init__(
        self,
        config_path: Path = Path("configs/alerts.yaml"),
        webhook_url: Optional[str] = None,
        cooldown_seconds: int = 300,
    ):
        self.config_path = config_path
        self._last_alert_time: Dict[str, float] = {}

        # Load config if present
        cfg = {}
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f).get("alerts", {})
            except Exception as e:
                logger.warning(f"Failed to read alerts.yaml: {e}")

        self.enabled = cfg.get("enabled", True)
        self.cooldown_seconds = cooldown_seconds or cfg.get("cooldown_seconds", 300)
        self.webhook_url = (
            webhook_url
            or os.getenv("SLACK_WEBHOOK_URL")
            or os.getenv("DISCORD_WEBHOOK_URL")
            or cfg.get("webhook_url", "")
        )

    def should_dispatch(self, event_type: str) -> bool:
        """Check if cooldown has elapsed for event type."""
        if not self.enabled:
            return False
        now = time.time()
        last = self._last_alert_time.get(event_type, 0.0)
        if (now - last) < self.cooldown_seconds:
            logger.info(f"Alert {event_type} suppressed due to cooldown ({int(now - last)}s < {self.cooldown_seconds}s)")
            return False
        return True

    def dispatch(
        self,
        event_type: str,
        severity: str,
        title: str,
        details: Dict[str, any],
    ) -> bool:
        """
        Dispatch alert to structured logs and optional webhook.
        """
        if not self.should_dispatch(event_type):
            return False

        payload = AlertPayload(
            event_type=event_type,
            severity=severity.upper(),
            title=title,
            details=details,
        )

        # 1. Log alert
        log_msg = f"[RISK ALERT - {payload.severity}] {payload.title} | {json.dumps(payload.details)}"
        if payload.severity == "CRITICAL":
            logger.critical(log_msg)
        elif payload.severity == "WARNING":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        # 2. Webhook dispatch if configured
        dispatched = True
        if self.webhook_url:
            try:
                formatted_body = {
                    "text": f"*{payload.severity}*: {payload.title}\n```{json.dumps(payload.details, indent=2)}```",
                }
                with httpx.Client(timeout=3.0) as client:
                    resp = client.post(self.webhook_url, json=formatted_body)
                    dispatched = (resp.status_code < 300)
            except Exception as e:
                logger.error(f"Failed to deliver alert webhook: {e}")
                dispatched = False

        self._last_alert_time[event_type] = time.time()
        return dispatched
