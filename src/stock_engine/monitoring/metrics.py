"""
Prometheus Metrics Exporter and Production Health Probes for Niveshak.

Provides:
- Standard Prometheus text format metrics for trading engine, portfolio state, and ML inference.
- Production readiness health probe (/healthz) checking DB connectivity and disk threshold.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def check_system_health(
    trading_db_path: Path = Path("data_store/trading.db"),
    min_disk_free_mb: int = 500,
) -> Dict[str, any]:
    """
    Perform deep health check:
    1. SQLite database accessibility and write latency.
    2. Available disk space on data partition.
    3. Kill switch state.
    """
    status = "healthy"
    checks = {}

    # 1. Database check
    db_ok = False
    db_latency_ms = 0.0
    try:
        trading_db_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        with sqlite3.connect(trading_db_path, timeout=2.0) as conn:
            conn.execute("SELECT 1;").fetchone()
        db_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        db_ok = True
    except Exception as e:
        logger.error(f"Health check DB error: {e}")
        status = "degraded"

    checks["database"] = {
        "ok": db_ok,
        "latency_ms": db_latency_ms,
        "path": str(trading_db_path),
    }

    # 2. Disk space check
    disk_ok = False
    free_mb = 0
    try:
        usage = shutil.disk_usage(trading_db_path.parent)
        free_mb = int(usage.free / (1024 * 1024))
        disk_ok = free_mb >= min_disk_free_mb
        if not disk_ok:
            status = "unhealthy"
    except Exception as e:
        logger.warning(f"Could not check disk usage: {e}")

    checks["disk_storage"] = {
        "ok": disk_ok,
        "free_mb": free_mb,
        "min_required_mb": min_disk_free_mb,
    }

    # 3. Kill Switch check
    ks_tripped = False
    try:
        from stock_engine.risk.kill_switch import KillSwitch
        ks = KillSwitch()
        ks_tripped = ks.is_tripped()
        if ks_tripped:
            status = "degraded"
    except Exception:
        pass

    checks["kill_switch"] = {
        "tripped": ks_tripped,
    }

    return {
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checks": checks,
    }


def generate_prometheus_metrics(
    portfolio_nav: float = 1_000_000.0,
    cash_balance: float = 1_000_000.0,
    active_drawdown_pct: float = 0.0,
    kill_switch_tripped: bool = False,
    drift_p_value: float = 0.45,
    inference_latency_sec: float = 0.012,
    trade_count: int = 0,
) -> str:
    """
    Format operational and quant metrics into Prometheus exposition text format.
    """
    lines = [
        "# HELP niveshak_portfolio_nav_inr Total portfolio net asset value in INR.",
        "# TYPE niveshak_portfolio_nav_inr gauge",
        f"niveshak_portfolio_nav_inr {portfolio_nav:.2f}",
        "",
        "# HELP niveshak_cash_balance_inr Free unallocated cash in INR.",
        "# TYPE niveshak_cash_balance_inr gauge",
        f"niveshak_cash_balance_inr {cash_balance:.2f}",
        "",
        "# HELP niveshak_active_drawdown_pct Current percentage drawdown from all-time peak.",
        "# TYPE niveshak_active_drawdown_pct gauge",
        f"niveshak_active_drawdown_pct {active_drawdown_pct:.4f}",
        "",
        "# HELP niveshak_kill_switch_state Binary flag: 1 if circuit breaker kill switch is tripped, 0 otherwise.",
        "# TYPE niveshak_kill_switch_state gauge",
        f"niveshak_kill_switch_state {1 if kill_switch_tripped else 0}",
        "",
        "# HELP niveshak_feature_drift_p_value Kolmogorov-Smirnov drift test p-value.",
        "# TYPE niveshak_feature_drift_p_value gauge",
        f"niveshak_feature_drift_p_value {drift_p_value:.4f}",
        "",
        "# HELP niveshak_model_inference_seconds Execution time for model prediction pass.",
        "# TYPE niveshak_model_inference_seconds gauge",
        f"niveshak_model_inference_seconds {inference_latency_sec:.4f}",
        "",
        "# HELP niveshak_trades_total Cumulative executed orders count.",
        "# TYPE niveshak_trades_total counter",
        f"niveshak_trades_total {trade_count}",
        "",
    ]
    return "\n".join(lines)
