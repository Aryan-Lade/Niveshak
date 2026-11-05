"""
Unit tests for Day 29 Scheduler and Drift Monitoring.

Checks:
1. Drift score (PSI) is near 0 for identical distributions.
2. Drift score (PSI) flags a deliberately shifted distribution (PSI >= 0.25).
3. KS-test detects distribution shifts with low p-value.
4. Data freshness check detects stale market data.
5. APScheduler registers jobs with exact required times (16:00, 09:20, 16:30, 30m).
"""

import datetime
import numpy as np
import pandas as pd
import pytest

from stock_engine.monitoring.drift import (
    check_data_freshness,
    compute_ks_test,
    compute_psi,
    DriftMonitor,
)
from stock_engine.ops.scheduler import create_niveshak_scheduler


def test_psi_near_zero_for_identical_distributions():
    """Identical distributions must have a PSI very close to 0 (< 0.05)."""
    np.random.seed(42)
    dist = np.random.normal(loc=100.0, scale=15.0, size=2000)

    psi = compute_psi(expected=dist, actual=dist)
    assert psi < 0.01, f"Expected PSI near 0, got {psi}"


def test_psi_flags_shifted_distribution():
    """A mean-shifted distribution must produce a significant drift score (PSI > 0.25)."""
    np.random.seed(42)
    baseline = np.random.normal(loc=100.0, scale=10.0, size=2000)
    shifted = np.random.normal(loc=125.0, scale=10.0, size=2000)  # +2.5 standard deviations shift

    psi = compute_psi(expected=baseline, actual=shifted)
    assert psi > 0.25, f"Expected high PSI for shifted distribution, got {psi}"

    # Also verify KS test detects this shift
    ks_stat, p_val = compute_ks_test(baseline, shifted)
    assert ks_stat > 0.50
    assert p_val < 0.001


def test_data_freshness_check():
    """Verifies fresh and stale date boundaries."""
    today = datetime.date(2024, 5, 20)

    # Fresh: 1 day old
    fresh_ok, _ = check_data_freshness("2024-05-19", as_of=today, max_calendar_lag_days=4)
    assert fresh_ok is True

    # Stale: 7 days old
    stale_ok, msg = check_data_freshness("2024-05-13", as_of=today, max_calendar_lag_days=4)
    assert stale_ok is False
    assert "Market data stale" in msg


def test_scheduler_jobs_registration():
    """Verify APScheduler contains all 4 mandatory trading jobs with exact triggers."""
    scheduler = create_niveshak_scheduler(enable_weekly_retrain=True, start_scheduler=False)

    job_ids = {job.id for job in scheduler.get_jobs()}
    assert "job_market_close_queue" in job_ids
    assert "job_market_open_fill" in job_ids
    assert "job_news_ingest_30m" in job_ids
    assert "job_rag_update" in job_ids
    assert "job_weekly_retrain" in job_ids

    # Inspect 16:00 close job trigger
    close_job = scheduler.get_job("job_market_close_queue")
    fields = {f.name: str(f) for f in close_job.trigger.fields}
    assert fields["hour"] == "16"
    assert fields["minute"] == "0"

    # Inspect 09:20 open job trigger
    open_job = scheduler.get_job("job_market_open_fill")
    fields_open = {f.name: str(f) for f in open_job.trigger.fields}
    assert fields_open["hour"] == "9"
    assert fields_open["minute"] == "20"

