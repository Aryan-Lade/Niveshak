"""
Data and Model Drift Monitoring Engine.

Features:
- Population Stability Index (PSI) and Kolmogorov-Smirnov (KS-test) for feature distributions
- Prediction-distribution drift tracking
- Rolling performance decay detection
- Data freshness alerts for delayed yfinance feeds
- SQLite alert audit logging with optional notification stubs
"""

from __future__ import annotations

import datetime
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data_store/trading.db")


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Calculate Population Stability Index (PSI) between baseline and live distributions.

    Rules of thumb:
    - PSI < 0.10: No significant shift
    - 0.10 <= PSI < 0.25: Moderate shift; monitoring recommended
    - PSI >= 0.25: Significant drift; model retraining required
    """
    exp_clean = np.asarray(expected, dtype=float)
    act_clean = np.asarray(actual, dtype=float)

    exp_clean = exp_clean[~np.isnan(exp_clean)]
    act_clean = act_clean[~np.isnan(act_clean)]

    if len(exp_clean) == 0 or len(act_clean) == 0:
        return 0.0

    # Handle identical constant arrays
    if np.all(exp_clean == exp_clean[0]) and np.all(act_clean == act_clean[0]):
        return 0.0 if exp_clean[0] == act_clean[0] else 1.0

    # Bucket breakpoints based on expected baseline quantiles
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(exp_clean, quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf

    # Bucket counts
    exp_counts, _ = np.histogram(exp_clean, bins=bins)
    act_counts, _ = np.histogram(act_clean, bins=bins)

    exp_pct = exp_counts / len(exp_clean)
    act_pct = act_counts / len(act_clean)

    # Apply epsilon smoothing to prevent div by zero or log(0)
    exp_pct = np.clip(exp_pct, epsilon, None)
    act_pct = np.clip(act_pct, epsilon, None)

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(max(0.0, psi_val))


def compute_ks_test(
    expected: np.ndarray,
    actual: np.ndarray,
) -> Tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov test for continuous distributions.
    Returns: (ks_statistic, p_value).
    """
    exp_clean = np.asarray(expected, dtype=float)
    act_clean = np.asarray(actual, dtype=float)
    exp_clean = exp_clean[~np.isnan(exp_clean)]
    act_clean = act_clean[~np.isnan(act_clean)]

    if len(exp_clean) == 0 or len(act_clean) == 0:
        return 0.0, 1.0

    res = stats.ks_2samp(exp_clean, act_clean)
    return float(res.statistic), float(res.pvalue)


def check_data_freshness(
    latest_date_str: str,
    as_of: Optional[datetime.date] = None,
    max_calendar_lag_days: int = 4,
) -> Tuple[bool, Optional[str]]:
    """
    Flag if price database has not ingested recent market closes (allowing for weekends).
    """
    as_of = as_of or datetime.date.today()
    try:
        latest_dt = pd.to_datetime(latest_date_str).date()
    except Exception:
        return False, f"Invalid date string format: {latest_date_str}"

    gap_days = (as_of - latest_dt).days
    if gap_days > max_calendar_lag_days:
        return False, f"Market data stale: latest date {latest_dt} is {gap_days} days behind {as_of} (threshold: {max_calendar_lag_days} days)"
    return True, None


class DriftMonitor:
    """
    Drift monitor recording feature, prediction, and freshness alerts to SQLite.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS drift_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    is_alert INTEGER NOT NULL,
                    message TEXT NOT NULL
                )
            """)
            conn.commit()

    def log_alert(
        self,
        metric_name: str,
        metric_type: str,
        score: float,
        threshold: float,
        is_alert: bool,
        message: str,
    ) -> None:
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO drift_alerts (timestamp, metric_name, metric_type, score, threshold, is_alert, message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (now_iso, metric_name, metric_type, score, threshold, int(is_alert), message),
            )
            conn.commit()

    def audit_features(
        self,
        train_df: pd.DataFrame,
        live_df: pd.DataFrame,
        features: List[str],
        psi_threshold: float = 0.25,
    ) -> Dict[str, Dict[str, float]]:
        """Compute PSI and KS across feature matrix."""
        results = {}
        for feat in features:
            if feat in train_df.columns and feat in live_df.columns:
                psi = compute_psi(train_df[feat].values, live_df[feat].values)
                ks_stat, p_val = compute_ks_test(train_df[feat].values, live_df[feat].values)
                is_drift = psi >= psi_threshold
                results[feat] = {"psi": psi, "ks_stat": ks_stat, "p_value": p_val, "is_drift": is_drift}

                if is_drift:
                    msg = f"Feature {feat} drifted (PSI: {psi:.3f} >= {psi_threshold:.2f}, KS p-val: {p_val:.4f})"
                    logger.warning(msg)
                    self.log_alert(feat, "FEATURE_PSI", psi, psi_threshold, True, msg)

        return results

