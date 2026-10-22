"""Signal generation for trading."""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration from a file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_threshold(training_probs: np.ndarray, quantile: float) -> float:
    """Return the probability threshold at the given quantile of training predictions."""
    return float(np.quantile(training_probs, quantile))


def generate_signals(
    predictions: pd.DataFrame,
    threshold: float,
    max_positions: int,
) -> pd.DataFrame:
    """Generate daily signals based on calibrated probabilities.

    Parameters
    ----------
    predictions : pd.DataFrame
        DataFrame with columns ['date', 'ticker', 'prob'] indexed by (date, ticker).
    threshold : float
        Probability threshold for signal activation.
    max_positions : int
        Maximum number of positions per day.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['date', 'ticker', 'signal'] where 'signal' is True for selected trades.
    """
    def select_top(group: pd.DataFrame) -> pd.DataFrame:
        eligible = group[group["prob"] >= threshold]
        if eligible.empty:
            return pd.DataFrame(columns=["date", "ticker", "signal"])
        top = eligible.nlargest(max_positions, "prob")
        top = top["date", "ticker"].copy()
        top["signal"] = True
        return top
    signals = predictions.groupby("date").apply(select_top).reset_index(drop=True)
    return signals
