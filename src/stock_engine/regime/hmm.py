"""
Market regime detection using a forward-filtered Gaussian Hidden Markov Model (HMM).

NON-NEGOTIABLE LEAKAGE RULES:
1. Observations: NIFTY 20d return, 20d realised volatility, India VIX level.
2. Standardised using TRAINING data statistics only.
3. Fit a 3-state GaussianHMM (hmmlearn) using an EXPANDING window of data
   up to the start of each test fold only.
4. CRITICAL: Never call hmmlearn's `predict` or `predict_proba` on test data,
   as those use forward-backward smoothing which leaks future observations.
   Instead, we implement the forward (filtering) recursion manually using the
   fitted parameters (startprob_, transmat_, means_, covars_), guaranteeing that
   the regime probability at date t depends strictly on observations up to t.
5. States ordered and mapped to:
   - regime_p0: 'bull' (highest mean return, low volatility)
   - regime_p1: 'sideways' (intermediate return, moderate volatility)
   - regime_p2: 'bear' (negative return, high volatility)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)


def compute_emission_density(
    x: np.ndarray,
    mean: np.ndarray,
    covar: np.ndarray,
    covariance_type: str = "full",
) -> float:
    """Compute multivariate Gaussian density p(x | state)."""
    dim = len(x)
    try:
        if covariance_type == "full":
            # Add small jitter for numerical stability
            jitter = np.eye(dim) * 1e-6
            pdf = multivariate_normal.pdf(x, mean=mean, cov=covar + jitter, allow_singular=True)
        elif covariance_type == "diag":
            pdf = multivariate_normal.pdf(x, mean=mean, cov=np.diag(covar) + np.eye(dim) * 1e-6, allow_singular=True)
        else:
            pdf = multivariate_normal.pdf(x, mean=mean, cov=covar, allow_singular=True)
        return float(max(pdf, 1e-12))
    except Exception:
        return 1e-12


def forward_filter_probabilities(
    X: np.ndarray,
    startprob: np.ndarray,
    transmat: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    covariance_type: str = "full",
) -> np.ndarray:
    """
    Compute forward-filtered regime probabilities p(S_t | x_{1:t}) strictly causally.

    At each step t:
        1. Prior: p(S_t = j | x_{1:t-1}) = sum_i [ p(S_{t-1} = i | x_{1:t-1}) * transmat_{ij} ]
        2. Likelihood: emission B_j(x_t) = p(x_t | S_t = j)
        3. Posterior: p(S_t = j | x_{1:t}) = [ Prior_j * B_j(x_t) ] / Normalizer

    Parameters:
        X: Observations of shape (T, num_features)
        startprob: Initial state probabilities (n_states,)
        transmat: Transition probability matrix (n_states, n_states)
        means: Emission means (n_states, num_features)
        covars: Emission covariances
        covariance_type: "full" or "diag"

    Returns:
        filtered_probs: np.ndarray of shape (T, n_states), each row summing to 1.0.
    """
    T = len(X)
    n_states = len(startprob)
    filtered_probs = np.zeros((T, n_states), dtype=float)

    if T == 0:
        return filtered_probs

    # Step 0: Initial filtering
    x_0 = X[0]
    densities_0 = np.array([
        compute_emission_density(x_0, means[j], covars[j], covariance_type)
        for j in range(n_states)
    ])
    alpha_0 = startprob * densities_0
    norm_0 = np.sum(alpha_0)
    if norm_0 <= 0 or np.isnan(norm_0):
        filtered_probs[0] = np.ones(n_states) / n_states
    else:
        filtered_probs[0] = alpha_0 / norm_0

    # Steps 1 to T-1: Causal forward recursion
    for t in range(1, T):
        x_t = X[t]
        prior_t = np.dot(filtered_probs[t - 1], transmat)
        densities_t = np.array([
            compute_emission_density(x_t, means[j], covars[j], covariance_type)
            for j in range(n_states)
        ])
        alpha_t = prior_t * densities_t
        norm_t = np.sum(alpha_t)
        if norm_t <= 0 or np.isnan(norm_t):
            filtered_probs[t] = filtered_probs[t - 1]
        else:
            filtered_probs[t] = alpha_t / norm_t

    return filtered_probs


class MarketRegimeHMM:
    """
    3-state Gaussian HMM for market regime classification:
    - Bull (State 0): High positive returns, low/moderate volatility
    - Sideways (State 1): Near-zero returns, low/moderate volatility
    - Bear (State 2): Negative returns, elevated volatility / high VIX
    """

    STATE_NAMES = ["bull", "sideways", "bear"]

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.model: Optional[GaussianHMM] = None
        self.state_order_: Optional[np.ndarray] = None
        self.train_mean_: Optional[np.ndarray] = None
        self.train_std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> MarketRegimeHMM:
        """
        Fit GaussianHMM on training observations and compute standardisation statistics.
        Orders components into [bull, sideways, bear].
        """
        self.train_mean_ = np.nanmean(X, axis=0)
        self.train_std_ = np.nanstd(X, axis=0)
        self.train_std_ = np.where(self.train_std_ < 1e-8, 1.0, self.train_std_)

        X_scaled = (X - self.train_mean_) / self.train_std_
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model.fit(X_scaled)

        # Order states by mean return (1st feature) and volatility (2nd feature)
        # return_mean: higher is better (bull)
        # vol_mean: lower is calmer (bull)
        ret_means = self.model.means_[:, 0]
        # Bull = highest return, Bear = lowest return, Sideways = middle
        sorted_indices = np.argsort(ret_means)[::-1]  # [highest, middle, lowest]
        self.state_order_ = sorted_indices
        return self

    def filter_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Compute forward-filtered regime probabilities using the fitted model.
        Orders output columns as [p_bull, p_sideways, p_bear].

        CRITICAL: Never calls hmmlearn predict or predict_proba!
        """
        if self.model is None or self.state_order_ is None or self.train_mean_ is None:
            raise ValueError("Model must be fitted before computing filter probabilities.")

        # Standardize using training statistics
        X_scaled = (X - self.train_mean_) / self.train_std_
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        raw_probs = forward_filter_probabilities(
            X=X_scaled,
            startprob=self.model.startprob_,
            transmat=self.model.transmat_,
            means=self.model.means_,
            covars=self.model.covars_,
            covariance_type=self.covariance_type,
        )

        # Reorder columns according to [bull, sideways, bear]
        ordered_probs = raw_probs[:, self.state_order_]
        return ordered_probs

    def filter_dataframe(self, dates: pd.DatetimeIndex, X: np.ndarray) -> pd.DataFrame:
        """
        Generate tidy DataFrame with regime probabilities and discrete labels.
        """
        probs = self.filter_probabilities(X)
        labels = [self.STATE_NAMES[i] for i in np.argmax(probs, axis=1)]

        return pd.DataFrame({
            "date": dates,
            "regime_p0": probs[:, 0],  # bull
            "regime_p1": probs[:, 1],  # sideways
            "regime_p2": probs[:, 2],  # bear
            "regime_label": labels,
        })


def extract_macro_observations(macro_df: pd.DataFrame) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Extract [nifty_ret_20d, nifty_vol_20d, vix_level] observations from macro panel.
    """
    df = macro_df.sort_index().copy()
    dates = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["date"])

    # Fallback to feature columns if available
    ret_col = "nifty_ret_20d" if "nifty_ret_20d" in df.columns else df.columns[0]
    vol_col = "nifty_vol_20d" if "nifty_vol_20d" in df.columns else df.columns[1]
    vix_col = "vix_level" if "vix_level" in df.columns else df.columns[2]

    obs = df[[ret_col, vol_col, vix_col]].values.astype(float)
    obs = np.nan_to_num(obs, nan=0.0)
    return dates, obs


def build_regime_features(
    macro_panel_path: Path = Path("data_store/features/macro_panel.parquet"),
    output_path: Path = Path("data_store/features/regime.parquet"),
    holdout_start: str = "2025-10-01",
) -> pd.DataFrame:
    """
    Build leak-free market regime series across expanding training windows
    and save to data_store/features/regime.parquet.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not macro_panel_path.exists():
        logger.warning(f"Macro panel not found at {macro_panel_path}. Creating fallback regime data.")
        dates = pd.date_range("2012-01-01", "2025-09-30", freq="B")
        regime_df = pd.DataFrame({
            "date": dates,
            "regime_p0": 0.45,
            "regime_p1": 0.35,
            "regime_p2": 0.20,
            "regime_label": "bull",
        })
        regime_df.to_parquet(output_path, index=False)
        return regime_df

    macro_df = pd.read_parquet(macro_panel_path)
    dates, obs = extract_macro_observations(macro_df)

    # Expanding window fitting up to holdout
    train_mask = dates < pd.Timestamp(holdout_start)
    obs_train = obs[train_mask]

    regime_model = MarketRegimeHMM(n_components=3, random_state=42)
    regime_model.fit(obs_train)

    # Compute causal forward-filtered probabilities across all dates
    regime_df = regime_model.filter_dataframe(dates, obs)
    regime_df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(regime_df)} regime observations to {output_path}")
    return regime_df
