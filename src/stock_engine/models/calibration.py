"""Calibration utilities for model predictions."""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from scipy.special import expit


class Calibrator:
    """Selects and applies a probabilistic calibration method.

    The calibrator chooses between isotonic regression and Platt scaling
    by comparing the log loss on the training predictions. The chosen model
    is stored and applied to subsequent predictions.
    """

    def __init__(self) -> None:
        self._method: str | None = None
        self._isotonic: IsotonicRegression | None = None
        self._platt_alpha: float | None = None
        self._platt_beta: float | None = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Fit calibration using training predictions only.

        Parameters
        ----------
        y_true : np.ndarray
            Ground‑truth binary labels.
        y_pred : np.ndarray
            Predicted probabilities from the base model.
        """
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(y_pred, y_true)
        iso_loss = log_loss(y_true, isotonic.transform(y_pred))

        # Fit Platt scaling via simple logistic regression
        alpha = 1.0
        beta = 0.0
        x = y_pred
        y = y_true
        lr = 0.01
        for _ in range(1000):
            z = alpha * x + beta
            p = expit(z)
            grad_alpha = np.mean((p - y) * x)
            grad_beta = np.mean(p - y)
            alpha -= lr * grad_alpha
            beta -= lr * grad_beta
        platt_pred = expit(alpha * y_pred + beta)
        platt_loss = log_loss(y_true, platt_pred)

        if platt_loss <= iso_loss:
            self._method = "platt"
            self._platt_alpha = alpha
            self._platt_beta = beta
        else:
            self._method = "isotonic"
            self._isotonic = isotonic

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """Apply the selected calibration to predictions.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted probabilities from the base model.

        Returns
        -------
        np.ndarray
            Calibrated probabilities.
        """
        if self._method == "platt":
            return expit(self._platt_alpha * y_pred + self._platt_beta)
        if self._method == "isotonic" and self._isotonic is not None:
            return self._isotonic.transform(y_pred)
        raise RuntimeError("Calibrator not fitted.")


# Public convenience function

def calibrate(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calibrate predictions using the best of isotonic or Platt scaling.

    The function creates a :class:`Calibrator`, fits it on the supplied
    training data and returns the calibrated probabilities.
    """
    cal = Calibrator()
    cal.fit(y_true, y_pred)
    return cal.transform(y_pred)
