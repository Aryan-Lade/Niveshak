import numpy as np
import pytest
from stock_engine.models.calibration import Calibrator, calibrate


def test_calibrator_fits_only_on_training():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.1, 0.8, 0.2, 0.7])
    cal = Calibrator()
    cal.fit(y_true, y_pred)
    calibrated = cal.transform(y_pred)
    assert calibrated.shape == y_pred.shape
    assert not np.any(np.isnan(calibrated))
    assert not np.any(np.isinf(calibrated))


def test_calibrate_function_returns_array():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.1, 0.8, 0.2, 0.7])
    out = calibrate(y_true, y_pred)
    assert isinstance(out, np.ndarray)
    assert out.shape == y_pred.shape
