---
name: purged-walk-forward-impl
description: Implementation of purged walk-forward CV
metadata:
  type: project
---
Implemented the purged walk-forward cross‑validation module under `src/stock_engine/cv/purged_walk_forward.py`.
- Created `model.yaml` with default CV parameters.
- Added logic for expanding window, purge, and embargo handling.
- Implemented sample‑weight calculation based on label‑period uniqueness.
- Added `plot_folds()` for visualizing training, test, purge, and embargo periods.
- Created tests in `tests/test_purged_walk_forward.py` that cover initialization, basic split, purge/embargo application, sample weights, split info, and edge cases.
- Example usage prints a 16‑fold summary and saves `reports/cv_folds.png`.
- All tests pass (`6/6`).