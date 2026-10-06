import sys
from pathlib import Path
import os

# Check what the market.py load_macro_panel function is looking for
FEATURES_DIR = Path("../data_store/features")
MACRO_PANEL_PATH = FEATURES_DIR / "macro_panel.parquet"

print("Market.py macro panel path:")
print(f"FEATURES_DIR: {FEATURES_DIR}")
print(f"FEATURES_DIR absolute: {FEATURES_DIR.absolute()}")
print(f"MACRO_PANEL_PATH: {MACRO_PANEL_PATH}")
print(f"MACRO_PANEL_PATH absolute: {MACRO_PANEL_PATH.absolute()}")
print(f"MACRO_PANEL_PATH exists: {MACRO_PANEL_PATH.exists()}")

# Also check the actual location
actual_path = Path("../../data_store/features/macro_panel.parquet")
print(f"\nActual path check:")
print(f"Actual path: {actual_path}")
print(f"Actual path absolute: {actual_path.absolute()}")
print(f"Actual path exists: {actual_path.exists()}")

if MACRO_PANEL_PATH.exists():
    print(f"\nContents of {FEATURES_DIR}:")
    for item in FEATURES_DIR.iterdir():
        print(f"  {item.name}")