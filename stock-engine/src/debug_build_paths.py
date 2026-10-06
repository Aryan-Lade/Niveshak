import sys
from pathlib import Path
import os

# Simulate what's in build.py
FEATURES_DIR = Path("../data_store/features")
TECHNICAL_FEATURES_PATH = FEATURES_DIR / "technical_features.parquet"
MARKET_FEATURES_PATH = FEATURES_DIR / "market_features.parquet"
OUTPUT_FEATURES_PATH = FEATURES_DIR / "features.parquet"

print("Build.py paths:")
print(f"FEATURES_DIR: {FEATURES_DIR}")
print(f"FEATURES_DIR absolute: {FEATURES_DIR.absolute()}")
print(f"TECHNICAL_FEATURES_PATH: {TECHNICAL_FEATURES_PATH}")
print(f"TECHNICAL_FEATURES_PATH absolute: {TECHNICAL_FEATURES_PATH.absolute()}")
print(f"MARKET_FEATURES_PATH: {MARKET_FEATURES_PATH}")
print(f"MARKET_FEATURES_PATH absolute: {MARKET_FEATURES_PATH.absolute()}")
print(f"OUTPUT_FEATURES_PATH: {OUTPUT_FEATURES_PATH}")
print(f"OUTPUT_FEATURES_PATH absolute: {OUTPUT_FEATURES_PATH.absolute()}")

print("\nFile existence:")
print(f"Technical features exists: {TECHNICAL_FEATURES_PATH.exists()}")
print(f"Market features exists: {MARKET_FEATURES_PATH.exists()}")
print(f"Output features exists: {OUTPUT_FEATURES_PATH.exists()}")

# Check what's in the features directory
if FEATURES_DIR.exists():
    print(f"\nContents of {FEATURES_DIR}:")
    for item in FEATURES_DIR.iterdir():
        print(f"  {item.name}")
else:
    print(f"\n{FEATURES_DIR} does not exist")