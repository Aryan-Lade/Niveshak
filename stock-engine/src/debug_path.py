import os
from pathlib import Path

# Simulate what's in save_market_features
FEATURES_DIR = Path("../../data_store/features")
print(f"FEATURES_DIR: {FEATURES_DIR}")
print(f"FEATURES_DIR absolute: {FEATURES_DIR.absolute()}")

output_path = FEATURES_DIR / "market_features.parquet"
print(f"output_path: {output_path}")
print(f"output_path absolute: {output_path.absolute()}")
print(f"output_path exists: {output_path.exists()}")

# Check what's in the directory
parent_dir = output_path.parent
print(f"Parent directory: {parent_dir}")
print(f"Parent directory exists: {parent_dir.exists()}")
if parent_dir.exists():
    print("Contents of parent directory:")
    for item in parent_dir.iterdir():
        print(f"  {item.name}")