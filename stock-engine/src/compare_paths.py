import os
from pathlib import Path

# Path from save_market_features (using Path)
FEATURES_DIR = Path("../../data_store/features")
output_path = FEATURES_DIR / "market_features.parquet"
path1 = output_path.absolute()

# Path from test_market_save.py (using os.path)
expected_path = "../data_store/features/market_features.parquet"
full_path = os.path.join(os.getcwd(), expected_path)
path2 = os.path.abspath(full_path)

print(f"Path 1 (Path): {path1}")
print(f"Path 2 (os.path): {path2}")
print(f"Paths equal: {path1 == path2}")
print(f"Path 1 exists: {os.path.exists(path1)}")
print(f"Path 2 exists: {os.path.exists(path2)}")