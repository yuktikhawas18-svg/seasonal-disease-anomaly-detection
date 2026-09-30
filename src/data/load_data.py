import pandas as pd
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw data folder
RAW_FOLDER = PROJECT_ROOT / "data" / "raw"

# Find CSV files
csv_files = list(RAW_FOLDER.glob("*.csv"))

print("=" * 60)
print("CDC NNDSS DATA LOADING")
print("=" * 60)

print("\nRaw data folder:")
print(RAW_FOLDER)

print("\nCSV files found:")

for file in csv_files:
    print("-", file.name)


# Check whether a CSV exists
if not csv_files:
    raise FileNotFoundError(
        "No CSV file found inside data/raw/"
    )

# Use the first CSV file
DATA_PATH = csv_files[0]

print("\nUsing dataset:")
print(DATA_PATH)

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("\n" + "=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

print(f"\nRows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

print("\nColumn names:")

for column in df.columns:
    print("-", column)

print("\nFirst 5 rows:")

print(df.head())