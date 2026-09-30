import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FOLDER = PROJECT_ROOT / "data" / "raw"
PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed"

PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND RAW CSV
# ============================================================

csv_files = list(RAW_FOLDER.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV file found inside data/raw/"
    )

DATA_PATH = csv_files[0]

print("=" * 60)
print("CDC DATA FILTERING")
print("=" * 60)

print("\nRaw file:")
print(DATA_PATH.name)


# ============================================================
# LOAD ONLY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Label",
    "Reporting Area",
    "Current MMWR Year",
    "MMWR WEEK",
    "Current week",
    "Current week, flag"
]

print("\nLoading required columns...")

df = pd.read_csv(
    DATA_PATH,
    usecols=required_columns,
    low_memory=False
)

print("Original rows loaded:", f"{len(df):,}")


# ============================================================
# FILTER DENGUE
# ============================================================

dengue_label = "Dengue virus infections, Dengue"

dengue = df[
    df["Label"].astype(str).str.strip() == dengue_label
].copy()

print("\nDengue records:", f"{len(dengue):,}")


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\nReporting areas found:")
print(
    dengue["Reporting Area"]
    .dropna()
    .astype(str)
    .value_counts()
    .head(30)
)


print("\nYear distribution:")
print(
    dengue["Current MMWR Year"]
    .value_counts()
    .sort_index()
)


# ============================================================
# SAVE DENGUE DATASET
# ============================================================

output_path = PROCESSED_FOLDER / "dengue_raw_filtered.csv"

dengue.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 60)
print("DENGUE DATASET CREATED")
print("=" * 60)

print("\nSaved to:")
print(output_path)

print("\nRows:", f"{len(dengue):,}")
print("Columns:", len(dengue.columns))