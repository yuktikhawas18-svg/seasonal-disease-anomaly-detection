import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dengue_working.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dengue_weekly.csv"
)

print("=" * 60)
print("CREATING WEEKLY DENGUE DATASET")
print("=" * 60)

# --------------------------------------------------
# 1. Load working dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH, low_memory=False)

print("\nRows loaded:", f"{len(df):,}")

# --------------------------------------------------
# 2. Clean column names
# --------------------------------------------------

df.columns = df.columns.str.strip()

# --------------------------------------------------
# 3. Convert important columns
# --------------------------------------------------

df["Current MMWR Year"] = pd.to_numeric(
    df["Current MMWR Year"],
    errors="coerce"
)

df["MMWR WEEK"] = pd.to_numeric(
    df["MMWR WEEK"],
    errors="coerce"
)

df["Current week"] = pd.to_numeric(
    df["Current week"],
    errors="coerce"
)

# --------------------------------------------------
# 4. Check invalid year/week values
# --------------------------------------------------

invalid_year = (
    df["Current MMWR Year"].isna()
)

invalid_week = (
    df["MMWR WEEK"].isna()
    | (df["MMWR WEEK"] < 1)
    | (df["MMWR WEEK"] > 53)
)

print("\nInvalid year values:", invalid_year.sum())
print("Invalid week values:", invalid_week.sum())

# Remove rows where date cannot be constructed
df = df[
    ~invalid_year
    & ~invalid_week
].copy()

# --------------------------------------------------
# 5. Create weekly date
# --------------------------------------------------
#
# CDC MMWR weeks are epidemiological weeks.
# We use ISO-week conversion to create a
# consistent weekly time index.
# --------------------------------------------------

df["date"] = pd.to_datetime(
    df["Current MMWR Year"].astype(int).astype(str)
    + "-W"
    + df["MMWR WEEK"].astype(int).astype(str).str.zfill(2)
    + "-1",
    format="%G-W%V-%u",
    errors="coerce"
)

print("\nDates that could not be created:",
      df["date"].isna().sum())

# --------------------------------------------------
# 6. Check negative case counts
# --------------------------------------------------

negative_cases = (
    df["Current week"].notna()
    & (df["Current week"] < 0)
)

print(
    "Negative case counts:",
    negative_cases.sum()
)

# Do NOT silently replace negatives.
# Remove only if invalid, and report it.

if negative_cases.any():
    print("\nWARNING: Negative case counts found.")
    print(df.loc[
        negative_cases,
        ["Reporting Area", "Current MMWR Year",
         "MMWR WEEK", "Current week"]
    ].head())

# --------------------------------------------------
# 7. Check duplicate location-week records
# --------------------------------------------------

duplicate_mask = df.duplicated(
    subset=[
        "Reporting Area",
        "Current MMWR Year",
        "MMWR WEEK"
    ],
    keep=False
)

print(
    "\nDuplicate location-week rows:",
    duplicate_mask.sum()
)

# --------------------------------------------------
# 8. Sort
# --------------------------------------------------

df = df.sort_values(
    ["Reporting Area", "date"]
).reset_index(drop=True)

# --------------------------------------------------
# 9. Save
# --------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print("WEEKLY DATASET CREATED")
print("=" * 60)

print("\nRows:", f"{len(df):,}")
print("Columns:", len(df.columns))

print("\nDate range:")
print("Start:", df["date"].min())
print("End  :", df["date"].max())

print("\nRecords by location:")
print(
    df["Reporting Area"]
    .value_counts()
)

print("\nSaved to:")
print(OUTPUT_PATH)