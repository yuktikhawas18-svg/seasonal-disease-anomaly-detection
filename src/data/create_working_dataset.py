import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dengue_raw_filtered.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dengue_working.csv"
)

print("=" * 60)
print("CREATING DENGUE WORKING DATASET")
print("=" * 60)

# --------------------------------------------------
# 1. Load filtered dengue data
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH, low_memory=False)

print("\nTotal dengue records:", f"{len(df):,}")

# --------------------------------------------------
# 2. Convert case count to numeric
# --------------------------------------------------

df["Current week"] = pd.to_numeric(
    df["Current week"],
    errors="coerce"
)

# --------------------------------------------------
# 3. Remove broad geographic aggregates
# --------------------------------------------------

excluded_locations = [
    "U.S. Residents",
    "South Atlantic",
    "Middle Atlantic",
    "Mountain",
    "West South",
    "East South",
    "New England",
    "Pacific",
    "West North",
    "East North"
]

df = df[
    ~df["Reporting Area"]
    .astype(str)
    .str.strip()
    .isin(excluded_locations)
].copy()

# --------------------------------------------------
# 4. Calculate usable observations
# --------------------------------------------------

location_stats = (
    df.groupby("Reporting Area")
    .agg(
        total_records=("Reporting Area", "size"),
        valid_case_counts=("Current week", "count")
    )
    .reset_index()
)

# Require at least 50 usable observations
eligible = location_stats[
    location_stats["valid_case_counts"] >= 50
].copy()

# Sort by usable observations
eligible = eligible.sort_values(
    "valid_case_counts",
    ascending=False
)

print("\nEligible locations:")
print(
    eligible.head(20).to_string(index=False)
)

# --------------------------------------------------
# 5. Select top 5
# --------------------------------------------------

selected_locations = (
    eligible
    .head(5)["Reporting Area"]
    .tolist()
)

if len(selected_locations) < 5:
    raise ValueError(
        "Fewer than 5 locations have at least 50 "
        "usable observations."
    )

print("\n" + "=" * 60)
print("SELECTED 5 LOCATIONS")
print("=" * 60)

for i, location in enumerate(selected_locations, start=1):
    print(f"{i}. {location}")

# --------------------------------------------------
# 6. Create working dataset
# --------------------------------------------------

working_df = df[
    df["Reporting Area"].isin(selected_locations)
].copy()

# Sort chronologically
working_df = working_df.sort_values(
    [
        "Reporting Area",
        "Current MMWR Year",
        "MMWR WEEK"
    ]
)

# --------------------------------------------------
# 7. Save
# --------------------------------------------------

working_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print("WORKING DATASET CREATED")
print("=" * 60)

print("\nRows:", f"{len(working_df):,}")
print("Columns:", len(working_df.columns))

print("\nSaved to:")
print(OUTPUT_PATH)

print("\nLocations:")
print(
    working_df["Reporting Area"]
    .value_counts()
)