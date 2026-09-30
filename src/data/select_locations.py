import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dengue_raw_filtered.csv"

print("=" * 60)
print("DENGUE LOCATION ANALYSIS")
print("=" * 60)

# Load filtered dengue data
df = pd.read_csv(INPUT_PATH, low_memory=False)

print("\nTotal dengue records:", f"{len(df):,}")

# Convert case count to numeric
df["Current week"] = pd.to_numeric(
    df["Current week"],
    errors="coerce"
)

# Location-wise statistics
location_summary = (
    df.groupby("Reporting Area")
    .agg(
        total_records=("Reporting Area", "size"),
        valid_case_counts=("Current week", "count"),
        total_cases=("Current week", "sum")
    )
    .reset_index()
)

# Calculate percentage of usable case counts
location_summary["valid_percentage"] = (
    location_summary["valid_case_counts"]
    / location_summary["total_records"]
    * 100
)

# Sort by number of valid observations
location_summary = location_summary.sort_values(
    "valid_case_counts",
    ascending=False
)

print("\n" + "=" * 60)
print("LOCATION SUMMARY")
print("=" * 60)

print(
    location_summary.head(30).to_string(index=False)
)

# Save summary
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "location_summary.csv"
)

location_summary.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print("LOCATION SUMMARY SAVED")
print("=" * 60)

print("\nSaved to:")
print(OUTPUT_PATH)