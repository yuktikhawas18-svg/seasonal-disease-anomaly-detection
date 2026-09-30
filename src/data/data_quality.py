import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dengue_weekly.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "data_quality_report.csv"
)

print("=" * 60)
print("DENGUE DATA QUALITY CHECK")
print("=" * 60)

df = pd.read_csv(INPUT_PATH, low_memory=False)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["Current week"] = pd.to_numeric(
    df["Current week"],
    errors="coerce"
)

print("\nTotal rows:", f"{len(df):,}")

# --------------------------------------------------
# Overall checks
# --------------------------------------------------

missing_cases = df["Current week"].isna().sum()
negative_cases = (df["Current week"] < 0).sum()
missing_dates = df["date"].isna().sum()

duplicate_rows = df.duplicated(
    subset=[
        "Reporting Area",
        "Current MMWR Year",
        "MMWR WEEK"
    ]
).sum()

print("\n" + "-" * 60)
print("OVERALL QUALITY")
print("-" * 60)

print("Missing case counts :", missing_cases)
print("Negative case counts:", negative_cases)
print("Missing dates       :", missing_dates)
print("Duplicate records   :", duplicate_rows)

# --------------------------------------------------
# Location-wise quality
# --------------------------------------------------

reports = []

for location, group in df.groupby("Reporting Area"):

    group = group.sort_values("date")

    total = len(group)

    valid_cases = group["Current week"].notna().sum()

    missing = group["Current week"].isna().sum()

    negative = (group["Current week"] < 0).sum()

    duplicates = group.duplicated(
        subset=[
            "Current MMWR Year",
            "MMWR WEEK"
        ]
    ).sum()

    # Expected weekly dates
    valid_dates = group["date"].dropna()

    if len(valid_dates) > 1:

        expected_dates = pd.date_range(
            start=valid_dates.min(),
            end=valid_dates.max(),
            freq="7D"
        )

        actual_dates = set(valid_dates)

        missing_weeks = sum(
            date not in actual_dates
            for date in expected_dates
        )

    else:
        missing_weeks = 0

    valid_percentage = (
        valid_cases / total * 100
        if total > 0
        else 0
    )

    reports.append({
        "location": location,
        "total_records": total,
        "valid_case_counts": valid_cases,
        "missing_case_counts": missing,
        "negative_case_counts": negative,
        "duplicate_records": duplicates,
        "missing_weeks": missing_weeks,
        "valid_percentage": round(
            valid_percentage,
            2
        )
    })

# --------------------------------------------------
# Create report
# --------------------------------------------------

quality_report = pd.DataFrame(reports)

quality_report = quality_report.sort_values(
    "valid_case_counts",
    ascending=False
)

print("\n" + "=" * 60)
print("LOCATION QUALITY REPORT")
print("=" * 60)

print(
    quality_report.to_string(index=False)
)

# --------------------------------------------------
# Save
# --------------------------------------------------

quality_report.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print("QUALITY REPORT SAVED")
print("=" * 60)

print("\nSaved to:")
print(OUTPUT_PATH)