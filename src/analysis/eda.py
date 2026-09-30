import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dengue_weekly.csv"
)

FIGURE_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "figures"
)

FIGURE_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 60)
print("DENGUE EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

df = pd.read_csv(
    INPUT_PATH,
    low_memory=False
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["Current week"] = pd.to_numeric(
    df["Current week"],
    errors="coerce"
)

print("\nRows:", f"{len(df):,}")

# --------------------------------------------------
# 2. Basic statistics
# --------------------------------------------------

print("\n" + "-" * 60)
print("BASIC STATISTICS")
print("-" * 60)

print(
    df.groupby("Reporting Area")["Current week"]
    .agg([
        "count",
        "mean",
        "median",
        "std",
        "min",
        "max"
    ])
    .round(2)
)

# --------------------------------------------------
# 3. Total cases by location
# --------------------------------------------------

print("\n" + "-" * 60)
print("TOTAL REPORTED CASES")
print("-" * 60)

total_cases = (
    df.groupby("Reporting Area")["Current week"]
    .sum()
    .sort_values(ascending=False)
)

print(total_cases)

# --------------------------------------------------
# 4. Plot time series for each location
# --------------------------------------------------

locations = df["Reporting Area"].dropna().unique()

for location in locations:

    location_df = (
        df[df["Reporting Area"] == location]
        .sort_values("date")
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        location_df["date"],
        location_df["Current week"],
        marker="o",
        markersize=2,
        linewidth=1
    )

    plt.title(
        f"Weekly Dengue Cases — {location}"
    )

    plt.xlabel("Date")
    plt.ylabel("Reported Cases")

    plt.xticks(rotation=45)

    plt.tight_layout()

    safe_name = (
        str(location)
        .replace(" ", "_")
        .replace("/", "_")
    )

    output_file = (
        FIGURE_FOLDER
        / f"{safe_name}_time_series.png"
    )

    plt.savefig(
        output_file,
        dpi=150
    )

    plt.close()

    print(
        f"Saved plot: {output_file.name}"
    )

# --------------------------------------------------
# 5. Compare locations
# --------------------------------------------------

location_weekly = (
    df.groupby(
        ["date", "Reporting Area"],
        as_index=False
    )["Current week"]
    .sum()
)

plt.figure(figsize=(13, 6))

for location in locations:

    temp = location_weekly[
        location_weekly["Reporting Area"] == location
    ]

    plt.plot(
        temp["date"],
        temp["Current week"],
        label=location
    )

plt.title(
    "Weekly Dengue Cases — Location Comparison"
)

plt.xlabel("Date")
plt.ylabel("Reported Cases")

plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()

comparison_file = (
    FIGURE_FOLDER
    / "location_comparison.png"
)

plt.savefig(
    comparison_file,
    dpi=150
)

plt.close()

print(
    f"\nSaved comparison plot: "
    f"{comparison_file}"
)

# --------------------------------------------------
# 6. Monthly pattern
# --------------------------------------------------

df["month"] = df["date"].dt.month

monthly = (
    df.groupby(
        ["Reporting Area", "month"]
    )["Current week"]
    .mean()
    .reset_index()
)

monthly_file = (
    FIGURE_FOLDER
    / "monthly_pattern.csv"
)

monthly.to_csv(
    monthly_file,
    index=False
)

print(
    f"Saved monthly pattern: "
    f"{monthly_file}"
)

print("\n" + "=" * 60)
print("EDA COMPLETED")
print("=" * 60)