import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "alert_persistence.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "geographic_comparison.csv"
)

print("=" * 60)
print("GEOGRAPHIC COMPARISON")
print("=" * 60)

# ---------------------------------------------------------
# Load persistence results
# ---------------------------------------------------------

df = pd.read_csv(
    INPUT_PATH,
    low_memory=False
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

numeric_columns = [
    "actual",
    "prediction",
    "lower_bound",
    "upper_bound",
    "deviation",
    "absolute_deviation",
    "relative_deviation_percent",
    "anomaly_streak"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

# ---------------------------------------------------------
# Calculate latest available status
# ---------------------------------------------------------

latest = (
    df.sort_values(
        ["location", "date"]
    )
    .groupby(
        "location",
        as_index=False
    )
    .tail(1)
    .copy()
)

print(
    "\nLatest observation for each location:"
)

print(
    latest[
        [
            "location",
            "date",
            "actual",
            "prediction",
            "lower_bound",
            "upper_bound",
            "is_anomaly",
            "anomaly_streak",
            "persistence_severity"
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------
# Geographic comparison metrics
# ---------------------------------------------------------

comparison = latest[
    [
        "location",
        "date",
        "actual",
        "prediction",
        "lower_bound",
        "upper_bound",
        "deviation",
        "absolute_deviation",
        "relative_deviation_percent",
        "is_anomaly",
        "anomaly_direction",
        "anomaly_streak",
        "persistence_severity",
        "persistent_alert"
    ]
].copy()

# ---------------------------------------------------------
# Relative deviation compared with all locations
# ---------------------------------------------------------

comparison["location_relative_rank"] = (
    comparison[
        "relative_deviation_percent"
    ]
    .rank(
        ascending=False,
        method="min"
    )
)

# ---------------------------------------------------------
# Actual case rank
# ---------------------------------------------------------

comparison["actual_case_rank"] = (
    comparison["actual"]
    .rank(
        ascending=False,
        method="min"
    )
)

# ---------------------------------------------------------
# Alert explanation
# ---------------------------------------------------------

def create_geographic_explanation(row):

    location = row["location"]

    actual = row["actual"]

    prediction = row["prediction"]

    severity = row[
        "persistence_severity"
    ]

    if not row["is_anomaly"]:

        return (
            f"{location}: Actual cases ({actual:.0f}) "
            f"are currently within the expected "
            f"forecast range."
        )

    return (
        f"{location}: Actual cases ({actual:.0f}) "
        f"are outside the expected range "
        f"around {prediction:.0f}. "
        f"Current persistence severity: {severity}."
    )


comparison[
    "geographic_explanation"
] = (
    comparison.apply(
        create_geographic_explanation,
        axis=1
    )
)

# ---------------------------------------------------------
# Sort by severity and deviation
# ---------------------------------------------------------

severity_order = {
    "HIGH": 1,
    "WARNING": 2,
    "WATCH": 3,
    "NORMAL": 4
}

comparison[
    "severity_order"
] = comparison[
    "persistence_severity"
].map(
    severity_order
).fillna(5)

comparison = comparison.sort_values(
    [
        "severity_order",
        "relative_deviation_percent"
    ],
    ascending=[
        True,
        False
    ]
)

comparison = comparison.drop(
    columns=["severity_order"]
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

comparison.to_csv(
    OUTPUT_PATH,
    index=False
)

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("GEOGRAPHIC COMPARISON SUMMARY")
print("=" * 60)

print(
    "\nNumber of locations:",
    comparison["location"].nunique()
)

print("\nComparison:")

print(
    comparison[
        [
            "location",
            "actual",
            "prediction",
            "relative_deviation_percent",
            "is_anomaly",
            "anomaly_streak",
            "persistence_severity",
            "location_relative_rank"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 60)
print("GEOGRAPHIC COMPARISON COMPLETED")
print("=" * 60)

print("\nSaved to:")

print(OUTPUT_PATH)