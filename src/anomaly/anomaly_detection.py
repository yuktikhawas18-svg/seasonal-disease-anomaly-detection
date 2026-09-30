import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast_intervals.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "anomaly_results.csv"
)

print("=" * 60)
print("DENGUE ANOMALY DETECTION")
print("=" * 60)

# ---------------------------------------------------------
# Load forecast interval data
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
    "upper_bound"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# ---------------------------------------------------------
# Remove rows where anomaly detection is impossible
# ---------------------------------------------------------

df = df.dropna(
    subset=[
        "actual",
        "prediction",
        "lower_bound",
        "upper_bound"
    ]
).copy()

# ---------------------------------------------------------
# Calculate deviation from expected value
# ---------------------------------------------------------

df["deviation"] = (
    df["actual"]
    - df["prediction"]
)

df["absolute_deviation"] = (
    df["deviation"]
    .abs()
)

df["relative_deviation_percent"] = (
    df["deviation"]
    / df["prediction"].replace(0, pd.NA)
    * 100
)

# ---------------------------------------------------------
# Detect whether actual value is outside interval
# ---------------------------------------------------------

df["above_upper"] = (
    df["actual"]
    > df["upper_bound"]
)

df["below_lower"] = (
    df["actual"]
    < df["lower_bound"]
)

df["is_anomaly"] = (
    df["above_upper"]
    | df["below_lower"]
)

# ---------------------------------------------------------
# Direction of anomaly
# ---------------------------------------------------------

def determine_direction(row):

    if row["above_upper"]:
        return "HIGH"

    if row["below_lower"]:
        return "LOW"

    return "NORMAL"


df["anomaly_direction"] = (
    df.apply(
        determine_direction,
        axis=1
    )
)

# ---------------------------------------------------------
# Severity
# ---------------------------------------------------------
#
# Initial prototype rules:
#
# NORMAL:
# Actual is inside interval
#
# WATCH:
# Actual is outside interval
#
# WARNING:
# Large deviation from prediction
#
# HIGH:
# Very large deviation
#
# These thresholds are project prototype thresholds,
# NOT medical/public-health guidelines.
# ---------------------------------------------------------

def determine_severity(row):

    if not row["is_anomaly"]:
        return "NORMAL"

    deviation = abs(
        row["relative_deviation_percent"]
    )

    if pd.isna(deviation):
        return "WATCH"

    if deviation >= 100:
        return "HIGH"

    if deviation >= 50:
        return "WARNING"

    return "WATCH"


df["severity"] = (
    df.apply(
        determine_severity,
        axis=1
    )
)

# ---------------------------------------------------------
# Explanation
# ---------------------------------------------------------

def create_explanation(row):

    actual = row["actual"]
    prediction = row["prediction"]
    lower = row["lower_bound"]
    upper = row["upper_bound"]

    if row["severity"] == "NORMAL":

        return (
            f"Actual cases ({actual:.0f}) are within "
            f"the expected forecast interval "
            f"({lower:.0f}–{upper:.0f})."
        )

    if row["anomaly_direction"] == "HIGH":

        return (
            f"Actual cases ({actual:.0f}) are above "
            f"the upper forecast limit ({upper:.0f}). "
            f"Expected value was approximately "
            f"{prediction:.0f}."
        )

    return (
        f"Actual cases ({actual:.0f}) are below "
        f"the lower forecast limit ({lower:.0f}). "
        f"Expected value was approximately "
        f"{prediction:.0f}."
    )


df["explanation"] = (
    df.apply(
        create_explanation,
        axis=1
    )
)

# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

df = df.sort_values(
    [
        "location",
        "date"
    ]
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ANOMALY SUMMARY")
print("=" * 60)

print(
    "\nTotal observations:",
    len(df)
)

print(
    "Anomalies:",
    int(df["is_anomaly"].sum())
)

print(
    "Normal observations:",
    int((~df["is_anomaly"]).sum())
)

print("\nSeverity counts:")

print(
    df["severity"]
    .value_counts()
    .to_string()
)

print("\nAnomaly direction:")

print(
    df["anomaly_direction"]
    .value_counts()
    .to_string()
)

print("\n" + "=" * 60)
print("ANOMALY RESULTS SAVED")
print("=" * 60)

print("\nSaved to:")

print(OUTPUT_PATH)

print("\nSample results:")

print(
    df[
        [
            "location",
            "date",
            "actual",
            "prediction",
            "lower_bound",
            "upper_bound",
            "is_anomaly",
            "anomaly_direction",
            "severity"
        ]
    ]
    .head(15)
    .to_string(index=False)
)