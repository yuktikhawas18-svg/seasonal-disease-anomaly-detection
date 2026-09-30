import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "anomaly_results.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "alert_persistence.csv"
)

print("=" * 60)
print("ALERT PERSISTENCE ANALYSIS")
print("=" * 60)

# ---------------------------------------------------------
# Load anomaly results
# ---------------------------------------------------------

df = pd.read_csv(
    INPUT_PATH,
    low_memory=False
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = df.sort_values(
    ["location", "date"]
).reset_index(drop=True)

# ---------------------------------------------------------
# Validate anomaly column
# ---------------------------------------------------------

df["is_anomaly"] = (
    df["is_anomaly"]
    .astype(str)
    .str.lower()
    .map({
        "true": True,
        "false": False
    })
)

# Missing values are treated as non-anomalous
df["is_anomaly"] = (
    df["is_anomaly"]
    .fillna(False)
)

# ---------------------------------------------------------
# Calculate consecutive anomaly streak
# ---------------------------------------------------------

df["anomaly_streak"] = 0

for location in df["location"].dropna().unique():

    location_mask = (
        df["location"] == location
    )

    streak = 0

    for index in df.index[location_mask]:

        if df.loc[index, "is_anomaly"]:

            streak += 1

        else:

            streak = 0

        df.loc[
            index,
            "anomaly_streak"
        ] = streak

# ---------------------------------------------------------
# Persistence severity
# ---------------------------------------------------------

def persistence_severity(row):

    if not row["is_anomaly"]:
        return "NORMAL"

    streak = row["anomaly_streak"]

    if streak >= 3:
        return "HIGH"

    if streak == 2:
        return "WARNING"

    return "WATCH"


df["persistence_severity"] = (
    df.apply(
        persistence_severity,
        axis=1
    )
)

# ---------------------------------------------------------
# Persistence status
# ---------------------------------------------------------

df["persistent_alert"] = (
    df["anomaly_streak"] >= 2
)

# ---------------------------------------------------------
# Alert explanation
# ---------------------------------------------------------

def create_alert_explanation(row):

    if not row["is_anomaly"]:

        return (
            "No statistical anomaly detected."
        )

    streak = int(
        row["anomaly_streak"]
    )

    if streak == 1:

        return (
            "Statistical anomaly detected for "
            "1 consecutive week. Monitoring required."
        )

    if streak == 2:

        return (
            "Statistical anomaly detected for "
            "2 consecutive weeks. Persistent "
            "signal requiring investigation."
        )

    return (
        f"Statistical anomaly detected for "
        f"{streak} consecutive weeks. "
        f"Persistent high-level statistical signal "
        f"requiring investigation."
    )


df["alert_explanation"] = (
    df.apply(
        create_alert_explanation,
        axis=1
    )
)

# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PERSISTENCE SUMMARY")
print("=" * 60)

print(
    "\nTotal observations:",
    len(df)
)

print(
    "Anomalous observations:",
    int(df["is_anomaly"].sum())
)

print(
    "Persistent alerts:",
    int(df["persistent_alert"].sum())
)

print("\nPersistence severity:")

print(
    df["persistence_severity"]
    .value_counts()
    .to_string()
)

print("\nMaximum anomaly streak by location:")

max_streak = (
    df.groupby("location")["anomaly_streak"]
    .max()
    .sort_values(
        ascending=False
    )
)

print(max_streak.to_string())

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ALERT PERSISTENCE COMPLETED")
print("=" * 60)

print("\nSaved to:")

print(OUTPUT_PATH)

print("\nSample results:")

print(
    df[
        [
            "location",
            "date",
            "is_anomaly",
            "anomaly_streak",
            "persistence_severity",
            "persistent_alert",
            "alert_explanation"
        ]
    ]
    .head(20)
    .to_string(index=False)
)