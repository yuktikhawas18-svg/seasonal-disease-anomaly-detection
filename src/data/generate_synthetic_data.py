import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_FOLDER
    / "synthetic_disease_data.csv"
)


# =========================================================
# SETTINGS
# =========================================================

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

START_DATE = "2022-01-03"

NUMBER_OF_WEEKS = 208

LOCATIONS = [
    "Florida",
    "Texas",
    "New York",
    "Maryland",
    "Virginia"
]

DISEASE = "Synthetic Dengue"


# =========================================================
# GENERATE BASELINE DATA
# =========================================================

print("=" * 60)
print("SYNTHETIC DISEASE DATA GENERATOR")
print("=" * 60)

dates = pd.date_range(
    start=START_DATE,
    periods=NUMBER_OF_WEEKS,
    freq="7D"
)

records = []

for location_index, location in enumerate(
    LOCATIONS
):

    # Different baseline for each location
    base_level = (
        20
        + location_index * 8
    )

    # Different seasonal strength
    seasonal_strength = (
        10
        + location_index * 2
    )

    for week_index, date in enumerate(
        dates
    ):

        # -------------------------------------------------
        # Seasonal component
        # -------------------------------------------------

        seasonal_component = (
            seasonal_strength
            * (
                1
                + np.sin(
                    2 * np.pi
                    * week_index
                    / 52
                )
            )
        )

        # -------------------------------------------------
        # Small long-term trend
        # -------------------------------------------------

        trend = (
            week_index * 0.03
        )

        # -------------------------------------------------
        # Random noise
        # -------------------------------------------------

        noise = np.random.normal(
            0,
            3
        )

        expected_cases = (
            base_level
            + seasonal_component
            + trend
            + noise
        )

        case_count = max(
            0,
            round(expected_cases)
        )

        records.append({

            "date": date,

            "disease": DISEASE,

            "location": location,

            "case_count": case_count,

            "population": (
                500000
                + location_index * 100000
            ),

            "source": "synthetic",

            # Ground truth
            "ground_truth_anomaly": False,

            "anomaly_type": "NORMAL"

        })


df = pd.DataFrame(records)


# =========================================================
# INJECT KNOWN ANOMALIES
# =========================================================

print("\nInjecting known anomalies...")


# ---------------------------------------------------------
# 1. Persistent outbreak
# ---------------------------------------------------------

outbreak_location = "Florida"

outbreak_start = (
    pd.Timestamp("2025-06-02")
)

outbreak_end = (
    pd.Timestamp("2025-07-07")
)

mask = (
    (df["location"] == outbreak_location)
    & (df["date"] >= outbreak_start)
    & (df["date"] <= outbreak_end)
)

df.loc[
    mask,
    "case_count"
] = (
    df.loc[
        mask,
        "case_count"
    ]
    * 3
).round().astype(int)

df.loc[
    mask,
    "ground_truth_anomaly"
] = True

df.loc[
    mask,
    "anomaly_type"
] = "PERSISTENT_OUTBREAK"


# ---------------------------------------------------------
# 2. Isolated spike
# ---------------------------------------------------------

spike_location = "Texas"

spike_date = pd.Timestamp(
    "2025-03-03"
)

mask = (
    (df["location"] == spike_location)
    & (df["date"] == spike_date)
)

df.loc[
    mask,
    "case_count"
] = (
    df.loc[
        mask,
        "case_count"
    ]
    * 5
).round().astype(int)

df.loc[
    mask,
    "ground_truth_anomaly"
] = True

df.loc[
    mask,
    "anomaly_type"
] = "ISOLATED_SPIKE"


# ---------------------------------------------------------
# 3. Short anomaly
# ---------------------------------------------------------

short_start = pd.Timestamp(
    "2025-09-01"
)

short_end = pd.Timestamp(
    "2025-09-08"
)

mask = (
    (df["location"] == "New York")
    & (df["date"] >= short_start)
    & (df["date"] <= short_end)
)

df.loc[
    mask,
    "case_count"
] = (
    df.loc[
        mask,
        "case_count"
    ]
    * 2.5
).round().astype(int)

df.loc[
    mask,
    "ground_truth_anomaly"
] = True

df.loc[
    mask,
    "anomaly_type"
] = "SHORT_PERSISTENCE"


# ---------------------------------------------------------
# 4. Reporting spike
# ---------------------------------------------------------

reporting_date = pd.Timestamp(
    "2024-11-04"
)

mask = (
    (df["location"] == "Maryland")
    & (df["date"] == reporting_date)
)

df.loc[
    mask,
    "case_count"
] = (
    df.loc[
        mask,
        "case_count"
    ]
    * 4
).round().astype(int)

df.loc[
    mask,
    "ground_truth_anomaly"
] = True

df.loc[
    mask,
    "anomaly_type"
] = "REPORTING_SPIKE"


# =========================================================
# INJECT MISSING OBSERVATIONS
# =========================================================

print("Injecting missing observations...")

missing_mask = (
    (df["location"] == "Virginia")
    & (
        df["date"].isin([
            pd.Timestamp("2024-04-01"),
            pd.Timestamp("2024-04-08")
        ])
    )
)

df.loc[
    missing_mask,
    "case_count"
] = np.nan


# =========================================================
# ADD DATA QUALITY INFORMATION
# =========================================================

df["data_quality_issue"] = "NONE"

df.loc[
    missing_mask,
    "data_quality_issue"
] = "MISSING_VALUE"


# =========================================================
# SORT DATA
# =========================================================

df = df.sort_values(
    [
        "location",
        "date"
    ]
).reset_index(
    drop=True
)


# =========================================================
# SAVE DATA
# =========================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("SYNTHETIC DATASET CREATED")
print("=" * 60)

print(
    "\nRows:",
    f"{len(df):,}"
)

print(
    "Locations:",
    df["location"].nunique()
)

print(
    "Date range:",
    df["date"].min(),
    "to",
    df["date"].max()
)

print(
    "\nGround-truth anomalies:",
    int(
        df["ground_truth_anomaly"].sum()
    )
)

print(
    "\nAnomaly types:"
)

print(
    df[
        df["ground_truth_anomaly"]
    ]["anomaly_type"]
    .value_counts()
    .to_string()
)

print(
    "\nMissing case values:",
    int(
        df["case_count"].isna().sum()
    )
)

print(
    "\nSaved to:"
)

print(OUTPUT_PATH)

print("\nFirst 10 rows:")

print(
    df.head(10).to_string(
        index=False
    )
)