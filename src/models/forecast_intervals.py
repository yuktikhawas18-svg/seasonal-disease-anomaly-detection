import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

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
    / "forecast_intervals.csv"
)

print("=" * 60)
print("FORECAST INTERVAL GENERATION")
print("=" * 60)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

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

results = []

TEST_RATIO = 0.20
CONFIDENCE_LEVEL = 0.95

locations = (
    df["Reporting Area"]
    .dropna()
    .unique()
)

# ---------------------------------------------------------
# Process each location
# ---------------------------------------------------------

for location in locations:

    print("\n" + "-" * 60)
    print("Location:", location)
    print("-" * 60)

    temp = (
        df[
            df["Reporting Area"] == location
        ][
            ["date", "Current week"]
        ]
        .dropna()
        .sort_values("date")
        .drop_duplicates("date")
    )

    print(
        "Usable observations:",
        len(temp)
    )

    # Need enough observations for train/test
    if len(temp) < 60:

        print(
            "SKIPPED: Fewer than 60 usable observations."
        )

        continue

    # -----------------------------------------------------
    # Create regular weekly series
    # -----------------------------------------------------

    series = (
        temp
        .set_index("date")["Current week"]
        .asfreq("7D")
    )

    missing_values = series.isna().sum()

    if missing_values > 0:

        print(
            "Missing weekly timestamps:",
            missing_values
        )

        # Model-only interpolation
        series = series.interpolate(
            method="linear",
            limit_direction="both"
        )

    # -----------------------------------------------------
    # Chronological split
    # -----------------------------------------------------

    split_index = int(
        len(series) * (1 - TEST_RATIO)
    )

    train = series.iloc[:split_index]
    test = series.iloc[split_index:]

    print(
        "Training observations:",
        len(train)
    )

    print(
        "Testing observations:",
        len(test)
    )

    if len(test) == 0:

        print(
            "SKIPPED: No test observations."
        )

        continue

    # -----------------------------------------------------
    # Choose model
    # -----------------------------------------------------

    if len(train) >= 104:

        print(
            "Model: Seasonal SARIMA "
            "(52-week seasonality)"
        )

        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 52)

    else:

        print(
            "Model: Non-seasonal SARIMA fallback"
        )

        order = (1, 1, 1)
        seasonal_order = (0, 0, 0, 0)

    # -----------------------------------------------------
    # Train model
    # -----------------------------------------------------

    try:

        print("Training model...")

        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted_model = model.fit(
            disp=False
        )

        # -------------------------------------------------
        # Generate forecast
        # -------------------------------------------------

        forecast = fitted_model.get_forecast(
            steps=len(test)
        )

        prediction = (
            forecast.predicted_mean
        )

        confidence_interval = (
            forecast.conf_int(
                alpha=1 - CONFIDENCE_LEVEL
            )
        )

        lower = confidence_interval.iloc[:, 0]
        upper = confidence_interval.iloc[:, 1]

        # -------------------------------------------------
        # Save results
        # -------------------------------------------------

        for date, actual, pred, low, high in zip(
            test.index,
            test.values,
            prediction.values,
            lower.values,
            upper.values
        ):

            results.append({

                "location": location,

                "date": date,

                "actual": float(actual),

                "prediction": max(
                    0,
                    float(pred)
                ),

                "lower_bound": max(
                    0,
                    float(low)
                ),

                "upper_bound": max(
                    0,
                    float(high)
                ),

                "confidence_level":
                    CONFIDENCE_LEVEL

            })

        print(
            "Forecast interval generated successfully."
        )

    except Exception as error:

        print(
            "MODEL FAILED:"
        )

        print(
            error
        )

# ---------------------------------------------------------
# Create output
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

if results_df.empty:

    raise RuntimeError(
        "\nNo forecast intervals were generated. "
        "Check the usable observations and model errors above."
    )

results_df = results_df.sort_values(
    [
        "location",
        "date"
    ]
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FORECAST INTERVALS COMPLETED")
print("=" * 60)

print(
    "\nTotal forecast records:",
    len(results_df)
)

print(
    "Locations:",
    results_df["location"].nunique()
)

print(
    "\nSaved to:"
)

print(OUTPUT_PATH)

print("\nColumns:")

for column in results_df.columns:
    print("-", column)

print("\nSample results:")

print(
    results_df.head(10).to_string(
        index=False
    )
)