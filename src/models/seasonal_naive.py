import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    / "seasonal_naive_results.csv"
)

print("=" * 60)
print("SEASONAL NAIVE BASELINE")
print("=" * 60)

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

# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEASONAL_PERIOD = 52

# Last 20% of chronological observations = test set
TEST_RATIO = 0.20

results = []

locations = (
    df["Reporting Area"]
    .dropna()
    .unique()
)

for location in locations:

    print("\n" + "-" * 60)
    print("Location:", location)
    print("-" * 60)

    temp = (
        df[df["Reporting Area"] == location]
        [["date", "Current week"]]
        .dropna()
        .sort_values("date")
        .drop_duplicates("date")
    )

    # Need at least two seasonal cycles
    if len(temp) < SEASONAL_PERIOD * 2:
        print(
            f"SKIPPED: Only {len(temp)} usable observations."
        )
        continue

    # --------------------------------------------------
    # Chronological split
    # --------------------------------------------------

    split_index = int(
        len(temp) * (1 - TEST_RATIO)
    )

    train = temp.iloc[:split_index].copy()
    test = temp.iloc[split_index:].copy()

    print("Training observations:", len(train))
    print("Testing observations :", len(test))

    # --------------------------------------------------
    # Seasonal Naive prediction
    # --------------------------------------------------

    train_values = train["Current week"].to_numpy()

    predictions = []

    actuals = []

    prediction_dates = []

    for i in range(len(test)):

        test_position = split_index + i

        seasonal_position = (
            test_position - SEASONAL_PERIOD
        )

        if seasonal_position < 0:
            continue

        prediction = (
            df[
                df["Reporting Area"] == location
            ]
            [["date", "Current week"]]
            .dropna()
            .sort_values("date")
            .drop_duplicates("date")
            .iloc[seasonal_position]["Current week"]
        )

        actual = test.iloc[i]["Current week"]

        predictions.append(prediction)
        actuals.append(actual)
        prediction_dates.append(
            test.iloc[i]["date"]
        )

    if len(actuals) == 0:
        print("No valid predictions.")
        continue

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    mae = mean_absolute_error(
        actuals,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actuals,
            predictions
        )
    )

    # MAPE only where actual != 0
    actual_array = np.array(actuals)
    prediction_array = np.array(predictions)

    non_zero = actual_array != 0

    if non_zero.any():

        mape = np.mean(
            np.abs(
                (
                    actual_array[non_zero]
                    - prediction_array[non_zero]
                )
                / actual_array[non_zero]
            )
        ) * 100

    else:
        mape = np.nan

    print("\nBaseline performance:")
    print("MAE :", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("MAPE:", round(mape, 2))

    # --------------------------------------------------
    # Store predictions
    # --------------------------------------------------

    for date, actual, prediction in zip(
        prediction_dates,
        actuals,
        predictions
    ):

        results.append({
            "location": location,
            "date": date,
            "actual": actual,
            "prediction": prediction,
            "absolute_error": abs(
                actual - prediction
            )
        })

# --------------------------------------------------
# Save predictions
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print("SEASONAL NAIVE COMPLETED")
print("=" * 60)

print("\nPredictions saved to:")
print(OUTPUT_PATH)

print(
    "\nTotal predictions:",
    len(results_df)
)