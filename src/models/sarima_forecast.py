import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    / "sarima_results.csv"
)

print("=" * 60)
print("SARIMA ADVANCED FORECASTING")
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

results = []

# --------------------------------------------------
# Configuration
# --------------------------------------------------

TEST_RATIO = 0.20

# Initial SARIMA configuration.
# We will validate/tune this later.
ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 52)

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

    # SARIMA with 52-week seasonality needs
    # a reasonably long series.
    if len(temp) < 156:

        print(
            f"SKIPPED: Only {len(temp)} observations."
        )

        continue

    # --------------------------------------------------
    # Create weekly time series
    # --------------------------------------------------

    series = (
        temp
        .set_index("date")["Current week"]
        .asfreq("7D")
    )

    # Do not modify the original dataset.
    # Interpolation is used only for this model.
    series = series.interpolate(
        method="linear",
        limit_direction="both"
    )

    # --------------------------------------------------
    # Chronological train/test split
    # --------------------------------------------------

    split_index = int(
        len(series) * (1 - TEST_RATIO)
    )

    train = series.iloc[:split_index]

    test = series.iloc[split_index:]

    print("Training observations:", len(train))
    print("Testing observations :", len(test))

    # --------------------------------------------------
    # Fit SARIMA
    # --------------------------------------------------

    print("\nTraining SARIMA...")

    try:

        model = SARIMAX(
            train,
            order=ORDER,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted_model = model.fit(
            disp=False
        )

        # --------------------------------------------------
        # Forecast test period
        # --------------------------------------------------

        forecast = fitted_model.get_forecast(
            steps=len(test)
        )

        predictions = forecast.predicted_mean

        # Align indexes
        predictions.index = test.index

        # --------------------------------------------------
        # Evaluation
        # --------------------------------------------------

        mae = mean_absolute_error(
            test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                test,
                predictions
            )
        )

        actual_array = test.to_numpy()
        prediction_array = predictions.to_numpy()

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

        print("\nSARIMA performance:")
        print("MAE :", round(mae, 3))
        print("RMSE:", round(rmse, 3))
        print("MAPE:", round(mape, 2))

        # --------------------------------------------------
        # Store results
        # --------------------------------------------------

        for date, actual, prediction in zip(
            test.index,
            test.values,
            predictions.values
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

    except Exception as error:

        print(
            "\nSARIMA failed for",
            location
        )

        print(
            "Reason:",
            error
        )

# --------------------------------------------------
# Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print("SARIMA COMPLETED")
print("=" * 60)

print("\nPredictions saved to:")
print(OUTPUT_PATH)

print(
    "\nTotal predictions:",
    len(results_df)
)