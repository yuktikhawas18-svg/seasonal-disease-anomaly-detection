import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
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
    / "xgboost_results.csv"
)

print("=" * 60)
print("XGBOOST DENGUE FORECASTING")
print("=" * 60)

df = pd.read_csv(INPUT_PATH, low_memory=False)

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

# --------------------------------------------------
# Feature creation
# --------------------------------------------------

def create_features(data):

    data = data.copy()

    data["lag_1"] = data["Current week"].shift(1)
    data["lag_2"] = data["Current week"].shift(2)
    data["lag_3"] = data["Current week"].shift(3)
    data["lag_4"] = data["Current week"].shift(4)

    # Same period approximately one year earlier
    data["lag_52"] = data["Current week"].shift(52)

    # Rolling statistics use only past observations
    data["rolling_mean_4"] = (
        data["Current week"]
        .shift(1)
        .rolling(4)
        .mean()
    )

    data["rolling_mean_12"] = (
        data["Current week"]
        .shift(1)
        .rolling(12)
        .mean()
    )

    data["rolling_std_12"] = (
        data["Current week"]
        .shift(1)
        .rolling(12)
        .std()
    )

    # Calendar features
    data["week_of_year"] = (
        data["date"].dt.isocalendar().week
        .astype(int)
    )

    data["month"] = data["date"].dt.month

    data["year"] = data["date"].dt.year

    return data


FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "lag_52",
    "rolling_mean_4",
    "rolling_mean_12",
    "rolling_std_12",
    "week_of_year",
    "month",
    "year"
]

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
        df[
            df["Reporting Area"] == location
        ][
            ["date", "Current week"]
        ]
        .dropna()
        .sort_values("date")
        .drop_duplicates("date")
    )

    if len(temp) < 120:

        print(
            f"SKIPPED: Only {len(temp)} observations."
        )

        continue

    # --------------------------------------------------
    # Create features
    # --------------------------------------------------

    temp = create_features(temp)

    # Remove rows where lag/rolling features
    # cannot be calculated.
    model_data = temp.dropna(
        subset=FEATURES + ["Current week"]
    ).copy()

    if len(model_data) < 80:

        print(
            "SKIPPED: Not enough usable observations "
            "after feature creation."
        )

        continue

    # --------------------------------------------------
    # Chronological split
    # --------------------------------------------------

    split_index = int(
        len(model_data) * (1 - TEST_RATIO)
    )

    train = model_data.iloc[:split_index]
    test = model_data.iloc[split_index:]

    X_train = train[FEATURES]
    y_train = train["Current week"]

    X_test = test[FEATURES]
    y_test = test["Current week"]

    print("Training observations:", len(train))
    print("Testing observations :", len(test))

    # --------------------------------------------------
    # Train XGBoost
    # --------------------------------------------------

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # Dengue cases cannot be negative
    predictions = np.maximum(
        predictions,
        0
    )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    actual_array = y_test.to_numpy()

    non_zero = actual_array != 0

    if non_zero.any():

        mape = np.mean(
            np.abs(
                (
                    actual_array[non_zero]
                    - predictions[non_zero]
                )
                / actual_array[non_zero]
            )
        ) * 100

    else:

        mape = np.nan

    print("\nXGBoost performance:")
    print("MAE :", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("MAPE:", round(mape, 2))

    # --------------------------------------------------
    # Store predictions
    # --------------------------------------------------

    for date, actual, prediction in zip(
        test["date"],
        y_test,
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
# Save
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print("XGBOOST COMPLETED")
print("=" * 60)

print("\nPredictions saved to:")
print(OUTPUT_PATH)

print(
    "\nTotal predictions:",
    len(results_df)
)