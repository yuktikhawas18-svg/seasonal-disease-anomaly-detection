import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.seasonal import STL

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
print("STL SEASONAL DECOMPOSITION")
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
    )

    # Remove duplicate dates if any
    temp = temp.drop_duplicates(
        subset="date"
    )

    # Create weekly time index
    temp = temp.set_index("date")

    temp = temp.asfreq("7D")

    # Interpolate ONLY for decomposition.
    # Original data remains unchanged.
    series = temp["Current week"].interpolate(
        method="linear",
        limit_direction="both"
    )

    print("Usable observations:", len(series))

    # STL requires enough observations
    if len(series) < 104:
        print(
            "SKIPPED: Less than 104 weekly observations."
        )
        continue

    # --------------------------------------------------
    # STL with annual weekly seasonality
    # --------------------------------------------------

    stl = STL(
        series,
        period=52,
        robust=True
    )

    result = stl.fit()

    # --------------------------------------------------
    # Save decomposition values
    # --------------------------------------------------

    decomposition = pd.DataFrame({
        "date": series.index,
        "observed": series.values,
        "trend": result.trend.values,
        "seasonal": result.seasonal.values,
        "residual": result.resid.values
    })

    safe_name = (
        str(location)
        .replace(" ", "_")
        .replace("/", "_")
    )

    csv_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / f"{safe_name}_stl.csv"
    )

    decomposition.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    fig = result.plot()

    fig.set_size_inches(
        12,
        8
    )

    fig.suptitle(
        f"STL Decomposition — {location}",
        fontsize=14
    )

    fig.tight_layout()

    figure_path = (
        FIGURE_FOLDER
        / f"{safe_name}_stl.png"
    )

    fig.savefig(
        figure_path,
        dpi=150
    )

    plt.close(fig)

    print(
        "STL decomposition saved:"
    )
    print(csv_path)

    print(
        "Plot saved:"
    )
    print(figure_path)

print("\n" + "=" * 60)
print("STL ANALYSIS COMPLETED")
print("=" * 60)