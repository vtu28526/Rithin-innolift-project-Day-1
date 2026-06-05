from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "disease_outbreak_sample.csv"
OUTPUT_DIR = BASE_DIR / "outbreak_outputs"
PLOT_PATH = OUTPUT_DIR / "predict_vs_actual_plot.png"


def add_risk_index(df):
    df = df.copy()

    vaccination_risk = (100 - df["vaccination_rate"]) / 100
    density_risk = df["population_density"] / df["population_density"].max()
    sanitation_risk = (100 - df["sanitation_score"]) / 100

    df["risk_index"] = (
        0.40 * vaccination_risk
        + 0.35 * density_risk
        + 0.25 * sanitation_risk
    ).round(3)

    return df


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = add_risk_index(df)

    numeric_df = df.select_dtypes(include="number")
    numeric_df = numeric_df.fillna(numeric_df.median())

    x = numeric_df.drop(columns=["cases_last_week"])
    y = numeric_df["cases_last_week"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
    )

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print("PREDICT VS ACTUAL PLOT")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.3f}")

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Cases")
    plt.ylabel("Predicted Cases")
    plt.title("Predicted vs Actual Outbreak Cases")
    plt.grid(True)

    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value], color="red")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.show()

    print(f"Plot saved to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
