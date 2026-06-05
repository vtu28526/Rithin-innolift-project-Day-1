from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "disease_outbreak_sample.csv"


def add_risk_index(df):
    df = df.copy()
    vaccination_risk = (100 - df["vaccination_rate"]) / 100
    density_risk = df["population_density"] / df["population_density"].max()
    sanitation_risk = (100 - df["sanitation_score"]) / 100

    df["risk_index"] = (
        0.40 * vaccination_risk + 0.35 * density_risk + 0.25 * sanitation_risk
    ).round(3)
    return df


def main():
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
    predictions = model.predict(x_test)

    print("YOUR FIRST REGRESSION")
    print("Numeric columns used:")
    print(list(x.columns))
    print()
    print(f"Train rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.2f}")
    print(f"Mean Squared Error: {mean_squared_error(y_test, predictions):.2f}")
    print(f"R2 Score: {r2_score(y_test, predictions):.3f}")

    result = pd.DataFrame(
        {
            "actual_cases": y_test.values,
            "predicted_cases": predictions.round(2),
        }
    )
    print()
    print("Actual vs Predicted:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
