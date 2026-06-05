from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "disease_outbreak_sample.csv"
OUTPUT_DIR = BASE_DIR / "outbreak_outputs"
OUTPUT_PATH = OUTPUT_DIR / "feature_comparison_rmse.csv"


def add_risk_index(df):
    df = df.copy()
    vaccination_risk = (100 - df["vaccination_rate"]) / 100
    density_risk = df["population_density"] / df["population_density"].max()
    sanitation_risk = (100 - df["sanitation_score"]) / 100

    df["risk_index"] = (
        0.40 * vaccination_risk + 0.35 * density_risk + 0.25 * sanitation_risk
    ).round(3)
    return df


def compare_features_by_rmse(df):
    numeric_df = df.select_dtypes(include="number")
    numeric_df = numeric_df.fillna(numeric_df.median())

    target = "cases_last_week"
    features = [column for column in numeric_df.columns if column != target]
    results = []

    for feature in features:
        x = numeric_df[[feature]]
        y = numeric_df[target]

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
        rmse = mean_squared_error(y_test, predictions) ** 0.5

        results.append(
            {
                "feature": feature,
                "rmse": round(rmse, 2),
            }
        )

    return pd.DataFrame(results).sort_values("rmse")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = add_risk_index(df)

    comparison = compare_features_by_rmse(df)

    print("FEATURE COMPARISON - RMSE")
    print(comparison.to_string(index=False))

    best_feature = comparison.iloc[0]
    print()
    print(
        f"Best single feature: {best_feature['feature']} "
        f"with RMSE {best_feature['rmse']}"
    )

    comparison.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
