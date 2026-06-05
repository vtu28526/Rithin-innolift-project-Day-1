from pathlib import Path

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
OUTPUT_PATH = OUTPUT_DIR / "train_test_split_explorer.csv"


def add_risk_index(df):
    df = df.copy()
    vaccination_risk = (100 - df["vaccination_rate"]) / 100
    density_risk = df["population_density"] / df["population_density"].max()
    sanitation_risk = (100 - df["sanitation_score"]) / 100

    df["risk_index"] = (
        0.40 * vaccination_risk + 0.35 * density_risk + 0.25 * sanitation_risk
    ).round(3)
    return df


def load_outbreak_dataset():
    df = pd.read_csv(DATA_PATH)
    df["vaccination_rate"] = df["vaccination_rate"].fillna(
        df["vaccination_rate"].median()
    )
    df["sanitation_score"] = df["sanitation_score"].fillna(
        df["sanitation_score"].median()
    )
    df["cases_last_week"] = df["cases_last_week"].fillna(
        df["cases_last_week"].median()
    )
    return add_risk_index(df)


def train_test_split_explorer(df, test_sizes=None):
    if test_sizes is None:
        test_sizes = [0.2, 0.25, 0.3, 0.35, 0.4]

    features = [
        "vaccination_rate",
        "population_density",
        "sanitation_score",
        "risk_index",
    ]

    x = df[features]
    y = df["cases_last_week"]
    results = []

    for test_size in test_sizes:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
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

        results.append(
            {
                "test_size": test_size,
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "mae": round(mean_absolute_error(y_test, predictions), 2),
                "r2_score": round(r2_score(y_test, predictions), 3),
            }
        )

    return pd.DataFrame(results)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_outbreak_dataset()
    results = train_test_split_explorer(df, test_sizes=[0.2, 0.25, 0.3, 0.35, 0.4])

    print("TRAIN TEST SPLIT EXPLORER")
    print(results.to_string(index=False))

    results.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
