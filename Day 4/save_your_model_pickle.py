from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "disease_outbreak_sample.csv"
OUTPUT_DIR = BASE_DIR / "outbreak_outputs"
MODEL_PATH = OUTPUT_DIR / "outbreak_risk_model_pickle.pkl"


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

    features = [
        "vaccination_rate",
        "population_density",
        "sanitation_score",
        "risk_index",
    ]

    x = df[features]
    y = df["risk_class"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print("SAVE YOUR MODEL")
    print(classification_report(y_test, y_pred, zero_division=0))

    model_bundle = {
        "model": model,
        "features": features,
        "classes": sorted(y.unique()),
    }

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model_bundle, file)

    print(f"Model saved to: {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as file:
        loaded_bundle = pickle.load(file)

    sample_input = pd.DataFrame(
        [
            {
                "vaccination_rate": 55,
                "population_density": 9000,
                "sanitation_score": 45,
                "risk_index": 0.536,
            }
        ]
    )

    loaded_model = loaded_bundle["model"]
    prediction = loaded_model.predict(sample_input[loaded_bundle["features"]])[0]
    probabilities = loaded_model.predict_proba(sample_input[loaded_bundle["features"]])[0]

    print()
    print("Loaded model test prediction")
    print(f"Predicted risk class: {prediction}")
    print(f"Prediction probability: {max(probabilities):.3f}")


if __name__ == "__main__":
    main()
