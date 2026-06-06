"""
Load saved model.pkl and predict outbreak risk for 3 real-world cases.

Run:
python predict.py
"""

from pathlib import Path
import pickle

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

FEATURES = ["vaccination_rate", "sanitation_score", "population_density"]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl not found. Run python train.py first.")

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return model


def predict_cases(model):
    new_cases = pd.DataFrame(
        [
            {
                "vaccination_rate": 85,
                "sanitation_score": 80,
                "population_density": 3500,
            },
            {
                "vaccination_rate": 60,
                "sanitation_score": 55,
                "population_density": 7000,
            },
            {
                "vaccination_rate": 35,
                "sanitation_score": 28,
                "population_density": 12000,
            },
        ]
    )

    predictions = model.predict(new_cases[FEATURES])

    print("Disease Outbreak Risk Predictions")
    print("=" * 45)

    for index, prediction in enumerate(predictions, start=1):
        print(f"\nCase {index}")
        print(new_cases.iloc[index - 1])
        print("Predicted Risk Class:", prediction)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(new_cases[FEATURES])
        class_names = model.classes_

        print("\nPrediction Probabilities")
        print("=" * 45)
        for case_index, row in enumerate(probabilities, start=1):
            print(f"\nCase {case_index}")
            for class_name, probability in zip(class_names, row):
                print(f"{class_name}: {round(probability * 100, 2)}%")


def main():
    model = load_model()
    predict_cases(model)


if __name__ == "__main__":
    main()
