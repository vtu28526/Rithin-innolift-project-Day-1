"""
Disease Outbreak Risk Predictor

Full pipeline:
load dataset -> clean missing values -> split -> train model -> evaluate -> save model
"""

from pathlib import Path

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "disease_outbreak_sample.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

FEATURES = ["vaccination_rate", "sanitation_score", "population_density"]
TARGET = "risk_class"
RISK_CLASSES = ["Low", "Medium", "High", "Critical"]


def load_data():
    """Load outbreak dataset from CSV."""
    df = pd.read_csv(DATA_PATH)
    return df


def clean_data(df):
    """Fill missing values and keep required columns."""
    for column in FEATURES:
        df[column] = df[column].fillna(df[column].median())

    df[TARGET] = df[TARGET].fillna(df[TARGET].mode()[0])
    return df


def train_model(df):
    """Train Random Forest classifier."""
    x = df[FEATURES]
    y = df[TARGET]
    test_size = 0.2
    test_count = round(len(df) * test_size)
    train_count = len(df) - test_count
    class_count = y.nunique()
    min_class_count = y.value_counts().min()
    stratify_target = y if test_count >= class_count and train_count >= class_count and min_class_count >= 2 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify_target,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Model Accuracy:", round(accuracy * 100, 2), "%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model


def save_model(model):
    """Save trained model as model.pkl."""
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)
    print(f"\nModel saved successfully as {MODEL_PATH.name}")


def main():
    df = load_data()

    print("Dataset loaded successfully")
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nRisk class counts:")
    print(df[TARGET].value_counts())

    df = clean_data(df)
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    model = train_model(df)
    save_model(model)


if __name__ == "__main__":
    main()
