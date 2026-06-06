"""
Training script for Disease Outbreak Risk Predictor.

Run:
python train.py
"""

from model import clean_data, load_data, save_model, train_model


def main():
    df = load_data()
    df = clean_data(df)
    model = train_model(df)
    save_model(model)


if __name__ == "__main__":
    main()
