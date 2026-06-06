# Disease Outbreak Risk Predictor

This project predicts disease outbreak risk using healthcare features such as
vaccination rate, sanitation score, and population density.

## Dataset

The dataset file is:

```text
disease_outbreak_sample.csv
```

Main columns:

- `region`
- `vaccination_rate`
- `sanitation_score`
- `population_density`
- `risk_class`
- `cases_last_week`

The target column is `risk_class`.

Risk classes:

- `Low`
- `Medium`
- `High`
- `Critical`

## Algorithm

The project uses `RandomForestClassifier` from scikit-learn.

## How to Run

Install requirements:

```bash
pip install -r requirements.txt
```

Train and save model:

```bash
python train.py
```

Predict 3 new cases:

```bash
python predict.py
```

Run full pipeline:

```bash
python model.py
```

## Features Used

- Vaccination rate
- Sanitation score
- Population density

## Goal

The goal is to classify each area into one of four outbreak risk levels:
`Low`, `Medium`, `High`, or `Critical`.
