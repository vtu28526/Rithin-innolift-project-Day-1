import csv
import os
import sqlite3
from statistics import mean

from flask import Flask, flash, render_template, request


app = Flask(__name__)
app.secret_key = "dev-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "disease_outbreak_predictions.db")
DATASET_PATH = os.path.join(DATA_DIR, "disease_outbreak_dataset.csv")

RISK_FIELDS = [
    "population_density_per_sqmi",
    "percent_fair_or_poor_health",
    "percent_adults_with_diabetes",
    "percent_uninsured",
    "percent_below_poverty",
    "percent_age_65_and_older",
    "percent_disabled",
    "percent_overcrowding",
    "percent_no_vehicle",
    "percentile_rank_social_vulnerability",
]


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_table():
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS OutbreakPredictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state TEXT,
            county TEXT,
            population_density REAL,
            percent_vaccinated REAL,
            percent_fair_or_poor_health REAL,
            percent_adults_with_diabetes REAL,
            percent_uninsured REAL,
            percent_below_poverty REAL,
            percent_age_65_and_older REAL,
            percent_disabled REAL,
            percent_overcrowding REAL,
            percent_no_vehicle REAL,
            social_vulnerability REAL,
            risk_score REAL,
            risk_level TEXT,
            prediction TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def to_float(value, default=0.0):
    if value in (None, "", "NA"):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_dataset_rows():
    if not os.path.exists(DATASET_PATH):
        return []

    with open(DATASET_PATH, newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


DATASET_ROWS = load_dataset_rows()


def numeric_values(field):
    return [
        to_float(row.get(field))
        for row in DATASET_ROWS
        if row.get(field) not in (None, "", "NA")
    ]


def percentile_rank(field, value):
    values = numeric_values(field)
    if not values:
        return 0.5

    below_or_equal = sum(1 for item in values if item <= value)
    return below_or_equal / len(values)


def average_for(field):
    values = numeric_values(field)
    return round(mean(values), 2) if values else 0


def dataset_summary():
    if not DATASET_ROWS:
        return {
            "total_records": 0,
            "states": 0,
            "avg_density": 0,
            "avg_vaccinated": 0,
            "avg_social_vulnerability": 0,
        }

    return {
        "total_records": len(DATASET_ROWS),
        "states": len({row.get("state") for row in DATASET_ROWS if row.get("state")}),
        "avg_density": average_for("population_density_per_sqmi"),
        "avg_vaccinated": average_for("percent_vaccinated"),
        "avg_social_vulnerability": average_for("percentile_rank_social_vulnerability"),
    }


SUMMARY = dataset_summary()


def get_form_data():
    return {
        "state": request.form.get("state", "").strip(),
        "county": request.form.get("county", "").strip(),
        "population_density": request.form.get("population_density", "").strip(),
        "percent_vaccinated": request.form.get("percent_vaccinated", "").strip(),
        "percent_fair_or_poor_health": request.form.get("percent_fair_or_poor_health", "").strip(),
        "percent_adults_with_diabetes": request.form.get("percent_adults_with_diabetes", "").strip(),
        "percent_uninsured": request.form.get("percent_uninsured", "").strip(),
        "percent_below_poverty": request.form.get("percent_below_poverty", "").strip(),
        "percent_age_65_and_older": request.form.get("percent_age_65_and_older", "").strip(),
        "percent_disabled": request.form.get("percent_disabled", "").strip(),
        "percent_overcrowding": request.form.get("percent_overcrowding", "").strip(),
        "percent_no_vehicle": request.form.get("percent_no_vehicle", "").strip(),
        "social_vulnerability": request.form.get("social_vulnerability", "").strip(),
    }


def predict_outbreak_risk(data):
    values = {
        "population_density_per_sqmi": to_float(data["population_density"]),
        "percent_fair_or_poor_health": to_float(data["percent_fair_or_poor_health"]),
        "percent_adults_with_diabetes": to_float(data["percent_adults_with_diabetes"]),
        "percent_uninsured": to_float(data["percent_uninsured"]),
        "percent_below_poverty": to_float(data["percent_below_poverty"]),
        "percent_age_65_and_older": to_float(data["percent_age_65_and_older"]),
        "percent_disabled": to_float(data["percent_disabled"]),
        "percent_overcrowding": to_float(data["percent_overcrowding"]),
        "percent_no_vehicle": to_float(data["percent_no_vehicle"]),
        "percentile_rank_social_vulnerability": to_float(data["social_vulnerability"]),
    }
    percent_vaccinated = to_float(data["percent_vaccinated"])

    weighted_risk = (
        percentile_rank("population_density_per_sqmi", values["population_density_per_sqmi"]) * 0.16
        + percentile_rank("percent_fair_or_poor_health", values["percent_fair_or_poor_health"]) * 0.10
        + percentile_rank("percent_adults_with_diabetes", values["percent_adults_with_diabetes"]) * 0.09
        + percentile_rank("percent_uninsured", values["percent_uninsured"]) * 0.09
        + percentile_rank("percent_below_poverty", values["percent_below_poverty"]) * 0.10
        + percentile_rank("percent_age_65_and_older", values["percent_age_65_and_older"]) * 0.09
        + percentile_rank("percent_disabled", values["percent_disabled"]) * 0.08
        + percentile_rank("percent_overcrowding", values["percent_overcrowding"]) * 0.09
        + percentile_rank("percent_no_vehicle", values["percent_no_vehicle"]) * 0.07
        + values["percentile_rank_social_vulnerability"] * 0.13
    )

    if percent_vaccinated < 35:
        weighted_risk += 0.12
    elif percent_vaccinated < 50:
        weighted_risk += 0.07
    elif percent_vaccinated >= 70:
        weighted_risk -= 0.08

    risk_score = max(0, min(round(weighted_risk * 100, 2), 100))

    if risk_score >= 70:
        return risk_score, "High Risk", "High Disease Outbreak Risk"
    if risk_score >= 45:
        return risk_score, "Medium Risk", "Moderate Disease Outbreak Risk"
    return risk_score, "Low Risk", "Low Disease Outbreak Risk"


@app.route("/")
def home():
    return render_template("index.html", summary=SUMMARY)


@app.route("/predict", methods=["GET", "POST"])
@app.route("/register", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        data = get_form_data()
        missing = [label.replace("_", " ").title() for label, value in data.items() if not value]
        if missing:
            flash(f"Missing fields: {', '.join(missing)}", "error")
            return render_template("register.html", data=data), 400

        risk_score, risk_level, prediction = predict_outbreak_risk(data)

        conn = get_db_connection()
        conn.execute(
            """
            INSERT INTO OutbreakPredictions
            (state, county, population_density, percent_vaccinated,
             percent_fair_or_poor_health, percent_adults_with_diabetes,
             percent_uninsured, percent_below_poverty, percent_age_65_and_older,
             percent_disabled, percent_overcrowding, percent_no_vehicle,
             social_vulnerability, risk_score, risk_level, prediction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["state"],
                data["county"],
                to_float(data["population_density"]),
                to_float(data["percent_vaccinated"]),
                to_float(data["percent_fair_or_poor_health"]),
                to_float(data["percent_adults_with_diabetes"]),
                to_float(data["percent_uninsured"]),
                to_float(data["percent_below_poverty"]),
                to_float(data["percent_age_65_and_older"]),
                to_float(data["percent_disabled"]),
                to_float(data["percent_overcrowding"]),
                to_float(data["percent_no_vehicle"]),
                to_float(data["social_vulnerability"]),
                risk_score,
                risk_level,
                prediction,
            ),
        )
        conn.commit()
        conn.close()

        flash(f"Prediction saved: {prediction} ({risk_score}% risk)", "success")
        return render_template(
            "register.html",
            data=data,
            risk_score=risk_score,
            risk_level=risk_level,
            prediction=prediction,
        )

    return render_template("register.html", data={})


@app.route("/predictions")
@app.route("/students")
def predictions():
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT state, county, population_density, percent_vaccinated,
               social_vulnerability, risk_score, risk_level, prediction
        FROM OutbreakPredictions
        ORDER BY id DESC
        """
    ).fetchall()
    conn.close()

    return render_template("students.html", predictions=rows)


@app.route("/about")
def about():
    return render_template("about.html", summary=SUMMARY)


create_table()


if __name__ == "__main__":
    create_table()
    app.run(debug=True)

