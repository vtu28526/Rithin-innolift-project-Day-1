import csv
import os
import sqlite3
from datetime import datetime

from flask import Flask, flash, render_template, request


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "data_set_1.csv")
DB_FILE = os.path.join(BASE_DIR, "predictions.db")

app = Flask(__name__)
app.secret_key = "disease-risk-predictor-secret"


RISK_FACTORS = [
    {
        "label": "Social vulnerability",
        "field": "percentile_rank_social_vulnerability",
        "weight": 18,
        "scale": "percentile",
        "direction": "high",
    },
    {
        "label": "Vaccination gap",
        "field": "percent_vaccinated",
        "weight": 16,
        "scale": "percent",
        "direction": "low",
    },
    {
        "label": "Population density",
        "field": "population_density_per_sqmi",
        "weight": 12,
        "scale": "density",
        "direction": "high",
    },
    {
        "label": "Uninsured population",
        "field": "percent_uninsured",
        "weight": 10,
        "scale": "percent",
        "direction": "high",
    },
    {
        "label": "Poor or fair health",
        "field": "percent_fair_or_poor_health",
        "weight": 10,
        "scale": "percent",
        "direction": "high",
    },
    {
        "label": "Diabetes prevalence",
        "field": "percent_adults_with_diabetes",
        "weight": 8,
        "scale": "percent",
        "direction": "high",
    },
    {
        "label": "Limited healthy food access",
        "field": "percent_limited_access_to_healthy_foods",
        "weight": 6,
        "scale": "percent",
        "direction": "high",
    },
    {
        "label": "Severe housing problems",
        "field": "percent_severe_housing_problems",
        "weight": 6,
        "scale": "percent",
        "direction": "high",
    },
    {
        "label": "Poverty",
        "field": "percent_below_poverty",
        "weight": 6,
        "scale": "percent",
        "direction": "high",
    },
    {
        "label": "Older population",
        "field": "percent_65_and_over",
        "weight": 4,
        "scale": "percent",
        "direction": "high",
    },
    {
        "label": "PM2.5 exposure",
        "field": "average_daily_pm2_5",
        "weight": 4,
        "scale": "pm25",
        "direction": "high",
    },
]

DISEASE_ADJUSTMENTS = {
    "Respiratory": {"average_daily_pm2_5": 1.25, "population_density_per_sqmi": 1.15},
    "Waterborne": {"presence_of_water_violation": 1.3, "percent_limited_access_to_healthy_foods": 1.1},
    "Vector-borne": {"percent_severe_housing_problems": 1.15, "percent_rural": 1.1},
    "General infectious": {},
}


def to_float(value, default=None):
    if value in (None, "", "NA"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value, low=0, high=100):
    return max(low, min(high, value))


def normalize(value, scale, direction):
    if value is None:
        return 0
    if scale == "percentile":
        score = value * 100
    elif scale == "density":
        score = min(value / 10, 100)
    elif scale == "pm25":
        score = min(value / 0.2, 100)
    else:
        score = value

    if direction == "low":
        score = 100 - score
    return clamp(score)


def risk_level(score):
    if score >= 75:
        return "Critical"
    if score >= 55:
        return "High"
    if score >= 35:
        return "Moderate"
    return "Low"


def load_dataset():
    if not os.path.exists(DATA_FILE):
        return []

    with open(DATA_FILE, newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))

    for row in rows:
        row["location"] = f"{row.get('county', 'Unknown')}, {row.get('state', 'Unknown')}"
    return rows


DATASET = load_dataset()


def get_states():
    return sorted({row["state"] for row in DATASET if row.get("state")})


def get_counties():
    return sorted(
        DATASET,
        key=lambda row: (row.get("state", ""), row.get("county", "")),
    )


def find_county(fips):
    return next((row for row in DATASET if row.get("fips") == fips), None)


def calculate_risk(row, disease_type):
    details = []
    total_weight = 0
    weighted_score = 0
    adjustments = DISEASE_ADJUSTMENTS.get(disease_type, {})

    for factor in RISK_FACTORS:
        value = to_float(row.get(factor["field"]))
        normalized = normalize(value, factor["scale"], factor["direction"])
        multiplier = adjustments.get(factor["field"], 1)
        weight = factor["weight"] * multiplier
        total_weight += weight
        weighted_score += normalized * weight
        details.append(
            {
                "label": factor["label"],
                "value": value,
                "score": round(normalized),
                "impact": round(normalized * weight / 100, 1),
            }
        )

    if row.get("presence_of_water_violation") == "TRUE" and disease_type == "Waterborne":
        weighted_score += 8 * total_weight

    score = round(weighted_score / total_weight if total_weight else 0)
    score = clamp(score)
    return score, risk_level(score), sorted(details, key=lambda item: item["impact"], reverse=True)


def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fips TEXT NOT NULL,
                state TEXT NOT NULL,
                county TEXT NOT NULL,
                disease_type TEXT NOT NULL,
                risk_score INTEGER NOT NULL,
                risk_level TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL
            )
            """
        )


def save_prediction(row, disease_type, score, level, notes):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """
            INSERT INTO predictions
            (fips, state, county, disease_type, risk_score, risk_level, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("fips"),
                row.get("state"),
                row.get("county"),
                disease_type,
                score,
                level,
                notes,
                datetime.now().strftime("%Y-%m-%d %H:%M"),
            ),
        )


def get_history():
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT 50"
        ).fetchall()


@app.route("/")
def home():
    states = get_states()
    total_counties = len(DATASET)
    latest = get_history()[:3]
    return render_template(
        "home.html",
        active_page="home",
        states=states,
        total_counties=total_counties,
        latest=latest,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    selected_fips = request.form.get("fips", "")
    disease_type = request.form.get("disease_type", "General infectious")
    notes = request.form.get("notes", "").strip()

    if request.method == "POST":
        row = find_county(selected_fips)
        if not row:
            flash("Please choose a county before predicting.", "error")
        elif disease_type not in DISEASE_ADJUSTMENTS:
            flash("Please choose a valid disease category.", "error")
        else:
            score, level, details = calculate_risk(row, disease_type)
            save_prediction(row, disease_type, score, level, notes)
            flash("Risk prediction completed and saved.", "success")
            result = {
                "row": row,
                "score": score,
                "level": level,
                "details": details,
                "disease_type": disease_type,
            }

    return render_template(
        "predict.html",
        active_page="predict",
        counties=get_counties(),
        disease_types=DISEASE_ADJUSTMENTS.keys(),
        selected_fips=selected_fips,
        selected_disease=disease_type,
        notes=notes,
        result=result,
    )


@app.route("/history")
def history():
    return render_template(
        "history.html",
        active_page="history",
        predictions=get_history(),
    )


@app.route("/dataset")
def dataset():
    state = request.args.get("state", "")
    rows = [row for row in get_counties() if not state or row.get("state") == state]
    return render_template(
        "dataset.html",
        active_page="dataset",
        states=get_states(),
        selected_state=state,
        rows=rows[:150],
        total=len(rows),
    )


@app.route("/about")
def about():
    return render_template("about.html", active_page="about")


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5001)
else:
    init_db()
