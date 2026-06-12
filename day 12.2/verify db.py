import os
import sqlite3


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "disease_outbreak_predictions.db")
DATASET_PATH = os.path.join(DATA_DIR, "disease_outbreak_dataset.csv")


print("DB_PATH:", DB_PATH)
print("DB exists:", os.path.exists(DB_PATH))
print("Dataset exists:", os.path.exists(DATASET_PATH))

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

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

rows = conn.execute("SELECT COUNT(*) AS c FROM OutbreakPredictions").fetchone()
print("Outbreak prediction rows:", rows["c"] if rows else None)

conn.close()
print("OK")

verify_db.py