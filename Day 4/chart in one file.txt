from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "disease_outbreak_sample.csv"
OUTPUT_DIR = BASE_DIR / "outbreak_outputs" / "charts"
DB_FILE = BASE_DIR / "outbreak_outputs" / "outbreak_charts.sqlite3"

RISK_ORDER = ["Low", "Medium", "High", "Critical"]
RISK_COLORS = {
    "Low": "#2e7d32",
    "Medium": "#f9a825",
    "High": "#ef6c00",
    "Critical": "#c62828",
}


def load_outbreak_data():
    if DATA_FILE.exists():
        data = pd.read_csv(DATA_FILE)
    else:
        data = pd.DataFrame(
            {
                "region": [
                    "North Zone",
                    "South Zone",
                    "East Zone",
                    "West Zone",
                    "Central Zone",
                    "River Ward",
                    "Hill Ward",
                    "Market Ward",
                    "Lake Ward",
                    "Industrial Ward",
                ],
                "vaccination_rate": [82, 64, 51, 38, 74, 45, 88, 41, 69, 34],
                "population_density": [4200, 6900, 8700, 11200, 5300, 9600, 3100, 10400, 6200, 12300],
                "sanitation_score": [78, 58, 43, 31, 69, 37, 82, 29, 62, 25],
                "risk_class": ["Low", "Medium", "High", "Critical", "Medium", "High", "Low", "Critical", "Medium", "Critical"],
                "cases_last_week": [12, 46, 98, 171, 33, 124, 7, 153, 41, 166],
            }
        )

    numeric_columns = [
        "vaccination_rate",
        "population_density",
        "sanitation_score",
        "cases_last_week",
    ]
    for column in numeric_columns:
        data[column] = data[column].fillna(data[column].median())

    data["risk_class"] = pd.Categorical(data["risk_class"], categories=RISK_ORDER, ordered=True)
    data["risk_index"] = (
        (100 - data["vaccination_rate"]) * 0.35
        + (data["population_density"] / data["population_density"].max() * 100) * 0.35
        + (100 - data["sanitation_score"]) * 0.30
    ).round(2)
    return data


def save_sqlite_summary(data):
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary = (
        data.groupby("risk_class", observed=False)
        .agg(
            total_regions=("region", "count"),
            avg_vaccination=("vaccination_rate", "mean"),
            avg_density=("population_density", "mean"),
            avg_sanitation=("sanitation_score", "mean"),
            avg_risk_index=("risk_index", "mean"),
        )
        .round(2)
        .reset_index()
    )

    sqlite_data = data.copy()
    sqlite_summary = summary.copy()
    sqlite_data["risk_class"] = sqlite_data["risk_class"].astype(str)
    sqlite_summary["risk_class"] = sqlite_summary["risk_class"].astype(str)

    with sqlite3.connect(DB_FILE) as connection:
        sqlite_data.to_sql("outbreak_data", connection, if_exists="replace", index=False)
        sqlite_summary.to_sql("risk_summary", connection, if_exists="replace", index=False)


def make_charts(data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    risk_counts = data["risk_class"].value_counts().reindex(RISK_ORDER, fill_value=0)

    plt.figure(figsize=(8, 5))
    plt.bar(risk_counts.index.astype(str), risk_counts.values, color=[RISK_COLORS[risk] for risk in RISK_ORDER])
    plt.title("Disease Outbreak Risk Class Count")
    plt.xlabel("Risk Class")
    plt.ylabel("Number of Regions")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_risk_class_count.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    sorted_data = data.sort_values("vaccination_rate", ascending=False)
    plt.bar(sorted_data["region"], sorted_data["vaccination_rate"], color="#1976d2")
    plt.title("Vaccination Rate by Region")
    plt.xlabel("Region")
    plt.ylabel("Vaccination Rate (%)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_vaccination_rate_by_region.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for risk in RISK_ORDER:
        group = data[data["risk_class"] == risk]
        plt.scatter(
            group["population_density"],
            group["sanitation_score"],
            label=risk,
            color=RISK_COLORS[risk],
            s=90,
            edgecolor="black",
        )
    plt.title("Population Density vs Sanitation Score")
    plt.xlabel("Population Density")
    plt.ylabel("Sanitation Score")
    plt.legend(title="Risk Class")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_density_vs_sanitation.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    sorted_index = data.sort_values("risk_index", ascending=False)
    plt.plot(sorted_index["region"], sorted_index["risk_index"], marker="o", color="#6a1b9a", linewidth=2)
    plt.title("Engineered Risk Index by Region")
    plt.xlabel("Region")
    plt.ylabel("Risk Index")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_risk_index_by_region.png", dpi=150)
    plt.close()


def main():
    outbreak_data = load_outbreak_data()
    save_sqlite_summary(outbreak_data)
    make_charts(outbreak_data)
    print("Charts saved in:", OUTPUT_DIR)
    print("SQLite file saved as:", DB_FILE)


if __name__ == "__main__":
    main()
