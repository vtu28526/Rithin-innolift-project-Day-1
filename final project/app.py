import csv
import json
import os
import secrets
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from functools import wraps

from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "data_set_1.csv")
DB_FILE = os.path.join(BASE_DIR, "predictions.db")
ENV_FILE = os.path.join(BASE_DIR, ".env")


def load_env_file(path=ENV_FILE):
    if not os.path.exists(path):
        return

    with open(path, encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ[key] = value


load_env_file()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "disease-risk-predictor-secret")

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


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
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                google_id TEXT,
                avatar_url TEXT,
                auth_provider TEXT NOT NULL DEFAULT 'password',
                created_at TEXT NOT NULL
            )
            """
        )
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
        columns = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "google_id" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN google_id TEXT")
        if "avatar_url" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN avatar_url TEXT")
        if "auth_provider" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN auth_provider TEXT NOT NULL DEFAULT 'password'")


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


def format_number(value):
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return "0"


def get_dashboard_context(latest):
    scored_rows = []
    for row in DATASET:
        score, level, _ = calculate_risk(row, "General infectious")
        scored_rows.append({"row": row, "score": score, "level": level})

    scored_rows.sort(key=lambda item: item["score"], reverse=True)
    hotspots = []
    for item in scored_rows[:9]:
        row = item["row"]
        lat = to_float(row.get("lat"), 38)
        lon = to_float(row.get("lon"), -97)
        x = clamp(((lon + 125) / 58) * 100, 4, 96)
        y = clamp(((50 - lat) / 25) * 100, 8, 92)
        hotspots.append(
            {
                "county": row.get("county", "Unknown"),
                "state": row.get("state", "Unknown"),
                "score": item["score"],
                "level": item["level"],
                "x": round(x, 1),
                "y": round(y, 1),
            }
        )

    average_score = round(
        sum(item["score"] for item in scored_rows) / len(scored_rows)
    ) if scored_rows else 0
    critical_count = sum(1 for item in scored_rows if item["level"] == "Critical")
    high_count = sum(1 for item in scored_rows if item["level"] == "High")
    total_population = sum(to_float(row.get("total_population"), 0) or 0 for row in DATASET)

    disease_counts = {}
    for item in latest:
        disease = item["disease_type"]
        disease_counts[disease] = disease_counts.get(disease, 0) + 1

    disease_snapshot = [
        {"name": "Respiratory", "count": disease_counts.get("Respiratory", 0), "tone": "amber"},
        {"name": "Waterborne", "count": disease_counts.get("Waterborne", 0), "tone": "blue"},
        {"name": "Vector-borne", "count": disease_counts.get("Vector-borne", 0), "tone": "green"},
        {"name": "General infectious", "count": disease_counts.get("General infectious", 0), "tone": "red"},
    ]
    global_points = [
        {"label": "North America", "x": 18, "y": 39, "level": "High", "score": hotspots[0]["score"] if hotspots else 68},
        {"label": "South America", "x": 31, "y": 70, "level": "High", "score": hotspots[1]["score"] if len(hotspots) > 1 else 63},
        {"label": "Western Europe", "x": 48, "y": 35, "level": "Moderate", "score": hotspots[2]["score"] if len(hotspots) > 2 else 51},
        {"label": "West Africa", "x": 49, "y": 56, "level": "High", "score": hotspots[3]["score"] if len(hotspots) > 3 else 66},
        {"label": "South Asia", "x": 69, "y": 49, "level": "Critical", "score": hotspots[4]["score"] if len(hotspots) > 4 else 78},
        {"label": "East Asia", "x": 78, "y": 42, "level": "High", "score": hotspots[5]["score"] if len(hotspots) > 5 else 64},
        {"label": "Australia", "x": 82, "y": 76, "level": "Moderate", "score": hotspots[6]["score"] if len(hotspots) > 6 else 54},
    ]

    return {
        "average_score": average_score,
        "critical_count": critical_count,
        "high_count": high_count,
        "total_population": format_number(total_population),
        "hotspots": hotspots,
        "global_points": global_points,
        "disease_snapshot": disease_snapshot,
    }


def average(values):
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 1) if values else 0


def get_analysis_report():
    scored_rows = []
    state_groups = {}
    level_counts = {"Low": 0, "Moderate": 0, "High": 0, "Critical": 0}
    disease_summary = []

    for row in DATASET:
        score, level, details = calculate_risk(row, "General infectious")
        level_counts[level] += 1
        scored_rows.append({"row": row, "score": score, "level": level, "details": details})
        state_groups.setdefault(row.get("state", "Unknown"), []).append(score)

    for disease in DISEASE_ADJUSTMENTS:
        scores = [calculate_risk(row, disease)[0] for row in DATASET]
        disease_summary.append(
            {
                "name": disease,
                "average": average(scores),
                "peak": max(scores) if scores else 0,
                "level": risk_level(average(scores)),
            }
        )

    top_states = sorted(
        (
            {
                "state": state,
                "average": average(scores),
                "counties": len(scores),
                "level": risk_level(average(scores)),
            }
            for state, scores in state_groups.items()
        ),
        key=lambda item: item["average"],
        reverse=True,
    )[:8]

    factor_rows = []
    for factor in RISK_FACTORS:
        scores = []
        values = []
        for row in DATASET:
            value = to_float(row.get(factor["field"]))
            values.append(value)
            scores.append(normalize(value, factor["scale"], factor["direction"]))
        factor_rows.append(
            {
                "label": factor["label"],
                "average_score": average(scores),
                "average_value": average(values),
                "weight": factor["weight"],
            }
        )

    factor_rows.sort(key=lambda item: item["average_score"] * item["weight"], reverse=True)
    scored_rows.sort(key=lambda item: item["score"], reverse=True)
    history = get_history()

    return {
        "average_score": average([item["score"] for item in scored_rows]),
        "total_counties": len(DATASET),
        "states": len(get_states()),
        "level_counts": level_counts,
        "top_counties": scored_rows[:10],
        "top_states": top_states,
        "factor_rows": factor_rows[:8],
        "disease_summary": disease_summary,
        "prediction_count": len(history),
        "latest_prediction": history[0] if history else None,
    }


def get_user_by_email(email):
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT * FROM users WHERE lower(email) = lower(?)",
            (email,),
        ).fetchone()


def get_user_by_id(user_id):
    if not user_id:
        return None

    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def get_user_by_google_id(google_id):
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM users WHERE google_id = ?", (google_id,)).fetchone()


def create_user(name, email, password):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            """
            INSERT INTO users (name, email, password_hash, auth_provider, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                name,
                email,
                generate_password_hash(password),
                "password",
                datetime.now().strftime("%Y-%m-%d %H:%M"),
            ),
        )
        return cursor.lastrowid


def upsert_google_user(profile):
    google_id = profile.get("sub")
    email = profile.get("email", "").strip().lower()
    name = profile.get("name") or email.split("@")[0]
    avatar_url = profile.get("picture")

    user = get_user_by_google_id(google_id) or get_user_by_email(email)
    with sqlite3.connect(DB_FILE) as conn:
        if user:
            conn.execute(
                """
                UPDATE users
                SET name = ?, google_id = ?, avatar_url = ?, auth_provider = ?
                WHERE id = ?
                """,
                (name, google_id, avatar_url, "google", user["id"]),
            )
            return user["id"]

        cursor = conn.execute(
            """
            INSERT INTO users
            (name, email, password_hash, google_id, avatar_url, auth_provider, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                email,
                generate_password_hash(secrets.token_urlsafe(32)),
                google_id,
                avatar_url,
                "google",
                datetime.now().strftime("%Y-%m-%d %H:%M"),
            ),
        )
        return cursor.lastrowid


def google_oauth_configured():
    client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
    placeholder_values = {
        "your-google-oauth-client-id.apps.googleusercontent.com",
        "your-google-oauth-client-secret",
    }
    return bool(client_id and client_secret) and client_id not in placeholder_values and client_secret not in placeholder_values


def get_google_redirect_uri():
    return os.environ.get("GOOGLE_REDIRECT_URI") or url_for("google_callback", _external=True)


def is_safe_redirect_target(target):
    if not target:
        return False

    host_url = urllib.parse.urlparse(request.host_url)
    redirect_url = urllib.parse.urlparse(urllib.parse.urljoin(request.host_url, target))
    return redirect_url.scheme in ("http", "https") and host_url.netloc == redirect_url.netloc


def fetch_google_profile(code, redirect_uri):
    token_payload = urllib.parse.urlencode(
        {
            "code": code,
            "client_id": os.environ["GOOGLE_CLIENT_ID"],
            "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
    ).encode("utf-8")
    token_request = urllib.request.Request(
        GOOGLE_TOKEN_URL,
        data=token_payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(token_request, timeout=10) as response:
        token_data = json.loads(response.read().decode("utf-8"))

    userinfo_request = urllib.request.Request(
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {token_data['access_token']}"},
    )
    with urllib.request.urlopen(userinfo_request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


@app.context_processor
def inject_current_user():
    return {
        "current_user": get_user_by_id(session.get("user_id")),
        "google_oauth_enabled": google_oauth_configured(),
    }


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("signup", next=request.full_path))
        return view(*args, **kwargs)

    return wrapped_view


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if session.get("user_id"):
        return redirect(url_for("home"))

    form = {"name": "", "email": ""}

    if request.method == "POST":
        form["name"] = request.form.get("name", "").strip()
        form["email"] = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not form["name"] or not form["email"] or not password:
            flash("Please complete all required fields.", "error")
        elif "@" not in form["email"]:
            flash("Please enter a valid email address.", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
        elif password != confirm_password:
            flash("Passwords do not match.", "error")
        elif get_user_by_email(form["email"]):
            flash("An account with that email already exists.", "error")
        else:
            session["user_id"] = create_user(form["name"], form["email"], password)
            flash("Account created successfully. You are now signed in.", "success")
            next_page = request.args.get("next")
            return redirect(next_page if is_safe_redirect_target(next_page) else url_for("home"))

    return render_template("signup.html", active_page="signup", form=form)


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if session.get("user_id"):
        return redirect(url_for("home"))

    email = ""

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = get_user_by_email(email)

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            flash("Welcome back.", "success")
            next_page = request.args.get("next")
            return redirect(next_page if is_safe_redirect_target(next_page) else url_for("home"))

        flash("Invalid email or password.", "error")

    return render_template("signin.html", active_page="signin", email=email)


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("You have been signed out.", "success")
    return redirect(url_for("home"))


@app.route("/auth/google")
def google_auth():
    if not google_oauth_configured():
        flash("Google authentication is not configured yet.", "error")
        return redirect(request.referrer or url_for("signin"))

    state = secrets.token_urlsafe(24)
    session["google_oauth_state"] = state
    next_page = request.args.get("next") or request.referrer
    if not is_safe_redirect_target(next_page):
        next_page = url_for("home")
    session["google_oauth_next"] = next_page
    redirect_uri = get_google_redirect_uri()
    query = urllib.parse.urlencode(
        {
            "client_id": os.environ["GOOGLE_CLIENT_ID"],
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "prompt": "select_account",
        }
    )
    return redirect(f"{GOOGLE_AUTH_URL}?{query}")


@app.route("/auth/google/callback")
def google_callback():
    expected_state = session.pop("google_oauth_state", None)
    received_state = request.args.get("state")
    if not expected_state or received_state != expected_state:
        flash("Google authentication could not be verified.", "error")
        return redirect(url_for("signin"))

    if request.args.get("error"):
        flash("Google authentication was cancelled.", "error")
        return redirect(url_for("signin"))

    code = request.args.get("code")
    if not code:
        flash("Google authentication did not return an authorization code.", "error")
        return redirect(url_for("signin"))

    try:
        profile = fetch_google_profile(code, get_google_redirect_uri())
    except (KeyError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        flash("Google authentication failed. Please try again.", "error")
        return redirect(url_for("signin"))

    if not profile.get("email_verified") or not profile.get("email"):
        flash("Google account email could not be verified.", "error")
        return redirect(url_for("signin"))

    session["user_id"] = upsert_google_user(profile)
    flash("Signed in with Google successfully.", "success")
    next_page = session.pop("google_oauth_next", None)
    return redirect(next_page if is_safe_redirect_target(next_page) else url_for("home"))


@app.route("/")
@login_required
def home():
    states = get_states()
    total_counties = len(DATASET)
    latest = get_history()[:3]
    dashboard = get_dashboard_context(latest)
    return render_template(
        "home.html",
        active_page="home",
        states=states,
        total_counties=total_counties,
        latest=latest,
        dashboard=dashboard,
    )


@app.route("/predict", methods=["GET", "POST"])
@login_required
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
@login_required
def history():
    return render_template(
        "history.html",
        active_page="history",
        predictions=get_history(),
    )


@app.route("/dataset")
@login_required
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


@app.route("/analysis")
@login_required
def analysis():
    return render_template(
        "analysis.html",
        active_page="analysis",
        report=get_analysis_report(),
    )


@app.route("/about")
@login_required
def about():
    return render_template("about.html", active_page="about")


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5001)
else:
    init_db()
