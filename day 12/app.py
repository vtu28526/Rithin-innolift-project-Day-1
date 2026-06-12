import sqlite3
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for

app = Flask(__name__)
app.config["SECRET_KEY"] = "student-registration-secret"

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "students.db"


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT NOT NULL,
                roll_number TEXT NOT NULL UNIQUE,
                department TEXT NOT NULL,
                year TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT NOT NULL,
                gender TEXT NOT NULL,
                address TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


@app.route("/")
def home():
    return render_template("index.html", active_page="home")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        student = {
            "student_name": request.form.get("student_name", "").strip(),
            "roll_number": request.form.get("roll_number", "").strip(),
            "department": request.form.get("department", "").strip(),
            "year": request.form.get("year", "").strip(),
            "email": request.form.get("email", "").strip(),
            "phone": request.form.get("phone", "").strip(),
            "gender": request.form.get("gender", "").strip(),
            "address": request.form.get("address", "").strip(),
        }

        if not all(student.values()):
            flash("Please fill in all student details before submitting.", "error")
            return render_template("register.html", active_page="register", student=student)

        try:
            with get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO students (
                        student_name, roll_number, department, year,
                        email, phone, gender, address
                    )
                    VALUES (
                        :student_name, :roll_number, :department, :year,
                        :email, :phone, :gender, :address
                    )
                    """,
                    student,
                )
                conn.commit()
        except sqlite3.IntegrityError:
            flash("A student with this roll number is already registered.", "error")
            return render_template("register.html", active_page="register", student=student)

        flash("Student registered successfully.", "success")
        return redirect(url_for("students"))
    return render_template("register.html", active_page="register", student={})


@app.route("/students")
def students():
    with get_db_connection() as conn:
        student_records = conn.execute(
            "SELECT * FROM students ORDER BY created_at DESC, id DESC"
        ).fetchall()
    return render_template("students.html", students=student_records, active_page="students")


@app.route("/about")
def about():
    return render_template("about.html", active_page="about")


init_db()


if __name__ == "__main__":
    app.run(debug=True)
