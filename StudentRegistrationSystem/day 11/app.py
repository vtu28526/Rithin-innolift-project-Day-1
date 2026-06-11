from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html", active_page="home")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        # No database integration today; we just redirect to the students page
        # after successful submission.
        return redirect(url_for("students"))
    return render_template("register.html", active_page="register")


@app.route("/students")
def students():
    dummy_students = [
        {"roll_no": "101", "name": "Arjun", "department": "IT", "year": "3"},
        {"roll_no": "102", "name": "Priya", "department": "CSE", "year": "2"},
        {"roll_no": "103", "name": "Karthik", "department": "ECE", "year": "1"},
        {"roll_no": "104", "name": "Meera", "department": "CSE", "year": "3"},
        {"roll_no": "105", "name": "Vikram", "department": "IT", "year": "2"},
        {"roll_no": "106", "name": "Ananya", "department": "ECE", "year": "4"},
        {"roll_no": "107", "name": "Rahul", "department": "MECH", "year": "3"},
        {"roll_no": "108", "name": "Swathi", "department": "CIVIL", "year": "2"},
        {"roll_no": "109", "name": "Sanjay", "department": "IT", "year": "1"},
        {"roll_no": "110", "name": "Divya", "department": "MECH", "year": "4"},
    ]
    return render_template("students.html", students=dummy_students, active_page="students")


@app.route("/about")
def about():
    return render_template("about.html", active_page="about")


if __name__ == "__main__":
    app.run(debug=True)

