from app import app, get_db_connection, init_db


FIELDS = [
    "student_name",
    "roll_number",
    "department",
    "year",
    "email",
    "phone",
    "gender",
    "address",
]

SAMPLE_STUDENTS = [
    ("Aarav Kumar", "DAY12-201", "CSE", "1", "aarav@example.com", "9876500001", "Male", "Chennai, Tamil Nadu"),
    ("Diya Sharma", "DAY12-202", "IT", "2", "diya@example.com", "9876500002", "Female", "Bengaluru, Karnataka"),
    ("Rohan Mehta", "DAY12-203", "ECE", "3", "rohan@example.com", "9876500003", "Male", "Hyderabad, Telangana"),
    ("Sneha Reddy", "DAY12-204", "MECH", "4", "sneha@example.com", "9876500004", "Female", "Coimbatore, Tamil Nadu"),
    ("Kavin Raj", "DAY12-205", "CIVIL", "1", "kavin@example.com", "9876500005", "Male", "Madurai, Tamil Nadu"),
    ("Nisha Patel", "DAY12-206", "CSE", "2", "nisha@example.com", "9876500006", "Female", "Pune, Maharashtra"),
    ("Aditya Singh", "DAY12-207", "IT", "3", "aditya@example.com", "9876500007", "Male", "Delhi"),
    ("Meera Iyer", "DAY12-208", "ECE", "4", "meera@example.com", "9876500008", "Female", "Kochi, Kerala"),
    ("Vikram Das", "DAY12-209", "MECH", "1", "vikram@example.com", "9876500009", "Male", "Kolkata, West Bengal"),
    ("Ananya Bose", "DAY12-210", "CIVIL", "2", "ananya@example.com", "9876500010", "Female", "Mysuru, Karnataka"),
]


def main():
    init_db()

    with app.test_client() as client:
        for sample in SAMPLE_STUDENTS:
            data = dict(zip(FIELDS, sample))
            response = client.post("/register", data=data, follow_redirects=True)
            if response.status_code != 200:
                raise RuntimeError(f"Registration failed for {data['roll_number']}")

        for route in ["/", "/register", "/students", "/about"]:
            response = client.get(route)
            if response.status_code != 200:
                raise RuntimeError(f"Route failed: {route}")

    with get_db_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        seeded = conn.execute(
            "SELECT COUNT(*) FROM students WHERE roll_number LIKE 'DAY12-%'"
        ).fetchone()[0]
        rows = conn.execute(
            """
            SELECT roll_number, student_name, department, year
            FROM students
            WHERE roll_number LIKE 'DAY12-%'
            ORDER BY roll_number
            """
        ).fetchall()

    print(f"total_records={total}")
    print(f"day12_seeded_records={seeded}")
    for row in rows:
        print(
            f"{row['roll_number']} - {row['student_name']} - "
            f"{row['department']} - Year {row['year']}"
        )


if __name__ == "__main__":
    main()
