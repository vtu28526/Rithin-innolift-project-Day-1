# Multiplication Table
# Usage: python multiplication_table.py


def main():
    try:
        n = int(input("Enter a number: ").strip())
        limit_str = input("Enter table length (default 10): ").strip()
        limit = int(limit_str) if limit_str else 10

        if limit <= 0:
            raise ValueError("Length must be positive")

        for i in range(1, limit + 1):
            print(f"{n} x {i} = {n * i}")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

