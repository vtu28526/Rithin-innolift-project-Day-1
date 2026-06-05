import matplotlib.pyplot as plt

# Sample outbreak risk data
risk_classes = ["Low", "Medium", "High", "Critical"]
risk_scores = [25, 48, 72, 91]

vaccination_rate = [88, 70, 52, 35]
sanitation_score = [82, 65, 45, 28]
population_density = [1200, 2800, 4500, 6800]

# Custom color list
color_list = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

plt.figure(figsize=(10, 6))

# Bar chart
bars = plt.bar(
    risk_classes,
    risk_scores,
    color=color_list,
    edgecolor="#222222",
    linewidth=1.5,
    label="Risk Index"
)

# Horizontal threshold lines
plt.axhline(
    y=50,
    color="#3498db",
    linestyle="--",
    linewidth=2,
    label="Medium Risk Threshold"
)

plt.axhline(
    y=80,
    color="#c0392b",
    linestyle="--",
    linewidth=2,
    label="Critical Risk Threshold"
)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 2,
        f"{height}%",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

# Custom styling
plt.title(
    "Disease Outbreak Risk Predictor",
    fontsize=18,
    fontweight="bold",
    color="#2c3e50"
)

plt.xlabel("Risk Classes", fontsize=13, fontweight="bold")
plt.ylabel("Risk Index (%)", fontsize=13, fontweight="bold")

plt.ylim(0, 110)
plt.grid(axis="y", linestyle=":", alpha=0.6)

plt.legend()
plt.tight_layout()

plt.show()
