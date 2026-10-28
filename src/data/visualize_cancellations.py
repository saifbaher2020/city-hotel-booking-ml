
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# STEP 1: Locate the project folder
# ==========================================

project_folder = Path(__file__).resolve().parents[2]

# ==========================================
# STEP 2: Load the dataset
# ==========================================

dataset_path = project_folder / "data" / "H2.csv"
df = pd.read_csv(dataset_path)

# ==========================================
# STEP 3: Count canceled and non-canceled bookings
# ==========================================

counts = df["IsCanceled"].value_counts()

not_canceled = counts.get(0, 0)
canceled = counts.get(1, 0)

total = len(df)

print("Total bookings:", total)
print("Not canceled:", not_canceled)
print("Canceled:", canceled)

print("\nCancellation percentages:")
print("Not canceled:", round(not_canceled / total * 100, 2), "%")
print("Canceled:", round(canceled / total * 100, 2), "%")

# ==========================================
# STEP 4: Create the bar chart
# ==========================================

labels = ["Not Canceled", "Canceled"]
values = [not_canceled, canceled]

plt.figure(figsize=(8, 6))

bars = plt.bar(labels, values)

plt.title("Hotel Booking Cancellation Distribution")
plt.xlabel("Booking Status")
plt.ylabel("Number of Bookings")

# Display the count above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height):,}",
        ha="center",
        va="bottom"
    )

plt.ylim(0, max(values) * 1.15)
plt.tight_layout()

# ==========================================
# STEP 5: Save the graph
# ==========================================

output_path = project_folder / "outputs" / "cancellation_distribution.png"

plt.savefig(output_path, dpi=150)

print("\nGraph saved successfully!")
print("Saved to:", output_path)

# ==========================================
# STEP 6: Display the graph
# ==========================================

plt.show()
