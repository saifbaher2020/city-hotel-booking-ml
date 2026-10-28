
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# STEP 1: Locate the project folder
# ==========================================

project_folder = Path(__file__).resolve().parents[2]

# ==========================================
# STEP 2: Load the original dataset
# ==========================================

dataset_path = project_folder / "data" / "H2.csv"
df = pd.read_csv(dataset_path)

# ==========================================
# STEP 3: Prepare data for visualization
# ==========================================

# Show the common ADR range between 0 and 300.
# The original dataset is NOT modified.
adr_for_plot = df.loc[df["ADR"].between(0, 300), "ADR"]

# ==========================================
# STEP 4: Create the histogram
# ==========================================

plt.figure(figsize=(10, 6))

plt.hist(adr_for_plot, bins=50)

plt.title("Distribution of Hotel Average Daily Rate (0-300)")
plt.xlabel("ADR (Average Daily Rate)")
plt.ylabel("Number of Bookings")

plt.xlim(0, 300)
plt.tight_layout()

# ==========================================
# STEP 5: Save the histogram
# ==========================================

output_path = project_folder / "outputs" / "adr_distribution.png"

plt.savefig(output_path, dpi=150)

print("Graph saved successfully!")
print("Saved to:", output_path)

# ==========================================
# STEP 6: Display the histogram
# ==========================================

plt.show()
