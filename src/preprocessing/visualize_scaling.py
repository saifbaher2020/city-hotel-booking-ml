
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ==================================================
# STAGE 3: DATA CLEANING AND PREPROCESSING
# STEP 3.4 - NUMERICAL SCALING VISUALIZATIONS
# ==================================================

# --------------------------------------------------
# 3.4.1 - Locate the project files
# --------------------------------------------------

project_folder = Path(__file__).resolve().parents[2]

dataset_path = project_folder / "data" / "H2.csv"

split_path = (
    project_folder
    / "outputs"
    / "splits"
    / "split_assignments.csv"
)

features_path = (
    project_folder
    / "outputs"
    / "regression_features.txt"
)

figures_folder = (
    project_folder
    / "outputs"
    / "preprocessing"
    / "figures"
)

figures_folder.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 3.4.2 - Load data and Training partition
# --------------------------------------------------

df = pd.read_csv(dataset_path)
assignments = pd.read_csv(split_path)

assert len(df) == len(assignments)

assert assignments["OriginalRowIndex"].tolist() == list(
    range(len(df))
)

features = [
    line.strip()
    for line in features_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

train_mask = assignments["Partition"] == "Train"

train_df = df.loc[train_mask, features].copy()

numerical_columns = train_df.select_dtypes(
    include="number"
).columns.tolist()

# --------------------------------------------------
# 3.4.3 - Impute and scale Training data
# --------------------------------------------------

imputer = SimpleImputer(strategy="most_frequent")

train_numbers = imputer.fit_transform(
    train_df[numerical_columns]
)

scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_numbers)

before_df = pd.DataFrame(
    train_numbers,
    columns=numerical_columns,
    index=train_df.index
)

after_df = pd.DataFrame(
    train_scaled,
    columns=numerical_columns,
    index=train_df.index
)

print("=" * 55)
print("STAGE 3.4 - SCALING VISUALIZATIONS")
print("=" * 55)

print("\nTraining rows:", len(before_df))
print("Numerical features:", len(numerical_columns))

# --------------------------------------------------
# 3.4.4 - Set professional graph style
# --------------------------------------------------

sns.set_theme(style="whitegrid")

BLUE = "#2563A6"
TEAL = "#159895"

# --------------------------------------------------
# 3.4.5 - FIGURE 1:
# LeadTime distribution before and after scaling
# --------------------------------------------------

fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 5)
)

sns.histplot(
    before_df["LeadTime"],
    bins=50,
    color=BLUE,
    ax=axes[0]
)

axes[0].set_title("Before Scaling")
axes[0].set_xlabel("Lead Time (Days)")
axes[0].set_ylabel("Number of Bookings")

sns.histplot(
    after_df["LeadTime"],
    bins=50,
    color=TEAL,
    ax=axes[1]
)

axes[1].set_title("After StandardScaler")
axes[1].set_xlabel("Standardized Lead Time (Z-Score)")
axes[1].set_ylabel("Number of Bookings")

fig.suptitle(
    "Lead Time Distribution Before and After Scaling",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

path_1 = (
    figures_folder
    / "04_lead_time_before_after_scaling.png"
)

plt.savefig(path_1, dpi=200, bbox_inches="tight")
plt.close()

print("\nFigure 1 saved:", path_1)

# --------------------------------------------------
# 3.4.6 - FIGURE 2:
# Compare numerical feature scales
# --------------------------------------------------

selected_features = [
    "LeadTime",
    "Adults",
    "StaysInWeekNights"
]

# Calculate Training standard deviations
# before and after scaling.

before_std = (
    before_df[selected_features]
    .std(ddof=0)
)

after_std = (
    after_df[selected_features]
    .std(ddof=0)
)

comparison = pd.DataFrame({
    "Feature": selected_features,
    "Before Scaling": before_std.values,
    "After Scaling": after_std.values
})

fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 5)
)

sns.barplot(
    data=comparison,
    x="Feature",
    y="Before Scaling",
    color=BLUE,
    ax=axes[0]
)

axes[0].set_title("Before Scaling")
axes[0].set_xlabel("")
axes[0].set_ylabel("Standard Deviation")

sns.barplot(
    data=comparison,
    x="Feature",
    y="After Scaling",
    color=TEAL,
    ax=axes[1]
)

axes[1].set_title("After StandardScaler")
axes[1].set_xlabel("")
axes[1].set_ylabel("Standard Deviation")
axes[1].set_ylim(0, 1.2)

for ax in axes:
    ax.tick_params(axis="x", rotation=15)

fig.suptitle(
    "Numerical Feature Scale Comparison",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

path_2 = (
    figures_folder
    / "05_numerical_feature_scale_comparison.png"
)

plt.savefig(path_2, dpi=200, bbox_inches="tight")
plt.close()

print("Figure 2 saved:", path_2)

# --------------------------------------------------
# 3.4.7 - Save graph statistics
# --------------------------------------------------

summary_path = (
    figures_folder
    / "scaling_visualization_statistics.csv"
)

comparison.to_csv(summary_path, index=False)

# --------------------------------------------------
# 3.4.8 - Final verification
# --------------------------------------------------

print("\n" + "=" * 55)
print("STAGE 3.4 - VISUALIZATION COMPLETE")
print("=" * 55)

print("\nFeature standard deviations:")
print(comparison.round(3).to_string(index=False))

print("\nGraphs generated: 2")
print("Summary tables generated: 1")
print("Original H2.csv modified: NO")
print("Test partition accessed: NO")
print("Models trained: NO")
