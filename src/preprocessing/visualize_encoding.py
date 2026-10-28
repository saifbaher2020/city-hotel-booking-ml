
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder

# ==================================================
# STAGE 3: DATA CLEANING AND PREPROCESSING
# STEP 3.3 - CATEGORICAL ENCODING VISUALIZATIONS
# ==================================================

# --------------------------------------------------
# 3.3.1 - Locate project files
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
# 3.3.2 - Load original data and training rows
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

categorical_columns = train_df.select_dtypes(
    include=["object", "str", "string", "category"]
).columns.tolist()

train_categories = train_df[
    categorical_columns
].fillna("Unknown")

# --------------------------------------------------
# 3.3.3 - Fit encoder on Training only
# --------------------------------------------------

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
    dtype=int
)

encoder.fit(train_categories)

encoded_names = encoder.get_feature_names_out(
    categorical_columns
)

original_count = len(categorical_columns)
encoded_count = len(encoded_names)

print("=" * 55)
print("STAGE 3.3 - ENCODING VISUALIZATIONS")
print("=" * 55)

print("\nOriginal categorical features:", original_count)
print("Encoded categorical features:", encoded_count)

# --------------------------------------------------
# 3.3.4 - Set professional graph style
# --------------------------------------------------

sns.set_theme(style="whitegrid")

BLUE = "#2563A6"
TEAL = "#159895"

# --------------------------------------------------
# 3.3.5 - FIGURE 1: One-Hot Encoding Example
# --------------------------------------------------

# Use actual meal categories from Training.
# The demonstration shows one booking per category.

meal_categories = sorted(
    train_categories["Meal"].unique()
)

example = pd.DataFrame({
    "Meal": meal_categories
})

meal_encoder = OneHotEncoder(
    sparse_output=False,
    dtype=int
)

example_encoded = meal_encoder.fit_transform(
    example[["Meal"]]
)

meal_encoded_names = meal_encoder.get_feature_names_out(
    ["Meal"]
)

fig, axes = plt.subplots(
    2, 1,
    figsize=(11, 7),
    gridspec_kw={"height_ratios": [1, 2]}
)

# Original categorical values
axes[0].axis("off")

original_table = axes[0].table(
    cellText=example.values,
    colLabels=["Original Meal"],
    cellLoc="center",
    loc="center"
)

original_table.auto_set_font_size(False)
original_table.set_fontsize(11)
original_table.scale(1, 1.6)

axes[0].set_title(
    "Before Encoding: One Text Column",
    fontsize=13,
    pad=8
)

# Encoded numerical matrix
sns.heatmap(
    example_encoded,
    annot=True,
    fmt="d",
    cmap=sns.color_palette(
        ["#F1F5F9", BLUE],
        as_cmap=True
    ),
    cbar=False,
    linewidths=1,
    linecolor="white",
    xticklabels=meal_encoded_names,
    yticklabels=example["Meal"],
    ax=axes[1]
)

axes[1].set_title(
    "After Encoding: Numerical Indicator Columns",
    fontsize=13,
    pad=12
)

axes[1].set_xlabel("One-Hot Encoded Features")
axes[1].set_ylabel("Original Meal Category")

plt.suptitle(
    "One-Hot Encoding: Meal Feature",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

path_1 = (
    figures_folder
    / "01_one_hot_encoding_example.png"
)

plt.savefig(path_1, dpi=200, bbox_inches="tight")
plt.close()

print("\nFigure 1 saved:", path_1)

# --------------------------------------------------
# 3.3.6 - FIGURE 2: Before vs. After
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(
    ["Before Encoding", "After Encoding"],
    [original_count, encoded_count],
    color=[TEAL, BLUE],
    width=0.55
)

for bar in bars:
    height = bar.get_height()

    ax.annotate(
        f"{int(height)}",
        xy=(
            bar.get_x() + bar.get_width() / 2,
            height
        ),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=14,
        fontweight="bold"
    )

ax.set_title(
    "Categorical Feature Count Before and After Encoding",
    fontsize=15,
    fontweight="bold"
)

ax.set_ylabel("Number of Columns")
ax.set_ylim(0, encoded_count * 1.18)

plt.tight_layout()

path_2 = (
    figures_folder
    / "02_feature_count_before_after.png"
)

plt.savefig(path_2, dpi=200, bbox_inches="tight")
plt.close()

print("Figure 2 saved:", path_2)

# --------------------------------------------------
# 3.3.7 - FIGURE 3: Expansion by original feature
# --------------------------------------------------

# categories_ contains the categories learned
# by the fitted OneHotEncoder for each feature.

category_counts = pd.Series(
    {
        column: len(categories)
        for column, categories in zip(
            categorical_columns,
            encoder.categories_
        )
    }
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(11, 6))

bars = ax.barh(
    category_counts.index[::-1],
    category_counts.values[::-1],
    color=BLUE
)

for bar in bars:
    width = bar.get_width()

    ax.annotate(
        str(int(width)),
        xy=(
            width,
            bar.get_y() + bar.get_height() / 2
        ),
        xytext=(5, 0),
        textcoords="offset points",
        va="center"
    )

ax.set_title(
    "Number of Encoded Features per Original Column",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Number of One-Hot Columns")
ax.set_ylabel("Original Categorical Feature")
ax.set_xlim(0, category_counts.max() * 1.15)

plt.tight_layout()

path_3 = (
    figures_folder
    / "03_encoded_features_by_column.png"
)

plt.savefig(path_3, dpi=200, bbox_inches="tight")
plt.close()

print("Figure 3 saved:", path_3)

# --------------------------------------------------
# 3.3.8 - Save summary table
# --------------------------------------------------

summary = pd.DataFrame({
    "OriginalFeature": category_counts.index,
    "EncodedColumns": category_counts.values
})

summary_path = (
    figures_folder
    / "encoding_feature_counts.csv"
)

summary.to_csv(summary_path, index=False)

# --------------------------------------------------
# 3.3.9 - Final results
# --------------------------------------------------

print("\n" + "=" * 55)
print("STAGE 3.3 - VISUALIZATION COMPLETE")
print("=" * 55)

print("\nOriginal categorical columns:", original_count)
print("Encoded categorical columns:", encoded_count)

print("\nEncoded columns by feature:")
print(category_counts.to_string())

print("\nGraphs generated: 3")
print("Summary tables generated: 1")
print("Original H2.csv modified: NO")
print("Test partition accessed: NO")
print("Models trained: NO")
