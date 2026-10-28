
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# STAGE 3: DATA CLEANING AND PREPROCESSING
# STEP 3.1 - DATA SPLITTING VISUALIZATION
# ==================================================

# --------------------------------------------------
# 3.1.1 - Locate project files
# --------------------------------------------------

project_folder = Path(__file__).resolve().parents[2]

dataset_path = project_folder / "data" / "H2.csv"

split_path = (
    project_folder
    / "outputs"
    / "splits"
    / "split_assignments.csv"
)

figures_folder = (
    project_folder
    / "outputs"
    / "splits"
    / "figures"
)

figures_folder.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 3.1.2 - Load dataset and split assignments
# --------------------------------------------------

df = pd.read_csv(dataset_path)
assignments = pd.read_csv(split_path)

assert len(df) == len(assignments)

# Check that every row has a split assignment
assert assignments["OriginalRowIndex"].tolist() == list(range(len(df)))

valid_splits = ["Train", "Validation", "Test"]

assert set(assignments["Partition"]) == set(valid_splits)

# Add partition labels to a temporary copy
# We do NOT modify the original H2.csv.

plot_df = df[["IsCanceled"]].copy()
plot_df["Partition"] = assignments["Partition"].to_numpy()

# --------------------------------------------------
# 3.1.3 - Set professional graph style
# --------------------------------------------------

sns.set_theme(style="whitegrid", context="talk")

colors = {
    "Train": "#278B79",
    "Validation": "#D49B45",
    "Test": "#6477B8",
}

# --------------------------------------------------
# 3.1.4 - Graph 1: Dataset split distribution
# --------------------------------------------------

split_counts = (
    plot_df["Partition"]
    .value_counts()
    .reindex(valid_splits)
)

total = len(plot_df)

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    valid_splits,
    split_counts.values,
    color=[colors[name] for name in valid_splits]
)

for bar, count in zip(bars, split_counts.values):
    percentage = count / total * 100

    ax.annotate(
        f"{count:,}\n({percentage:.2f}%)",
        xy=(bar.get_x() + bar.get_width() / 2, count),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=12
    )

ax.set_title("Dataset Split Distribution")
ax.set_xlabel("Dataset Partition")
ax.set_ylabel("Number of Bookings")
ax.set_ylim(0, split_counts.max() * 1.2)

plt.tight_layout()

graph_1_path = figures_folder / "01_split_distribution.png"

plt.savefig(graph_1_path, dpi=200, bbox_inches="tight")
plt.close()

print("Graph 1 saved:", graph_1_path)

# --------------------------------------------------
# 3.1.5 - Calculate cancellation statistics
# --------------------------------------------------

balance = pd.crosstab(
    plot_df["Partition"],
    plot_df["IsCanceled"]
)

balance = balance.reindex(
    index=valid_splits,
    columns=[0, 1],
    fill_value=0
)

balance.columns = ["NotCanceled", "Canceled"]

balance["Total"] = (
    balance["NotCanceled"] + balance["Canceled"]
)

balance["CancellationRate"] = (
    balance["Canceled"] / balance["Total"] * 100
)

print("\nCancellation balance by partition:")
print(balance.round(2).to_string())

# --------------------------------------------------
# 3.1.6 - Graph 2: Cancellation balance
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(valid_splits))
bar_width = 0.35

not_canceled_bars = ax.bar(
    [i - bar_width / 2 for i in x],
    balance["NotCanceled"],
    bar_width,
    label="Not Canceled",
    color="#278B79"
)

canceled_bars = ax.bar(
    [i + bar_width / 2 for i in x],
    balance["Canceled"],
    bar_width,
    label="Canceled",
    color="#D49B45"
)

for bars in [not_canceled_bars, canceled_bars]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10
        )

ax.set_xticks(list(x))
ax.set_xticklabels(valid_splits)

ax.set_title("Cancellation Class Balance Across Data Splits")
ax.set_xlabel("Dataset Partition")
ax.set_ylabel("Number of Bookings")
ax.set_ylim(0, balance["NotCanceled"].max() * 1.2)

ax.legend()

plt.tight_layout()

graph_2_path = figures_folder / "02_cancellation_balance.png"

plt.savefig(graph_2_path, dpi=200, bbox_inches="tight")
plt.close()

print("\nGraph 2 saved:", graph_2_path)

# --------------------------------------------------
# 3.1.7 - Save summary table
# --------------------------------------------------

summary_path = figures_folder / "split_statistics.csv"

balance.to_csv(summary_path)

print("Summary table saved:", summary_path)

# --------------------------------------------------
# 3.1.8 - Final verification
# --------------------------------------------------

print("\n" + "=" * 55)
print("STAGE 3.1 - VISUALIZATION COMPLETE")
print("=" * 55)

print("Graphs generated: 2")
print("Summary tables generated: 1")
print("Original H2.csv modified: NO")
print("Data split modified: NO")
print("Models trained: NO")
