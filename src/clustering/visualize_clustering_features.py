
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# STAGE 4: INITIAL MACHINE LEARNING BASELINES
# STEP 4.3 - INTERPRETABLE CLUSTER VISUALIZATIONS
# ==========================================================

# ----------------------------------------------------------
# 4.3.24 - Locate project files
# ----------------------------------------------------------

project_folder = Path(__file__).resolve().parents[2]

dataset_path = project_folder / "data" / "H2.csv"

splits_path = (
    project_folder
    / "outputs"
    / "splits"
    / "split_assignments.csv"
)

cluster_path = (
    project_folder
    / "outputs"
    / "baselines"
    / "clustering"
    / "clustering_train_assignments.csv"
)

figures_folder = (
    project_folder
    / "outputs"
    / "baselines"
    / "clustering"
    / "figures"
)

figures_folder.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# 4.3.25 - Load data and saved cluster assignments
# ----------------------------------------------------------

df = pd.read_csv(dataset_path)
splits = pd.read_csv(splits_path)
clusters = pd.read_csv(cluster_path)

assert len(df) == len(splits)

assert splits["OriginalRowIndex"].tolist() == list(
    range(len(df))
)

train_indices = splits.loc[
    splits["Partition"] == "Train",
    "OriginalRowIndex"
]

train_df = df.loc[
    train_indices,
    ["LeadTime", "Children"]
].copy()

clusters = clusters.set_index("OriginalRowIndex")

assert clusters.index.is_unique
assert set(clusters.index) == set(train_df.index)

train_df["Cluster"] = clusters.loc[
    train_df.index,
    "Cluster"
].to_numpy()

# Children missing values were previously
# imputed with zero using Training information.

train_df["Children"] = train_df["Children"].fillna(0)

cluster_ids = sorted(train_df["Cluster"].unique())

print("=" * 60)
print("STAGE 4.3 - INTERPRETABLE CLUSTER VISUALIZATIONS")
print("=" * 60)

print("\nTraining bookings:", len(train_df))

print("\nCluster sizes:")
print(train_df["Cluster"].value_counts().sort_index())

# ----------------------------------------------------------
# 4.3.26 - Configure professional plotting
# ----------------------------------------------------------

sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold"
})

colors = {
    0: "#218B78",
    1: "#3766A8",
    2: "#D69B43"
}

# ----------------------------------------------------------
# 4.3.27 - Create reproducible sample
# ----------------------------------------------------------

# Use an equal-sized sample from each cluster
# so smaller groups remain visible.
#
# Sampling is only for visualization.
# Cluster sizes and model results stay unchanged.

SAMPLE_PER_CLUSTER = 1200

plot_parts = []

for cluster_id in cluster_ids:

    subset = train_df[
        train_df["Cluster"] == cluster_id
    ]

    sampled = subset.sample(
        n=min(SAMPLE_PER_CLUSTER, len(subset)),
        random_state=42
    )

    plot_parts.append(sampled)

plot_df = pd.concat(plot_parts).copy()

# Jitter is used for plotting only.
# It stops bookings with identical Children
# values from completely covering one another.

rng = np.random.default_rng(42)

plot_df["ChildrenJittered"] = (
    plot_df["Children"]
    + rng.uniform(
        -0.13,
        0.13,
        size=len(plot_df)
    )
)

# ----------------------------------------------------------
# 4.3.28 - FIGURE 1:
# LeadTime vs. Children, colored by cluster
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 7))

for cluster_id in cluster_ids:

    subset = plot_df[
        plot_df["Cluster"] == cluster_id
    ]

    ax.scatter(
        subset["LeadTime"],
        subset["ChildrenJittered"],
        s=20,
        alpha=0.45,
        color=colors[cluster_id],
        label=f"Cluster {cluster_id}",
        edgecolors="none"
    )

ax.set_title(
    "Hotel Booking Clusters: Lead Time vs. Children"
)

ax.set_xlabel("Lead Time (Days)")
ax.set_ylabel("Number of Children (Visual Jitter Applied)")

ax.legend(
    title="Assigned Cluster",
    frameon=True
)

plt.tight_layout()

figure_1 = (
    figures_folder
    / "05_clusters_leadtime_vs_children.png"
)

plt.savefig(
    figure_1,
    dpi=220,
    bbox_inches="tight"
)

plt.close()

print("\nFigure 1 saved:", figure_1)

# ----------------------------------------------------------
# 4.3.29 - FIGURE 2:
# Separate panel for each cluster
# ----------------------------------------------------------

fig, axes = plt.subplots(
    1,
    len(cluster_ids),
    figsize=(16, 5),
    sharex=True,
    sharey=True
)

for ax, cluster_id in zip(axes, cluster_ids):

    subset = plot_df[
        plot_df["Cluster"] == cluster_id
    ]

    ax.scatter(
        subset["LeadTime"],
        subset["ChildrenJittered"],
        s=16,
        alpha=0.45,
        color=colors[cluster_id],
        edgecolors="none"
    )

    total_count = (
        train_df["Cluster"] == cluster_id
    ).sum()

    ax.set_title(
        f"Cluster {cluster_id}\n"
        f"{total_count:,} bookings"
    )

    ax.set_xlabel("Lead Time (Days)")

axes[0].set_ylabel(
    "Children (Visual Jitter Applied)"
)

fig.suptitle(
    "Hotel Booking Cluster Characteristics",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

figure_2 = (
    figures_folder
    / "06_separate_cluster_panels.png"
)

plt.savefig(
    figure_2,
    dpi=220,
    bbox_inches="tight"
)

plt.close()

print("Figure 2 saved:", figure_2)

# ----------------------------------------------------------
# 4.3.30 - FIGURE 3:
# LeadTime distribution by cluster
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 7))

for cluster_id in cluster_ids:

    subset = train_df[
        train_df["Cluster"] == cluster_id
    ]

    sns.histplot(
        subset["LeadTime"],
        stat="density",
        bins=50,
        element="step",
        fill=False,
        linewidth=2,
        color=colors[cluster_id],
        label=f"Cluster {cluster_id}",
        ax=ax
    )

ax.set_title(
    "Lead Time Distribution Across Clusters"
)

ax.set_xlabel("Lead Time (Days)")
ax.set_ylabel("Density")

ax.legend(title="Cluster")

plt.tight_layout()

figure_3 = (
    figures_folder
    / "07_leadtime_distribution_by_cluster.png"
)

plt.savefig(
    figure_3,
    dpi=220,
    bbox_inches="tight"
)

plt.close()

print("Figure 3 saved:", figure_3)

# ----------------------------------------------------------
# 4.3.31 - Save real cluster statistics
# ----------------------------------------------------------

summary = train_df.groupby("Cluster").agg(
    NumberOfBookings=("LeadTime", "size"),
    MeanLeadTime=("LeadTime", "mean"),
    MedianLeadTime=("LeadTime", "median"),
    MeanChildren=("Children", "mean"),
    MedianChildren=("Children", "median")
)

summary_path = (
    figures_folder
    / "interpretable_cluster_statistics.csv"
)

summary.to_csv(summary_path)

print("\nActual cluster characteristics:")
print(summary.round(3).to_string())

# ----------------------------------------------------------
# 4.3.32 - Final verification
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STAGE 4.3 - FEATURE VISUALIZATION COMPLETE")
print("=" * 60)

print("\nFigures generated: 3")
print("Summary tables generated: 1")
print("Clustering model retrained: NO")
print("Cluster assignments modified: NO")
print("Test partition accessed: NO")
print("Original H2.csv modified: NO")
