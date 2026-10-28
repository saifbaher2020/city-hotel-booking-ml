
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_samples, silhouette_score

# ==========================================================
# STAGE 4: INITIAL MACHINE LEARNING BASELINES
# STEP 4.3 - CLUSTERING EVALUATION VISUALIZATIONS
# ==========================================================

# ----------------------------------------------------------
# 4.3.13 - Locate project files
# ----------------------------------------------------------

project_folder = Path(__file__).resolve().parents[2]

dataset_path = project_folder / "data" / "H2.csv"

splits_path = (
    project_folder
    / "outputs"
    / "splits"
    / "split_assignments.csv"
)

pipeline_folder = (
    project_folder
    / "outputs"
    / "preprocessing"
    / "pipelines"
)

results_folder = (
    project_folder
    / "outputs"
    / "baselines"
    / "clustering"
)

figures_folder = results_folder / "figures"
figures_folder.mkdir(parents=True, exist_ok=True)

assignments_path = (
    results_folder
    / "clustering_train_assignments.csv"
)

model_path = (
    results_folder
    / "minibatch_kmeans.joblib"
)

preprocessor_path = (
    pipeline_folder
    / "clustering_preprocessor.joblib"
)

features_path = (
    pipeline_folder
    / "clustering_selected_features.txt"
)

# ----------------------------------------------------------
# 4.3.14 - Load saved results
# ----------------------------------------------------------

df = pd.read_csv(dataset_path)
splits = pd.read_csv(splits_path)
cluster_assignments = pd.read_csv(assignments_path)

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

features = [
    line.strip()
    for line in features_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

assert len(df) == len(splits)

assert splits["OriginalRowIndex"].tolist() == list(
    range(len(df))
)

# Select Training rows ONLY.
train_mask = splits["Partition"] == "Train"

train_df = df.loc[train_mask, features].copy()

# Match saved cluster labels to the correct rows.
cluster_assignments = (
    cluster_assignments
    .set_index("OriginalRowIndex")
)

assert cluster_assignments.index.is_unique
assert set(cluster_assignments.index) == set(train_df.index)

cluster_assignments = cluster_assignments.loc[
    train_df.index
]

labels = cluster_assignments["Cluster"].to_numpy()

cluster_ids = np.sort(np.unique(labels))

print("=" * 60)
print("STAGE 4.3 - CLUSTERING VISUALIZATIONS")
print("=" * 60)

print("\nTraining bookings:", len(train_df))
print("Clusters:", cluster_ids.tolist())

# ----------------------------------------------------------
# 4.3.15 - Apply saved preprocessing
# ----------------------------------------------------------

# Transform only: no fitting or retraining.

X_prepared = preprocessor.transform(train_df)

assert X_prepared.shape[0] == len(labels)

# Check that the saved model produces the same labels.
predicted_labels = model.predict(X_prepared)

assert np.array_equal(predicted_labels, labels), (
    "Saved labels do not match the saved model."
)

print("Saved model labels verified: PASSED")

# ----------------------------------------------------------
# 4.3.16 - Configure professional graph styling
# ----------------------------------------------------------

sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "savefig.facecolor": "white"
})

COLORS = ["#3766A8", "#218B78", "#D69B43"]

# ----------------------------------------------------------
# 4.3.17 - FIGURE 1: Cluster size distribution
# ----------------------------------------------------------

cluster_counts = (
    pd.Series(labels)
    .value_counts()
    .sort_index()
)

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    [f"Cluster {i}" for i in cluster_counts.index],
    cluster_counts.values,
    color=COLORS[:len(cluster_counts)]
)

total = len(labels)

for bar, count in zip(bars, cluster_counts.values):

    percentage = count / total * 100

    ax.annotate(
        f"{count:,}\n({percentage:.1f}%)",
        xy=(
            bar.get_x() + bar.get_width() / 2,
            count
        ),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontweight="bold"
    )

ax.set_title(
    "K-Means Baseline: Cluster Size Distribution"
)

ax.set_ylabel("Number of Bookings")
ax.set_xlabel("Assigned Cluster")
ax.set_ylim(0, cluster_counts.max() * 1.2)

plt.tight_layout()

figure_1 = (
    figures_folder
    / "01_cluster_size_distribution.png"
)

plt.savefig(figure_1, dpi=220, bbox_inches="tight")
plt.close()

print("\nFigure 1 saved:", figure_1)

# ----------------------------------------------------------
# 4.3.18 - Prepare a reproducible visualization sample
# ----------------------------------------------------------

# We use a 2,000-booking sample for the remaining
# visualizations to keep computations manageable.
#
# Cluster labels still come from the original
# model trained on all 56,664 Training bookings.

SAMPLE_SIZE = min(2000, len(train_df))

rng = np.random.default_rng(42)

sample_indices = rng.choice(
    len(train_df),
    size=SAMPLE_SIZE,
    replace=False
)

X_sample = X_prepared[sample_indices]
labels_sample = labels[sample_indices]

# ----------------------------------------------------------
# 4.3.19 - FIGURE 2: Two-dimensional cluster projection
# ----------------------------------------------------------

# Our encoded matrix may be sparse.
# TruncatedSVD supports sparse matrices without
# requiring us to convert all data to dense.
#
# This is a visualization only.
# It does NOT affect the fitted clustering model.

projection = TruncatedSVD(
    n_components=2,
    random_state=42
)

X_2d = projection.fit_transform(X_sample)

projection_df = pd.DataFrame({
    "Component1": X_2d[:, 0],
    "Component2": X_2d[:, 1],
    "Cluster": [
        f"Cluster {i}"
        for i in labels_sample
    ]
})

fig, ax = plt.subplots(figsize=(10, 7))

sns.scatterplot(
    data=projection_df,
    x="Component1",
    y="Component2",
    hue="Cluster",
    palette=COLORS[:len(cluster_ids)],
    s=25,
    alpha=0.6,
    edgecolor=None,
    ax=ax
)

variance_explained = (
    projection.explained_variance_ratio_.sum() * 100
)

ax.set_title(
    "Clustering Visualization (2D TruncatedSVD Projection)"
)

ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")

ax.text(
    0.02,
    0.02,
    f"Variance represented: {variance_explained:.1f}%",
    transform=ax.transAxes,
    fontsize=10,
    bbox={
        "facecolor": "white",
        "alpha": 0.9,
        "edgecolor": "lightgray"
    }
)

ax.legend(title="Cluster", frameon=True)

plt.tight_layout()

figure_2 = (
    figures_folder
    / "02_clustering_2d_projection.png"
)

plt.savefig(figure_2, dpi=220, bbox_inches="tight")
plt.close()

print("Figure 2 saved:", figure_2)

# ----------------------------------------------------------
# 4.3.20 - FIGURE 3: Silhouette plot
# ----------------------------------------------------------

sample_silhouette = silhouette_samples(
    X_sample,
    labels_sample
)

average_silhouette = silhouette_score(
    X_sample,
    labels_sample
)

fig, ax = plt.subplots(figsize=(10, 7))

y_lower = 10

for index, cluster_id in enumerate(cluster_ids):

    cluster_values = sample_silhouette[
        labels_sample == cluster_id
    ]

    cluster_values.sort()

    cluster_size = len(cluster_values)
    y_upper = y_lower + cluster_size

    ax.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        cluster_values,
        facecolor=COLORS[index],
        alpha=0.85
    )

    ax.text(
        -0.08,
        y_lower + 0.5 * cluster_size,
        f"Cluster {cluster_id}",
        fontsize=10
    )

    y_lower = y_upper + 10

ax.axvline(
    x=average_silhouette,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Average Silhouette = {average_silhouette:.3f}"
)

ax.set_title(
    "Silhouette Analysis of K-Means Baseline"
)

ax.set_xlabel("Silhouette Coefficient")
ax.set_ylabel("Sampled Bookings by Cluster")
ax.set_yticks([])
ax.set_xlim(-0.3, 1.0)
ax.legend(loc="lower right")

plt.tight_layout()

figure_3 = (
    figures_folder
    / "03_clustering_silhouette_plot.png"
)

plt.savefig(figure_3, dpi=220, bbox_inches="tight")
plt.close()

print("Figure 3 saved:", figure_3)

# ----------------------------------------------------------
# 4.3.21 - FIGURE 4: Cluster profile heatmap
# ----------------------------------------------------------

# Use real, interpretable original booking features
# rather than the 30 encoded numerical indicators.

profile_features = [
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies"
]

profile_df = train_df[
    profile_features
].copy()

profile_df["Cluster"] = labels

# Children has four missing values in Training.
# Use the previously selected imputation value (0).
profile_df["Children"] = (
    profile_df["Children"].fillna(0)
)

# Mean of each feature within each cluster.
cluster_means = (
    profile_df
    .groupby("Cluster")[profile_features]
    .mean()
)

# Standardize the cluster means using the overall
# Training mean and standard deviation.
#
# This is ONLY for visualizing differences.
# The original saved clustering model is unchanged.

overall_mean = profile_df[
    profile_features
].mean()

overall_std = profile_df[
    profile_features
].std(ddof=0)

overall_std = overall_std.replace(0, 1)

normalized_profiles = (
    cluster_means - overall_mean
) / overall_std

normalized_profiles.index = [
    f"Cluster {i}"
    for i in normalized_profiles.index
]

fig, ax = plt.subplots(figsize=(12, 5))

sns.heatmap(
    normalized_profiles,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.7,
    linecolor="white",
    cbar_kws={
        "label": "Deviation from Overall Training Mean (SD)"
    },
    ax=ax
)

ax.set_title(
    "Cluster Profiles: Standardized Booking Characteristics"
)

ax.set_xlabel("Booking Feature")
ax.set_ylabel("Cluster")
ax.tick_params(axis="x", rotation=25)

plt.tight_layout()

figure_4 = (
    figures_folder
    / "04_cluster_profile_heatmap.png"
)

plt.savefig(figure_4, dpi=220, bbox_inches="tight")
plt.close()

print("Figure 4 saved:", figure_4)

# ----------------------------------------------------------
# 4.3.22 - Save profile statistics
# ----------------------------------------------------------

cluster_means.to_csv(
    figures_folder / "cluster_profile_means.csv"
)

normalized_profiles.to_csv(
    figures_folder / "cluster_profile_standardized.csv"
)

# ----------------------------------------------------------
# 4.3.23 - Final verification
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STAGE 4.3 - VISUALIZATION COMPLETE")
print("=" * 60)

print("\nCluster sizes:")
print(cluster_counts.to_string())

print(f"\nSampled Silhouette Score: {average_silhouette:.4f}")
print(f"2D projection variance represented: {variance_explained:.2f}%")

print("\nFigures generated: 4")
print("Profile tables generated: 2")
print("Model retrained: NO")
print("Preprocessor refitted: NO")
print("Test partition accessed: NO")
print("Original H2.csv modified: NO")

print("\nCLUSTERING BASELINE EVALUATION COMPLETE")
