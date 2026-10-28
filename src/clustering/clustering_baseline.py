
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

# ==========================================================
# STAGE 4: INITIAL MACHINE LEARNING BASELINES
# STEP 4.3 - K-MEANS CLUSTERING BASELINE
# ==========================================================

# ----------------------------------------------------------
# 4.3.1 - Locate project files
# ----------------------------------------------------------

project_folder = Path(__file__).resolve().parents[2]

dataset_path = (
    project_folder / "data" / "H2.csv"
)

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

preprocessor_path = (
    pipeline_folder
    / "clustering_preprocessor.joblib"
)

features_path = (
    pipeline_folder
    / "clustering_selected_features.txt"
)

output_folder = (
    project_folder
    / "outputs"
    / "baselines"
    / "clustering"
)

output_folder.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# 4.3.2 - Load the dataset and existing split
# ----------------------------------------------------------

df = pd.read_csv(dataset_path)
assignments = pd.read_csv(splits_path)

assert len(df) == len(assignments)
assert assignments["OriginalRowIndex"].tolist() == list(
    range(len(df))
)

train_mask = assignments["Partition"] == "Train"

print("=" * 60)
print("STAGE 4.3 - CLUSTERING BASELINE")
print("=" * 60)

print("\nTraining rows:", train_mask.sum())

# ----------------------------------------------------------
# 4.3.3 - Load selected clustering features
# ----------------------------------------------------------

features = [
    line.strip()
    for line in features_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

forbidden_targets = {
    "ADR",
    "IsCanceled",
    "ReservationStatus"
}

if forbidden_targets.intersection(features):
    raise ValueError(
        "Target or outcome information found in clustering features!"
    )

X_train = df.loc[
    train_mask, features
].copy()

print("\nClustering features:")
print(features)

# ----------------------------------------------------------
# 4.3.4 - Apply saved preprocessing pipeline
# ----------------------------------------------------------

preprocessor = joblib.load(preprocessor_path)

# Transform only.
# Preprocessing rules were learned in Stage 3.
# No refitting takes place here.

X_prepared = preprocessor.transform(X_train)

print("\nPrepared Training matrix:", X_prepared.shape)

# ----------------------------------------------------------
# 4.3.5 - Train a fixed, untuned clustering baseline
# ----------------------------------------------------------

K = 3

model = MiniBatchKMeans(
    n_clusters=K,
    random_state=42,
    batch_size=1024,
    n_init=10,
    max_iter=100
)

cluster_labels = model.fit_predict(X_prepared)

print("\nClustering model trained.")
print("Number of clusters:", K)

# ----------------------------------------------------------
# 4.3.6 - Calculate cluster sizes
# ----------------------------------------------------------

cluster_sizes = (
    pd.Series(cluster_labels)
    .value_counts()
    .sort_index()
)

print("\nCluster sizes:")
print(cluster_sizes.to_string())

assert cluster_sizes.sum() == len(X_train)

# ----------------------------------------------------------
# 4.3.7 - Evaluate sampled silhouette score
# ----------------------------------------------------------

# The Silhouette Score compares distances within
# clusters to distances to other clusters.
#
# Using all 56,664 rows can be computationally
# expensive, so use a reproducible sample.

SAMPLE_SIZE = min(2000, len(X_train))

rng = np.random.default_rng(42)

sample_positions = rng.choice(
    len(X_train),
    size=SAMPLE_SIZE,
    replace=False
)

X_sample = X_prepared[sample_positions]
labels_sample = cluster_labels[sample_positions]

# Both metrics below use Euclidean distances.

if len(np.unique(labels_sample)) >= 2:

    silhouette = silhouette_score(
        X_sample,
        labels_sample,
        metric="euclidean"
    )

    # Davies-Bouldin expects dense input.
    if hasattr(X_sample, "toarray"):
        X_sample_dense = X_sample.toarray()
    else:
        X_sample_dense = np.asarray(X_sample)

    db_index = davies_bouldin_score(
        X_sample_dense,
        labels_sample
    )

else:
    silhouette = np.nan
    db_index = np.nan

# ----------------------------------------------------------
# 4.3.8 - Display clustering metrics
# ----------------------------------------------------------

metrics = pd.DataFrame([{
    "Algorithm": "MiniBatchKMeans",
    "K": K,
    "TrainingRows": len(X_train),
    "PreparedFeatures": X_prepared.shape[1],
    "SilhouetteSampleSize": SAMPLE_SIZE,
    "SilhouetteScore": silhouette,
    "DaviesBouldinIndex": db_index,
    "Inertia": model.inertia_
}])

print("\n" + "=" * 60)
print("CLUSTERING BASELINE METRICS")
print("=" * 60)

print(metrics.round(4).to_string(index=False))

# ----------------------------------------------------------
# 4.3.9 - Save cluster assignments
# ----------------------------------------------------------

cluster_results = pd.DataFrame({
    "OriginalRowIndex": X_train.index,
    "Cluster": cluster_labels
})

cluster_results_path = (
    output_folder
    / "clustering_train_assignments.csv"
)

cluster_results.to_csv(
    cluster_results_path,
    index=False
)

# ----------------------------------------------------------
# 4.3.10 - Save cluster sizes
# ----------------------------------------------------------

sizes_df = pd.DataFrame({
    "Cluster": cluster_sizes.index,
    "NumberOfBookings": cluster_sizes.values
})

sizes_df["Percentage"] = (
    sizes_df["NumberOfBookings"]
    / len(X_train)
    * 100
)

sizes_path = (
    output_folder
    / "clustering_cluster_sizes.csv"
)

sizes_df.to_csv(sizes_path, index=False)

# ----------------------------------------------------------
# 4.3.11 - Save metrics and trained model
# ----------------------------------------------------------

metrics_path = (
    output_folder
    / "clustering_baseline_metrics.csv"
)

metrics.to_csv(metrics_path, index=False)

model_path = (
    output_folder
    / "minibatch_kmeans.joblib"
)

joblib.dump(model, model_path)

# ----------------------------------------------------------
# 4.3.12 - Final verification
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STAGE 4.3 - CLUSTERING BASELINE COMPLETE")
print("=" * 60)

print("\nModel trained: MiniBatchKMeans")
print("Clusters requested:", K)
print("Training bookings clustered:", len(X_train))
print("Model saved:", model_path)
print("Test partition accessed: NO")
print("Preprocessor refitted: NO")
print("Hyperparameter tuning: NO")
print("Original H2.csv modified: NO")
