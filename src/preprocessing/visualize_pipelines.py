
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ==========================================================
# STAGE 3.6 / 5.1.8
# PREPROCESSING PIPELINE VISUALIZATION AND VERIFICATION
# ==========================================================

project_folder = Path(__file__).resolve().parents[2]

dataset_path = project_folder / "data" / "H2.csv"
splits_path = (
    project_folder / "outputs" / "splits"
    / "split_assignments.csv"
)
pipelines_folder = (
    project_folder / "outputs" / "preprocessing" / "pipelines"
)
summary_path = (
    pipelines_folder / "preprocessing_pipeline_summary.csv"
)
figures_folder = (
    project_folder / "outputs" / "preprocessing" / "figures"
)

figures_folder.mkdir(parents=True, exist_ok=True)

expected_tasks = [
    "regression",
    "classification",
    "clustering",
]

expected_original = {
    "regression": 19,
    "classification": 19,
    "clustering": 10,
}

expected_prepared = {
    "regression": 211,
    "classification": 211,
    "clustering": 30,
}

# ----------------------------------------------------------
# 1. Read and verify pipeline summary
# ----------------------------------------------------------

summary = pd.read_csv(summary_path)

assert len(summary) == 3
assert set(summary["Task"]) == set(expected_tasks)

summary = (
    summary.set_index("Task")
    .loc[expected_tasks]
    .reset_index()
)

for _, row in summary.iterrows():
    task = row["Task"]
    assert int(row["OriginalFeatures"]) == expected_original[task]
    assert int(row["PreparedFeatures"]) == expected_prepared[task]

print("=" * 60)
print("STAGE 5.1.8 - PIPELINE VISUALIZATION")
print("=" * 60)
print(summary.to_string(index=False))

# ----------------------------------------------------------
# 2. Figure styling
# ----------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

BLUE = "#3766A8"
TEAL = "#218B78"
AMBER = "#D69B43"
DARK = "#263348"
LIGHT = "#EDF2F8"

# ----------------------------------------------------------
# 3. Figure 1 - Complete preprocessing workflow
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 10))
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.axis("off")


def draw_box(
    x, y, width, height, title, subtitle,
    facecolor=LIGHT, edgecolor=BLUE
):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.15",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)

    ax.text(
        x + width / 2, y + height * 0.65,
        title,
        ha="center", va="center",
        fontsize=12, fontweight="bold",
        color=DARK,
    )

    ax.text(
        x + width / 2, y + height * 0.30,
        subtitle,
        ha="center", va="center",
        fontsize=9.5, color=DARK,
    )


def arrow(x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.6,
            color=DARK,
        )
    )


draw_box(
    3.5, 10.35, 5, 1.05,
    "Original Hotel Dataset",
    "79,330 bookings",
)
arrow(6, 10.35, 6, 9.8)

draw_box(
    3.5, 8.75, 5, 1.05,
    "Reproducible Data Split",
    "Train: 56,664 | Validation: 11,333 | Test: 11,333",
)
arrow(6, 8.75, 6, 8.2)

draw_box(
    3.5, 7.15, 5, 1.05,
    "Task-Specific Feature Selection",
    "19 regression | 19 classification | 10 clustering",
)

arrow(6, 7.15, 6, 6.7)
arrow(6, 6.7, 3, 6.7)
arrow(6, 6.7, 9, 6.7)
arrow(3, 6.7, 3, 6.2)
arrow(9, 6.7, 9, 6.2)

draw_box(
    0.5, 4.85, 5, 1.35,
    "Numerical Pipeline",
    "Most-frequent imputation + StandardScaler",
    facecolor="#E7F3F0",
    edgecolor=TEAL,
)

draw_box(
    6.5, 4.85, 5, 1.35,
    "Categorical Pipeline",
    "Unknown imputation + One-Hot Encoding",
    facecolor="#FFF5E5",
    edgecolor=AMBER,
)

arrow(3, 4.85, 3, 4.25)
arrow(9, 4.85, 9, 4.25)
arrow(3, 4.25, 6, 4.25)
arrow(9, 4.25, 6, 4.25)
arrow(6, 4.25, 6, 3.8)

prepared = summary.set_index("Task")["PreparedFeatures"]

prepared_text = (
    f"{int(prepared['regression'])} regression | "
    f"{int(prepared['classification'])} classification | "
    f"{int(prepared['clustering'])} clustering"
)

draw_box(
    3.5, 2.65, 5, 1.15,
    "Prepared Feature Matrix",
    prepared_text,
    facecolor="#E7F3F0",
    edgecolor=TEAL,
)

ax.text(
    6, 1.65,
    "Learned transformations are fitted on TRAINING ONLY.",
    ha="center", va="center",
    fontsize=12, fontweight="bold", color=TEAL,
)

ax.text(
    6, 1.1,
    "Validation uses fitted rules. Test is reserved.",
    ha="center", va="center",
    fontsize=10, color=DARK,
)

ax.set_title(
    "Hotel Booking Dataset: Preprocessing Workflow",
    fontsize=17, fontweight="bold", pad=15,
)

plt.tight_layout()

figure_1 = (
    figures_folder / "06_complete_preprocessing_workflow.png"
)
plt.savefig(figure_1, dpi=220, bbox_inches="tight")
plt.close()

print("\nFigure 1 saved:", figure_1)

# ----------------------------------------------------------
# 4. Figure 2 - Feature expansion
# ----------------------------------------------------------

task_labels = [
    "Regression",
    "Classification",
    "Clustering",
]

original_counts = summary["OriginalFeatures"].to_numpy()
prepared_counts = summary["PreparedFeatures"].to_numpy()

x = np.arange(len(task_labels))
width = 0.34

fig, ax = plt.subplots(figsize=(11, 6))

bars_before = ax.bar(
    x - width / 2,
    original_counts,
    width,
    label="Original features",
    color=BLUE,
)

bars_after = ax.bar(
    x + width / 2,
    prepared_counts,
    width,
    label="Prepared features",
    color=TEAL,
)

for bars in (bars_before, bars_after):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height,
            ),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

ax.set_xticks(x)
ax.set_xticklabels(task_labels)
ax.set_ylabel("Number of Features")
ax.set_title("Feature Expansion After Preprocessing")
ax.set_ylim(0, prepared_counts.max() * 1.17)
ax.legend(frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

figure_2 = (
    figures_folder / "07_preprocessing_feature_expansion.png"
)
plt.savefig(figure_2, dpi=220, bbox_inches="tight")
plt.close()

print("Figure 2 saved:", figure_2)

# ----------------------------------------------------------
# 5. Load dataset and saved split assignments
# ----------------------------------------------------------

df = pd.read_csv(dataset_path)
assignments = pd.read_csv(splits_path)

assert len(df) == len(assignments)
assert assignments["OriginalRowIndex"].tolist() == list(
    range(len(df))
)
assert set(assignments["Partition"]) == {
    "Train", "Validation", "Test"
}

train_mask = assignments["Partition"] == "Train"
validation_mask = assignments["Partition"] == "Validation"

# Test rows are not selected or transformed.

# ----------------------------------------------------------
# 6. Verify saved pipelines
# ----------------------------------------------------------

verification_rows = []

for task_name in expected_tasks:

    print("\n" + "-" * 55)
    print("VERIFYING:", task_name.upper())
    print("-" * 55)

    pipeline_path = (
        pipelines_folder
        / f"{task_name}_preprocessor.joblib"
    )

    features_path = (
        pipelines_folder
        / f"{task_name}_selected_features.txt"
    )

    preprocessor = joblib.load(pipeline_path)

    features = [
        line.strip()
        for line in features_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert len(features) == expected_original[task_name]
    assert "DepositType" not in features
    assert "ADR" not in features
    assert "IsCanceled" not in features

    prepared_names = preprocessor.get_feature_names_out()

    assert len(prepared_names) == expected_prepared[task_name]
    assert not any(
        "DepositType" in name for name in prepared_names
    )

    X_train = df.loc[train_mask, features].copy()
    X_validation = df.loc[
        validation_mask, features
    ].copy()

    # Only transform, never refit.
    prepared_train = preprocessor.transform(X_train)
    prepared_validation = preprocessor.transform(
        X_validation
    )

    expected_row = summary.loc[
        summary["Task"] == task_name
    ].iloc[0]

    expected_shape_train = (
        int(expected_row["TrainingRows"]),
        expected_prepared[task_name],
    )

    expected_shape_validation = (
        int(expected_row["ValidationRows"]),
        expected_prepared[task_name],
    )

    assert prepared_train.shape == expected_shape_train
    assert prepared_validation.shape == (
        expected_shape_validation
    )

    def is_finite(matrix):
        if hasattr(matrix, "toarray"):
            return np.isfinite(matrix.data).all()
        return np.isfinite(matrix).all()

    assert is_finite(prepared_train)
    assert is_finite(prepared_validation)

    sample = X_validation.head(10)
    first = preprocessor.transform(sample)
    second = preprocessor.transform(sample)

    if hasattr(first, "toarray"):
        first = first.toarray()
        second = second.toarray()

    assert np.array_equal(first, second)

    print("Pipeline loaded successfully.")
    print("Training shape:", prepared_train.shape)
    print("Validation shape:", prepared_validation.shape)
    print("Repeated transformation: IDENTICAL")
    print("Verification: PASSED")

    verification_rows.append({
        "Task": task_name,
        "OriginalFeatures": len(features),
        "PreparedFeatures": len(prepared_names),
        "TrainingRows": prepared_train.shape[0],
        "ValidationRows": prepared_validation.shape[0],
        "FiniteValues": True,
        "DeterministicTransform": True,
        "VerificationPassed": True,
    })

# ----------------------------------------------------------
# 7. Save verification summary
# ----------------------------------------------------------

verification_df = pd.DataFrame(verification_rows)

verification_path = (
    pipelines_folder / "pipeline_verification_summary.csv"
)

verification_df.to_csv(
    verification_path,
    index=False,
)

print("\n" + "=" * 60)
print("STAGE 5.1.8 - PIPELINE VERIFICATION COMPLETE")
print("=" * 60)

print(verification_df.to_string(index=False))

print("\nGraphs generated: 2")
print("Saved pipelines verified: 3")
print("Original H2.csv modified: NO")
print("Test bookings transformed: NO")
print("Pipelines refitted: NO")
print("Models retrained: NO")
