
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score
)

# ==========================================================
# STAGE 4: INITIAL MACHINE LEARNING BASELINES
# STEP 4.2 - CLASSIFICATION EVALUATION VISUALIZATIONS
# ==========================================================

# ----------------------------------------------------------
# 4.2.16 - Locate saved classification results
# ----------------------------------------------------------

project_folder = Path(__file__).resolve().parents[2]

results_folder = (
    project_folder
    / "outputs"
    / "baselines"
    / "classification"
)

metrics_path = (
    results_folder
    / "classification_baseline_metrics.csv"
)

predictions_path = (
    results_folder
    / "classification_validation_predictions.csv"
)

figures_folder = results_folder / "figures"
figures_folder.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# 4.2.17 - Load existing Validation results
# ----------------------------------------------------------

metrics = pd.read_csv(metrics_path)
predictions = pd.read_csv(predictions_path)

required_columns = [
    "ActualIsCanceled",
    "DummyPrediction",
    "LogisticPrediction",
    "DummyCancellationProbability",
    "LogisticCancellationProbability"
]

for column in required_columns:
    if column not in predictions.columns:
        raise ValueError(f"Missing column: {column}")

actual = predictions["ActualIsCanceled"].to_numpy()
dummy_pred = predictions["DummyPrediction"].to_numpy()
logistic_pred = predictions["LogisticPrediction"].to_numpy()

dummy_prob = predictions[
    "DummyCancellationProbability"
].to_numpy()

logistic_prob = predictions[
    "LogisticCancellationProbability"
].to_numpy()

assert len(actual) == 11333
assert set(np.unique(actual)) == {0, 1}
assert np.isfinite(logistic_prob).all()
assert np.isfinite(dummy_prob).all()

print("=" * 60)
print("STAGE 4.2 - CLASSIFICATION VISUALIZATIONS")
print("=" * 60)

print("\nValidation bookings:", len(actual))
print("\nSaved baseline results:")
print(metrics.round(4).to_string(index=False))

# ----------------------------------------------------------
# 4.2.18 - Configure professional graph styling
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

BLUE = "#3766A8"
TEAL = "#218B78"
AMBER = "#D69B43"
DARK = "#263348"

# ----------------------------------------------------------
# 4.2.19 - FIGURE 1: Confusion Matrix
# ----------------------------------------------------------

cm = confusion_matrix(
    actual,
    logistic_pred,
    labels=[0, 1]
)

fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt=",d",
    cmap="Blues",
    cbar=False,
    linewidths=1,
    linecolor="white",
    xticklabels=["Not Canceled (0)", "Canceled (1)"],
    yticklabels=["Not Canceled (0)", "Canceled (1)"],
    annot_kws={"size": 15, "weight": "bold"},
    ax=ax
)

ax.set_title(
    "Logistic Regression - Validation Confusion Matrix"
)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")

plt.tight_layout()

figure_1 = (
    figures_folder / "01_confusion_matrix.png"
)

plt.savefig(
    figure_1,
    dpi=220,
    bbox_inches="tight"
)

plt.close()

print("\nFigure 1 saved:", figure_1)

# ----------------------------------------------------------
# 4.2.20 - FIGURE 2: Precision-Recall Curve
# ----------------------------------------------------------

precision, recall, pr_thresholds = (
    precision_recall_curve(
        actual,
        logistic_prob
    )
)

average_precision = average_precision_score(
    actual,
    logistic_prob
)

# Values at our fixed original threshold of 0.50
tp = ((actual == 1) & (logistic_pred == 1)).sum()
fp = ((actual == 0) & (logistic_pred == 1)).sum()
fn = ((actual == 1) & (logistic_pred == 0)).sum()

baseline_precision = tp / (tp + fp)
baseline_recall = tp / (tp + fn)

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(
    recall,
    precision,
    color=BLUE,
    linewidth=2.5,
    label=f"Logistic Regression (AP = {average_precision:.3f})"
)

ax.scatter(
    baseline_recall,
    baseline_precision,
    color=AMBER,
    s=100,
    zorder=5,
    label="Original Threshold = 0.50"
)

# Cancellation prevalence is a useful reference
prevalence = actual.mean()

ax.axhline(
    prevalence,
    color=DARK,
    linestyle="--",
    alpha=0.6,
    label=f"Cancellation Rate = {prevalence:.3f}"
)

ax.set_title(
    "Precision-Recall Curve - Logistic Regression"
)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_xlim(0, 1.01)
ax.set_ylim(0, 1.05)
ax.legend(loc="lower left", frameon=True)

plt.tight_layout()

figure_2 = (
    figures_folder / "02_precision_recall_curve.png"
)

plt.savefig(
    figure_2,
    dpi=220,
    bbox_inches="tight"
)

plt.close()

print("Figure 2 saved:", figure_2)

# ----------------------------------------------------------
# 4.2.21 - FIGURE 3: ROC Curve
# ----------------------------------------------------------

fpr_logistic, tpr_logistic, _ = roc_curve(
    actual,
    logistic_prob
)

fpr_dummy, tpr_dummy, _ = roc_curve(
    actual,
    dummy_prob
)

auc_logistic = auc(
    fpr_logistic,
    tpr_logistic
)

auc_dummy = auc(
    fpr_dummy,
    tpr_dummy
)

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(
    fpr_logistic,
    tpr_logistic,
    linewidth=2.5,
    color=BLUE,
    label=f"Logistic Regression (AUC = {auc_logistic:.3f})"
)

ax.plot(
    fpr_dummy,
    tpr_dummy,
    linestyle="--",
    linewidth=1.8,
    color=AMBER,
    label=f"Dummy Classifier (AUC = {auc_dummy:.3f})"
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle=":",
    color=DARK,
    alpha=0.7,
    label="Random Ranking Reference"
)

ax.set_title(
    "ROC Curve - Classification Baseline Comparison"
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(loc="lower right", frameon=True)

plt.tight_layout()

figure_3 = (
    figures_folder / "03_roc_curve.png"
)

plt.savefig(
    figure_3,
    dpi=220,
    bbox_inches="tight"
)

plt.close()

print("Figure 3 saved:", figure_3)

# ----------------------------------------------------------
# 4.2.22 - FIGURE 4: Baseline Metric Comparison
# ----------------------------------------------------------

selected_metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC"
]

comparison = metrics.melt(
    id_vars="Model",
    value_vars=selected_metrics,
    var_name="Metric",
    value_name="Score"
)

fig, ax = plt.subplots(figsize=(12, 6))

sns.barplot(
    data=comparison,
    x="Metric",
    y="Score",
    hue="Model",
    palette=[AMBER, BLUE],
    ax=ax
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f",
        padding=3,
        fontsize=10
    )

ax.set_title(
    "Classification Baseline - Validation Performance"
)

ax.set_xlabel("")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.12)
ax.legend(
    title="Model",
    loc="upper left",
    frameon=False
)

plt.tight_layout()

figure_4 = (
    figures_folder / "04_classification_baseline_comparison.png"
)

plt.savefig(
    figure_4,
    dpi=220,
    bbox_inches="tight"
)

plt.close()

print("Figure 4 saved:", figure_4)

# ----------------------------------------------------------
# 4.2.23 - Save additional figure statistics
# ----------------------------------------------------------

figure_summary = pd.DataFrame([
    {
        "Statistic": "Validation Bookings",
        "Value": len(actual)
    },
    {
        "Statistic": "Cancellation Prevalence",
        "Value": prevalence
    },
    {
        "Statistic": "Logistic Average Precision",
        "Value": average_precision
    },
    {
        "Statistic": "Logistic ROC AUC",
        "Value": auc_logistic
    },
    {
        "Statistic": "Dummy ROC AUC",
        "Value": auc_dummy
    },
    {
        "Statistic": "Baseline Precision",
        "Value": baseline_precision
    },
    {
        "Statistic": "Baseline Recall",
        "Value": baseline_recall
    }
])

summary_path = (
    figures_folder
    / "classification_figure_statistics.csv"
)

figure_summary.to_csv(
    summary_path,
    index=False
)

# ----------------------------------------------------------
# 4.2.24 - Final verification
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STAGE 4.2 - VISUALIZATION COMPLETE")
print("=" * 60)

print("\nConfusion Matrix:")
print(cm)

print(f"\nROC AUC: {auc_logistic:.4f}")
print(f"Average Precision: {average_precision:.4f}")

print("\nFigures generated: 4")
print("Summary tables generated: 1")
print("Models retrained: NO")
print("Decision threshold changed: NO")
print("Test partition accessed: NO")
print("Original H2.csv modified: NO")

print("\nCLASSIFICATION BASELINE EVALUATION COMPLETE")
