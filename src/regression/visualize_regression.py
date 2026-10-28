
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# STAGE 4: INITIAL MACHINE LEARNING BASELINES
# STEP 4.1 - REGRESSION EVALUATION VISUALIZATIONS
# ==========================================================

# ----------------------------------------------------------
# 4.1.13 - Locate the saved regression results
# ----------------------------------------------------------

project_folder = Path(__file__).resolve().parents[2]

results_folder = (
    project_folder
    / "outputs"
    / "baselines"
    / "regression"
)

metrics_path = (
    results_folder
    / "regression_baseline_metrics.csv"
)

predictions_path = (
    results_folder
    / "regression_validation_predictions.csv"
)

figures_folder = results_folder / "figures"

figures_folder.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# 4.1.14 - Load the existing results
# ----------------------------------------------------------

metrics = pd.read_csv(metrics_path)
predictions = pd.read_csv(predictions_path)

required_columns = [
    "ActualADR",
    "DummyPredictedADR",
    "LinearPredictedADR"
]

assert all(
    column in predictions.columns
    for column in required_columns
)

assert len(predictions) == 11333

print("=" * 60)
print("STAGE 4.1 - REGRESSION VISUALIZATIONS")
print("=" * 60)

print("\nValidation bookings:", len(predictions))
print("\nSaved model metrics:")
print(metrics.round(4).to_string(index=False))

# ----------------------------------------------------------
# 4.1.15 - Prepare actual values and prediction errors
# ----------------------------------------------------------

actual = predictions["ActualADR"]

dummy_predicted = predictions["DummyPredictedADR"]

linear_predicted = predictions["LinearPredictedADR"]

# Residual = Actual - Predicted
# Positive residual: model underestimated ADR.
# Negative residual: model overestimated ADR.

linear_residuals = actual - linear_predicted

print("\nLinear Regression residual statistics:")
print(linear_residuals.describe().round(3))

# ----------------------------------------------------------
# 4.1.16 - Configure professional graph styling
# ----------------------------------------------------------

sns.set_theme(
    style="whitegrid",
    context="notebook"
)

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
# 4.1.17 - FIGURE 1:
# Actual vs. Predicted ADR
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 7))

# Random sampling for visual clarity only.
# All 11,333 bookings remain included in the
# model evaluation metrics.

plot_data = predictions.sample(
    n=min(5000, len(predictions)),
    random_state=42
)

sns.scatterplot(
    data=plot_data,
    x="ActualADR",
    y="LinearPredictedADR",
    s=28,
    alpha=0.35,
    color=BLUE,
    edgecolor=None,
    ax=ax
)

# The diagonal represents perfect prediction.
# Every point on this line would have:
# Actual ADR = Predicted ADR.

max_value = max(
    plot_data["ActualADR"].max(),
    plot_data["LinearPredictedADR"].max()
)

min_value = min(
    plot_data["ActualADR"].min(),
    plot_data["LinearPredictedADR"].min()
)

ax.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=2,
    color=AMBER,
    label="Perfect Prediction"
)

ax.set_title(
    "Linear Regression: Actual vs. Predicted ADR"
)

ax.set_xlabel("Actual ADR")
ax.set_ylabel("Predicted ADR")

ax.legend(frameon=True)

plt.tight_layout()

figure_1 = (
    figures_folder
    / "01_actual_vs_predicted_adr.png"
)

plt.savefig(
    figure_1,
    dpi=220,
    bbox_inches="tight"
)

plt.close()

print("\nFigure 1 saved:", figure_1)

# ----------------------------------------------------------
# 4.1.18 - FIGURE 2:
# Prediction error distribution
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

sns.histplot(
    linear_residuals,
    bins=70,
    color=TEAL,
    edgecolor="white",
    ax=ax
)

ax.axvline(
    x=0,
    color=AMBER,
    linestyle="--",
    linewidth=2,
    label="Zero Error"
)

ax.set_title(
    "Linear Regression: Validation Residual Distribution"
)

ax.set_xlabel(
    "Residual = Actual ADR - Predicted ADR"
)

ax.set_ylabel("Number of Bookings")

ax.legend()

plt.tight_layout()

figure_2 = (
    figures_folder
    / "02_regression_residual_distribution.png"
)

plt.savefig(
    figure_2,
    dpi=220,
    bbox_inches="tight"
)

plt.close()

print("Figure 2 saved:", figure_2)

# ----------------------------------------------------------
# 4.1.19 - FIGURE 3:
# Compare Dummy and Linear Regression
# ----------------------------------------------------------

comparison = metrics.melt(
    id_vars="Model",
    value_vars=["MAE", "RMSE"],
    var_name="Metric",
    value_name="Error"
)

fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(
    data=comparison,
    x="Metric",
    y="Error",
    hue="Model",
    palette=[AMBER, BLUE],
    ax=ax
)

# Display metric values above the bars.

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f",
        padding=4,
        fontsize=10
    )

ax.set_title(
    "Regression Baseline Performance Comparison"
)

ax.set_xlabel("")
ax.set_ylabel("Validation Error (ADR Units)")
ax.set_ylim(0, comparison["Error"].max() * 1.2)

ax.legend(
    title="Model",
    frameon=False
)

plt.tight_layout()

figure_3 = (
    figures_folder
    / "03_regression_baseline_comparison.png"
)

plt.savefig(
    figure_3,
    dpi=220,
    bbox_inches="tight"
)

plt.close()

print("Figure 3 saved:", figure_3)

# ----------------------------------------------------------
# 4.1.20 - Save residual statistics
# ----------------------------------------------------------

residual_summary = pd.DataFrame({
    "Statistic": [
        "Count",
        "Mean Residual",
        "Median Residual",
        "Residual Standard Deviation",
        "Minimum Residual",
        "Maximum Residual"
    ],
    "Value": [
        len(linear_residuals),
        linear_residuals.mean(),
        linear_residuals.median(),
        linear_residuals.std(),
        linear_residuals.min(),
        linear_residuals.max()
    ]
})

summary_path = (
    figures_folder
    / "regression_residual_summary.csv"
)

residual_summary.to_csv(
    summary_path,
    index=False
)

# ----------------------------------------------------------
# 4.1.21 - Final verification
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STAGE 4.1 - VISUALIZATION COMPLETE")
print("=" * 60)

print("\nFigures generated: 3")
print("Summary tables generated: 1")
print("Validation predictions analyzed:", len(predictions))
print("Models retrained: NO")
print("Test partition accessed: NO")
print("Original H2.csv modified: NO")

print("\nREGRESSION BASELINE EVALUATION COMPLETE")
