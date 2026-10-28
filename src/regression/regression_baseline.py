
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================================
# STAGE 4: INITIAL MACHINE LEARNING BASELINES
# STEP 4.1 - REGRESSION BASELINE (PREDICT ADR)
# ==========================================================

# ----------------------------------------------------------
# 4.1.1 - Locate project files
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

preprocessor_path = (
    pipeline_folder
    / "regression_preprocessor.joblib"
)

features_path = (
    pipeline_folder
    / "regression_selected_features.txt"
)

output_folder = (
    project_folder
    / "outputs"
    / "baselines"
    / "regression"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)

# ----------------------------------------------------------
# 4.1.2 - Load data and split assignments
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

print("=" * 60)
print("STAGE 4.1 - REGRESSION BASELINE")
print("=" * 60)

print("\nTraining rows:", train_mask.sum())
print("Validation rows:", validation_mask.sum())

# ----------------------------------------------------------
# 4.1.3 - Load selected features and fitted preprocessor
# ----------------------------------------------------------

features = [
    line.strip()
    for line in features_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

target = "ADR"

if target in features:
    raise ValueError(
        "Target leakage: ADR is included in input features!"
    )

preprocessor = joblib.load(preprocessor_path)

print("\nOriginal input features:", len(features))

# ----------------------------------------------------------
# 4.1.4 - Separate X (features) and y (target)
# ----------------------------------------------------------

X_train = df.loc[
    train_mask, features
].copy()

X_validation = df.loc[
    validation_mask, features
].copy()

y_train = df.loc[
    train_mask, target
].copy()

y_validation = df.loc[
    validation_mask, target
].copy()

assert y_train.notna().all()
assert y_validation.notna().all()

print("Training targets:", len(y_train))
print("Validation targets:", len(y_validation))

# ----------------------------------------------------------
# 4.1.5 - Transform using the saved pipeline
# ----------------------------------------------------------

# IMPORTANT:
# Transform only. The preprocessing pipeline
# was already fitted on Training in Stage 3.
# Never fit it again using Validation.

X_train_prepared = preprocessor.transform(
    X_train
)

X_validation_prepared = preprocessor.transform(
    X_validation
)

print("\nPrepared Training shape:")
print(X_train_prepared.shape)

print("\nPrepared Validation shape:")
print(X_validation_prepared.shape)

# ----------------------------------------------------------
# 4.1.6 - Define evaluation function
# ----------------------------------------------------------

def evaluate_model(
    model_name,
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual, predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual, predicted
        )
    )

    r2 = r2_score(
        actual, predicted
    )

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

# ----------------------------------------------------------
# 4.1.7 - MODEL A: Dummy Mean Regressor
# ----------------------------------------------------------

dummy_model = DummyRegressor(
    strategy="mean"
)

dummy_model.fit(
    X_train_prepared,
    y_train
)

dummy_predictions = dummy_model.predict(
    X_validation_prepared
)

dummy_results = evaluate_model(
    "Dummy Mean Regressor",
    y_validation,
    dummy_predictions
)

print("\nDummy Regressor trained.")

# ----------------------------------------------------------
# 4.1.8 - MODEL B: Linear Regression
# ----------------------------------------------------------

linear_model = LinearRegression()

linear_model.fit(
    X_train_prepared,
    y_train
)

linear_predictions = linear_model.predict(
    X_validation_prepared
)

linear_results = evaluate_model(
    "Linear Regression",
    y_validation,
    linear_predictions
)

print("Linear Regression trained.")

# ----------------------------------------------------------
# 4.1.9 - Compare baseline results
# ----------------------------------------------------------

results = pd.DataFrame([
    dummy_results,
    linear_results
])

print("\n" + "=" * 60)
print("VALIDATION PERFORMANCE")
print("=" * 60)

print(
    results.round(4).to_string(index=False)
)

# ----------------------------------------------------------
# 4.1.10 - Save results and Validation predictions
# ----------------------------------------------------------

results_path = (
    output_folder
    / "regression_baseline_metrics.csv"
)

results.to_csv(
    results_path,
    index=False
)

predictions = pd.DataFrame({
    "OriginalRowIndex": df.index[validation_mask],
    "ActualADR": y_validation.to_numpy(),
    "DummyPredictedADR": dummy_predictions,
    "LinearPredictedADR": linear_predictions
})

predictions_path = (
    output_folder
    / "regression_validation_predictions.csv"
)

predictions.to_csv(
    predictions_path,
    index=False
)

# ----------------------------------------------------------
# 4.1.11 - Save trained regression models
# ----------------------------------------------------------

joblib.dump(
    dummy_model,
    output_folder / "dummy_regressor.joblib"
)

joblib.dump(
    linear_model,
    output_folder / "linear_regression.joblib"
)

# ----------------------------------------------------------
# 4.1.12 - Final verification
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STAGE 4.1 - REGRESSION BASELINES COMPLETE")
print("=" * 60)

print("\nModels trained: 2")
print("Metrics saved:", results_path)
print("Predictions saved:", predictions_path)
print("Original H2.csv modified: NO")
print("Test partition accessed: NO")
print("Preprocessing pipeline refitted: NO")
print("Hyperparameter tuning performed: NO")
