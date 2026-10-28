
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# ==========================================================
# STAGE 4: INITIAL MACHINE LEARNING BASELINES
# STEP 4.2 - CLASSIFICATION BASELINE
# TARGET: IsCanceled
# ==========================================================

# ----------------------------------------------------------
# 4.2.1 - Locate project files
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
    / "classification_preprocessor.joblib"
)

features_path = (
    pipeline_folder
    / "classification_selected_features.txt"
)

output_folder = (
    project_folder
    / "outputs"
    / "baselines"
    / "classification"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)

# ----------------------------------------------------------
# 4.2.2 - Load dataset and split assignments
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

train_mask = (
    assignments["Partition"] == "Train"
)

validation_mask = (
    assignments["Partition"] == "Validation"
)

print("=" * 60)
print("STAGE 4.2 - CLASSIFICATION BASELINE")
print("=" * 60)

print("\nTraining rows:", train_mask.sum())
print("Validation rows:", validation_mask.sum())

# ----------------------------------------------------------
# 4.2.3 - Load feature list and saved preprocessor
# ----------------------------------------------------------

features = [
    line.strip()
    for line in features_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

target = "IsCanceled"

if target in features:
    raise ValueError(
        "Target leakage detected: IsCanceled is an input!"
    )

# These columns are not available as reliable
# initial-booking inputs in our chosen prediction setup.

forbidden_features = {
    "ADR",
    "ReservationStatus",
    "ReservationStatusDate"
}

unexpected = forbidden_features.intersection(features)

if unexpected:
    raise ValueError(
        f"Potential leakage features: {unexpected}"
    )

preprocessor = joblib.load(
    preprocessor_path
)

print("\nOriginal input features:", len(features))

# ----------------------------------------------------------
# 4.2.4 - Separate features (X) and labels (y)
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

assert set(y_train.unique()) == {0, 1}
assert set(y_validation.unique()) == {0, 1}

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nValidation class distribution:")
print(y_validation.value_counts().sort_index())

# ----------------------------------------------------------
# 4.2.5 - Apply saved preprocessing pipeline
# ----------------------------------------------------------

# IMPORTANT:
# Transform only.
# The preprocessor was already fitted on Training
# during Stage 3.

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
# 4.2.6 - Define our evaluation function
# ----------------------------------------------------------

def evaluate_classifier(
    model_name,
    actual,
    predicted,
    probabilities
):

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(
            actual, predicted
        ),
        "Precision": precision_score(
            actual,
            predicted,
            zero_division=0
        ),
        "Recall": recall_score(
            actual,
            predicted,
            zero_division=0
        ),
        "F1": f1_score(
            actual,
            predicted,
            zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            actual,
            probabilities
        )
    }

# ----------------------------------------------------------
# 4.2.7 - MODEL A: Dummy Majority Classifier
# ----------------------------------------------------------

dummy_model = DummyClassifier(
    strategy="most_frequent"
)

dummy_model.fit(
    X_train_prepared,
    y_train
)

dummy_predictions = dummy_model.predict(
    X_validation_prepared
)

dummy_probabilities = dummy_model.predict_proba(
    X_validation_prepared
)[:, 1]

dummy_results = evaluate_classifier(
    "Dummy Majority Classifier",
    y_validation,
    dummy_predictions,
    dummy_probabilities
)

print("\nDummy Classifier trained.")

# ----------------------------------------------------------
# 4.2.8 - MODEL B: Logistic Regression
# ----------------------------------------------------------

# Fixed, untuned baseline configuration.
# No class balancing or threshold tuning yet.

logistic_model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    random_state=42
)

logistic_model.fit(
    X_train_prepared,
    y_train
)

logistic_predictions = logistic_model.predict(
    X_validation_prepared
)

logistic_probabilities = logistic_model.predict_proba(
    X_validation_prepared
)[:, 1]

logistic_results = evaluate_classifier(
    "Logistic Regression",
    y_validation,
    logistic_predictions,
    logistic_probabilities
)

print("Logistic Regression trained.")

# ----------------------------------------------------------
# 4.2.9 - Display validation results
# ----------------------------------------------------------

results = pd.DataFrame([
    dummy_results,
    logistic_results
])

print("\n" + "=" * 60)
print("VALIDATION CLASSIFICATION PERFORMANCE")
print("=" * 60)

print(
    results.round(4).to_string(index=False)
)

# ----------------------------------------------------------
# 4.2.10 - Display confusion matrices
# ----------------------------------------------------------

dummy_cm = confusion_matrix(
    y_validation,
    dummy_predictions,
    labels=[0, 1]
)

logistic_cm = confusion_matrix(
    y_validation,
    logistic_predictions,
    labels=[0, 1]
)

print("\nDummy Classifier Confusion Matrix:")
print(dummy_cm)

print("\nLogistic Regression Confusion Matrix:")
print(logistic_cm)

print("\nConfusion matrix layout:")
print("[[True Negative, False Positive],")
print(" [False Negative, True Positive]]")

# ----------------------------------------------------------
# 4.2.11 - Save evaluation metrics
# ----------------------------------------------------------

metrics_path = (
    output_folder
    / "classification_baseline_metrics.csv"
)

results.to_csv(
    metrics_path,
    index=False
)

# ----------------------------------------------------------
# 4.2.12 - Save validation predictions
# ----------------------------------------------------------

predictions_df = pd.DataFrame({
    "OriginalRowIndex": df.index[validation_mask],
    "ActualIsCanceled": y_validation.to_numpy(),
    "DummyPrediction": dummy_predictions,
    "LogisticPrediction": logistic_predictions,
    "DummyCancellationProbability": dummy_probabilities,
    "LogisticCancellationProbability": logistic_probabilities
})

predictions_path = (
    output_folder
    / "classification_validation_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

# ----------------------------------------------------------
# 4.2.13 - Save the confusion matrices
# ----------------------------------------------------------

dummy_cm_df = pd.DataFrame(
    dummy_cm,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"]
)

logistic_cm_df = pd.DataFrame(
    logistic_cm,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"]
)

dummy_cm_df.to_csv(
    output_folder / "dummy_confusion_matrix.csv"
)

logistic_cm_df.to_csv(
    output_folder / "logistic_confusion_matrix.csv"
)

# ----------------------------------------------------------
# 4.2.14 - Save trained models
# ----------------------------------------------------------

joblib.dump(
    dummy_model,
    output_folder / "dummy_classifier.joblib"
)

joblib.dump(
    logistic_model,
    output_folder / "logistic_regression.joblib"
)

# ----------------------------------------------------------
# 4.2.15 - Final verification
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STAGE 4.2 - CLASSIFICATION BASELINES COMPLETE")
print("=" * 60)

print("\nModels trained: 2")
print("Metrics saved:", metrics_path)
print("Predictions saved:", predictions_path)
print("Original H2.csv modified: NO")
print("Test partition accessed: NO")
print("Preprocessing pipeline refitted: NO")
print("Hyperparameter tuning performed: NO")
print("Class balancing performed: NO")
print("Decision threshold: 0.50")
