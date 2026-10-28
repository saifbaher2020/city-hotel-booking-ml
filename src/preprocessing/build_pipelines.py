
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==================================================
# STAGE 3: DATA CLEANING AND PREPROCESSING
# STEP 3.5 - REUSABLE PREPROCESSING PIPELINES
# ==================================================

# --------------------------------------------------
# 3.5.1 - Locate project files
# --------------------------------------------------

project_folder = Path(__file__).resolve().parents[2]

dataset_path = project_folder / "data" / "H2.csv"

split_path = (
    project_folder
    / "outputs"
    / "splits"
    / "split_assignments.csv"
)

output_folder = (
    project_folder
    / "outputs"
    / "preprocessing"
    / "pipelines"
)

output_folder.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 3.5.2 - Define the three ML tasks
# --------------------------------------------------

tasks = {
    "regression": {
        "features_file": "regression_features.txt",
        "target": "ADR"
    },
    "classification": {
        "features_file": "classification_features.txt",
        "target": "IsCanceled"
    },
    "clustering": {
        "features_file": "clustering_features.txt",
        "target": None
    }
}

# --------------------------------------------------
# 3.5.3 - Load dataset and existing splits
# --------------------------------------------------

df = pd.read_csv(dataset_path)
assignments = pd.read_csv(split_path)

assert len(df) == len(assignments)

assert assignments["OriginalRowIndex"].tolist() == list(
    range(len(df))
)

assert set(assignments["Partition"]) == {
    "Train",
    "Validation",
    "Test"
}

train_mask = assignments["Partition"] == "Train"
validation_mask = assignments["Partition"] == "Validation"

print("=" * 60)
print("STAGE 3.5 - REUSABLE PREPROCESSING PIPELINES")
print("=" * 60)

print("\nTraining rows:", train_mask.sum())
print("Validation rows:", validation_mask.sum())

# --------------------------------------------------
# 3.5.4 - Build a reusable preprocessing function
# --------------------------------------------------

def create_preprocessor(numerical_columns, categorical_columns):

    # Numerical preprocessing:
    # 1. Replace missing values with most frequent
    # 2. Standardize using Training statistics

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical preprocessing:
    # 1. Replace missing categories with Unknown
    # 2. Convert categories into numeric indicators

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                    keep_empty_features=True
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            )
        ]
    )

    # Apply the correct pipeline to each feature type

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ],
        sparse_threshold=1.0
    )

    return preprocessor

# --------------------------------------------------
# 3.5.5 - Process each machine learning task
# --------------------------------------------------

summary_rows = []

for task_name, configuration in tasks.items():

    print("\n" + "-" * 60)
    print("TASK:", task_name.upper())
    print("-" * 60)

    # Read the feature list from Stage 2

    features_path = (
        project_folder
        / "outputs"
        / configuration["features_file"]
    )

    features = [
        line.strip()
        for line in features_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    # Confirm that all selected features exist

    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    # Prevent accidentally including the prediction target

    target = configuration["target"]

    if target is not None and target in features:
        raise ValueError(
            f"Target leakage detected: {target}"
        )

    # Separate the selected Training and Validation rows

    X_train = df.loc[train_mask, features].copy()
    X_validation = df.loc[validation_mask, features].copy()

    # Identify numerical and categorical columns

    numerical_columns = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = [
        column
        for column in features
        if column not in numerical_columns
    ]

    print("Original features:", len(features))
    print("Numerical features:", len(numerical_columns))
    print("Categorical features:", len(categorical_columns))

    # Create a new independent pipeline for this task

    preprocessor = create_preprocessor(
        numerical_columns,
        categorical_columns
    )

    # IMPORTANT:
    # Fit on Training only.

    X_train_prepared = preprocessor.fit_transform(
        X_train
    )

    # Apply Training rules to Validation.
    # Do NOT fit again.

    X_validation_prepared = preprocessor.transform(
        X_validation
    )

    # Both matrices must have the same columns

    assert (
        X_train_prepared.shape[1]
        == X_validation_prepared.shape[1]
    )

    # Check that no NaN values remain

    if hasattr(X_train_prepared, "data"):
        assert np.isfinite(X_train_prepared.data).all()
        assert np.isfinite(X_validation_prepared.data).all()
    else:
        assert np.isfinite(X_train_prepared).all()
        assert np.isfinite(X_validation_prepared).all()

    # --------------------------------------------------
    # 3.5.6 - Save fitted pipeline
    # --------------------------------------------------

    pipeline_path = (
        output_folder
        / f"{task_name}_preprocessor.joblib"
    )

    joblib.dump(preprocessor, pipeline_path)

    # Save selected features for reproducibility

    feature_output_path = (
        output_folder
        / f"{task_name}_selected_features.txt"
    )

    feature_output_path.write_text(
        "\n".join(features),
        encoding="utf-8"
    )

    # --------------------------------------------------
    # 3.5.7 - Print task results
    # --------------------------------------------------

    print(
        "Training matrix shape:",
        X_train_prepared.shape
    )

    print(
        "Validation matrix shape:",
        X_validation_prepared.shape
    )

    print(
        "Prepared features:",
        X_train_prepared.shape[1]
    )

    print("Pipeline saved:", pipeline_path.name)

    summary_rows.append({
        "Task": task_name,
        "OriginalFeatures": len(features),
        "NumericalFeatures": len(numerical_columns),
        "CategoricalFeatures": len(categorical_columns),
        "PreparedFeatures": X_train_prepared.shape[1],
        "TrainingRows": X_train_prepared.shape[0],
        "ValidationRows": X_validation_prepared.shape[0]
    })

# --------------------------------------------------
# 3.5.8 - Save overall preprocessing summary
# --------------------------------------------------

summary_df = pd.DataFrame(summary_rows)

summary_path = (
    output_folder
    / "preprocessing_pipeline_summary.csv"
)

summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 60)
print("STAGE 3.5 - PIPELINES COMPLETE")
print("=" * 60)

print("\nSummary:")
print(summary_df.to_string(index=False))

print("\nPipelines saved: 3")
print("Original dataset modified: NO")
print("Test partition accessed: NO")
print("Models trained: NO")

print("\nAll preprocessing parameters were learned")
print("from the Training partition only.")
