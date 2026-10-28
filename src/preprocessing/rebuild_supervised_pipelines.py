
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

# ==========================================================
# STAGE 5: MILESTONE 1 PROJECT AUDIT
# STEP 5.1.5 - REBUILD SUPERVISED PREPROCESSING
# ==========================================================

# ----------------------------------------------------------
# 5.1.5.1 - Locate project files
# ----------------------------------------------------------

project_folder = Path(__file__).resolve().parents[2]

dataset_path = project_folder / "data" / "H2.csv"

splits_path = (
    project_folder
    / "outputs"
    / "splits"
    / "split_assignments.csv"
)

outputs_folder = project_folder / "outputs"

pipeline_folder = (
    outputs_folder
    / "preprocessing"
    / "pipelines"
)

backup_folder = (
    outputs_folder
    / "preprocessing"
    / "pipelines_before_leakage_review"
)

summary_path = (
    pipeline_folder
    / "preprocessing_pipeline_summary.csv"
)

# ----------------------------------------------------------
# 5.1.5.2 - Verify backups before modifying pipelines
# ----------------------------------------------------------

required_backups = [
    backup_folder / "regression_preprocessor.joblib",
    backup_folder / "classification_preprocessor.joblib",
    outputs_folder
    / "regression_features_before_leakage_review.txt",
    outputs_folder
    / "classification_features_before_leakage_review.txt"
]

for path in required_backups:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required backup missing: {path}"
        )

print("Old pipelines and feature lists backed up: YES")

# ----------------------------------------------------------
# 5.1.5.3 - Load dataset and saved split assignments
# ----------------------------------------------------------

df = pd.read_csv(dataset_path)

splits = pd.read_csv(splits_path)

assert len(df) == len(splits)

assert splits["OriginalRowIndex"].tolist() == list(
    range(len(df))
)

assert set(splits["Partition"]) == {
    "Train",
    "Validation",
    "Test"
}

train_mask = splits["Partition"] == "Train"

validation_mask = (
    splits["Partition"] == "Validation"
)

print("\n" + "=" * 60)
print("STAGE 5.1.5 - REBUILD SUPERVISED PIPELINES")
print("=" * 60)

print("\nTraining rows:", int(train_mask.sum()))
print(
    "Validation rows:",
    int(validation_mask.sum())
)

# ----------------------------------------------------------
# 5.1.5.4 - Define reusable preprocessing
# ----------------------------------------------------------

def create_preprocessor(
    numerical_columns,
    categorical_columns
):

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

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

# ----------------------------------------------------------
# 5.1.5.5 - Define affected tasks only
# ----------------------------------------------------------

tasks = {
    "regression": "ADR",
    "classification": "IsCanceled"
}

new_summary_rows = []

prepared_pipelines = {}

# ----------------------------------------------------------
# 5.1.5.6 - Rebuild each supervised pipeline
# ----------------------------------------------------------

for task_name, target in tasks.items():

    print("\n" + "-" * 60)
    print("TASK:", task_name.upper())
    print("-" * 60)

    features_path = (
        outputs_folder
        / f"{task_name}_features.txt"
    )

    features = [
        line.strip()
        for line in features_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    # Verify updated feature selection.

    assert len(features) == 19
    assert len(features) == len(set(features))

    assert "DepositType" not in features
    assert target not in features

    assert "ReservationStatus" not in features
    assert "ReservationStatusDate" not in features

    if not set(features).issubset(df.columns):
        raise ValueError(
            f"Unknown feature in {task_name} list."
        )

    # Separate Training and Validation.

    X_train = df.loc[
        train_mask,
        features
    ].copy()

    X_validation = df.loc[
        validation_mask,
        features
    ].copy()

    # Identify numerical and categorical features.

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

    # Create a fresh preprocessing pipeline.

    preprocessor = create_preprocessor(
        numerical_columns,
        categorical_columns
    )

    # Fit using Training data ONLY.

    X_train_prepared = preprocessor.fit_transform(
        X_train
    )

    # Transform Validation using Training rules.

    X_validation_prepared = preprocessor.transform(
        X_validation
    )

    # Verify both partitions have compatible shapes.

    assert X_train_prepared.shape[0] == len(X_train)
    assert X_validation_prepared.shape[0] == len(
        X_validation
    )

    assert (
        X_train_prepared.shape[1]
        == X_validation_prepared.shape[1]
    )

    # Verify no missing or infinite values remain.

    def has_only_finite_values(matrix):

        if sparse.issparse(matrix):
            return np.isfinite(matrix.data).all()

        return np.isfinite(matrix).all()

    assert has_only_finite_values(
        X_train_prepared
    )

    assert has_only_finite_values(
        X_validation_prepared
    )

    # Verify DepositType did not survive encoding.

    prepared_names = (
        preprocessor.get_feature_names_out()
    )

    assert not any(
        "DepositType" in name
        for name in prepared_names
    )

    # Store the fitted pipeline until all checks pass.

    prepared_pipelines[task_name] = (
        preprocessor,
        features
    )

    print(
        "Prepared Training shape:",
        X_train_prepared.shape
    )

    print(
        "Prepared Validation shape:",
        X_validation_prepared.shape
    )

    print("DepositType excluded: YES")
    print("Verification: PASSED")

    new_summary_rows.append({
        "Task": task_name,
        "OriginalFeatures": len(features),
        "NumericalFeatures": len(numerical_columns),
        "CategoricalFeatures": len(categorical_columns),
        "PreparedFeatures": X_train_prepared.shape[1],
        "TrainingRows": X_train_prepared.shape[0],
        "ValidationRows": X_validation_prepared.shape[0]
    })

# ----------------------------------------------------------
# 5.1.5.7 - Save updated supervised pipelines
# ----------------------------------------------------------

for task_name, (preprocessor, features) in (
    prepared_pipelines.items()
):

    model_path = (
        pipeline_folder
        / f"{task_name}_preprocessor.joblib"
    )

    feature_output_path = (
        pipeline_folder
        / f"{task_name}_selected_features.txt"
    )

    joblib.dump(preprocessor, model_path)

    feature_output_path.write_text(
        "\n".join(features) + "\n",
        encoding="utf-8"
    )

    print("\nSaved:", model_path.name)

# ----------------------------------------------------------
# 5.1.5.8 - Update the preprocessing summary
# ----------------------------------------------------------

old_summary = pd.read_csv(summary_path)

assert set(old_summary["Task"]) == {
    "regression",
    "classification",
    "clustering"
}

# Preserve the original clustering summary row.

clustering_row = old_summary.loc[
    old_summary["Task"] == "clustering"
].copy()

assert len(clustering_row) == 1

assert int(
    clustering_row.iloc[0]["PreparedFeatures"]
) == 30

new_summary = pd.concat(
    [
        pd.DataFrame(new_summary_rows),
        clustering_row
    ],
    ignore_index=True
)

new_summary.to_csv(
    summary_path,
    index=False
)

# ----------------------------------------------------------
# 5.1.5.9 - Final results
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STAGE 5.1.5 - PIPELINES REBUILT SUCCESSFULLY")
print("=" * 60)

print("\nUpdated preprocessing summary:")
print(new_summary.to_string(index=False))

print("\nSupervised pipelines rebuilt: 2")
print("Clustering pipeline changed: NO")
print("Prediction models retrained: NO")
print("Original H2.csv modified: NO")
print("Test partition selected for processing: NO")

print("\nNEXT: Retrain Regression and Classification.")
