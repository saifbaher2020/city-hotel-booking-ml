
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ==================================================
# STAGE 3: DATA CLEANING AND PREPROCESSING
# STEP 3.4 - NUMERICAL FEATURE SCALING
# ==================================================

# --------------------------------------------------
# 3.4.1 - Locate project files
# --------------------------------------------------

project_folder = Path(__file__).resolve().parents[2]

dataset_path = project_folder / "data" / "H2.csv"

split_path = (
    project_folder
    / "outputs"
    / "splits"
    / "split_assignments.csv"
)

features_path = (
    project_folder
    / "outputs"
    / "regression_features.txt"
)

output_folder = (
    project_folder
    / "outputs"
    / "preprocessing"
)

output_folder.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 3.4.2 - Load data and feature list
# --------------------------------------------------

df = pd.read_csv(dataset_path)
assignments = pd.read_csv(split_path)

assert len(df) == len(assignments)
assert assignments["OriginalRowIndex"].tolist() == list(range(len(df)))

features = [
    line.strip()
    for line in features_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

train_mask = assignments["Partition"] == "Train"
validation_mask = assignments["Partition"] == "Validation"

train_df = df.loc[train_mask, features].copy()
validation_df = df.loc[validation_mask, features].copy()

# --------------------------------------------------
# 3.4.3 - Identify numerical columns
# --------------------------------------------------

numerical_columns = train_df.select_dtypes(
    include="number"
).columns.tolist()

print("=" * 55)
print("STAGE 3.4 - NUMERICAL SCALING")
print("=" * 55)

print("\nNumerical columns:")
print(numerical_columns)

# --------------------------------------------------
# 3.4.4 - Handle missing numerical values
# --------------------------------------------------

imputer = SimpleImputer(strategy="most_frequent")

train_numbers = imputer.fit_transform(
    train_df[numerical_columns]
)

validation_numbers = imputer.transform(
    validation_df[numerical_columns]
)

# --------------------------------------------------
# 3.4.5 - Apply StandardScaler
# --------------------------------------------------

scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_numbers)
validation_scaled = scaler.transform(validation_numbers)

# --------------------------------------------------
# 3.4.6 - Create scaled DataFrames
# --------------------------------------------------

train_scaled_df = pd.DataFrame(
    train_scaled,
    columns=numerical_columns,
    index=train_df.index
)

validation_scaled_df = pd.DataFrame(
    validation_scaled,
    columns=numerical_columns,
    index=validation_df.index
)

# --------------------------------------------------
# 3.4.7 - Show scaling results
# --------------------------------------------------

print("\nBefore scaling - first 5 Training rows:")
print(
    train_df[numerical_columns]
    .head()
    .to_string()
)

print("\nAfter scaling - first 5 Training rows:")
print(
    train_scaled_df
    .head()
    .round(3)
    .to_string()
)

print("\nTraining means after scaling:")
print(train_scaled_df.mean().round(3))

print("\nTraining standard deviations after scaling:")
print(train_scaled_df.std(ddof=0).round(3))

# --------------------------------------------------
# 3.4.8 - Save scaling statistics
# --------------------------------------------------

summary = pd.DataFrame({
    "Feature": numerical_columns,
    "TrainingMean": scaler.mean_,
    "TrainingScale": scaler.scale_,
})

summary_path = (
    output_folder
    / "numerical_scaling_summary.csv"
)

summary.to_csv(summary_path, index=False)

print("\nScaling summary saved:", summary_path)

# --------------------------------------------------
# 3.4.9 - Final verification
# --------------------------------------------------

assert train_scaled_df.isna().sum().sum() == 0
assert validation_scaled_df.isna().sum().sum() == 0

print("\nIMPORTANT:")
print("Scaler fitted on Training only.")
print("Validation transformed using Training parameters.")
print("Test partition was not accessed.")
print("Original H2.csv modified: NO")
print("Models trained: NO")
