
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# ==================================================
# STAGE 3: DATA CLEANING AND PREPROCESSING
# STEP 3.3 - CATEGORICAL ENCODING
# ==================================================

# --------------------------------------------------
# 3.3.1 - Locate project files
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
# 3.3.2 - Load data and feature selections
# --------------------------------------------------

df = pd.read_csv(dataset_path)
assignments = pd.read_csv(split_path)

features = [
    line.strip()
    for line in features_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

assert len(df) == len(assignments)
assert assignments["OriginalRowIndex"].tolist() == list(range(len(df)))

train_mask = assignments["Partition"] == "Train"
validation_mask = assignments["Partition"] == "Validation"

train_df = df.loc[train_mask, features].copy()
validation_df = df.loc[validation_mask, features].copy()

# --------------------------------------------------
# 3.3.3 - Identify categorical features
# --------------------------------------------------

categorical_columns = train_df.select_dtypes(
    include=["object", "str", "string", "category"]
).columns.tolist()

print("=" * 55)
print("STAGE 3.3 - CATEGORICAL ENCODING")
print("=" * 55)

print("\nCategorical columns:")
print(categorical_columns)

# --------------------------------------------------
# 3.3.4 - Prepare missing categories
# --------------------------------------------------

# Do not learn missing-value replacements
# from Validation or Test.

train_categories = train_df[
    categorical_columns
].fillna("Unknown")

validation_categories = validation_df[
    categorical_columns
].fillna("Unknown")

# --------------------------------------------------
# 3.3.5 - Create the One-Hot Encoder
# --------------------------------------------------

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
    dtype=int
)

# Learn categories from Training ONLY.
train_encoded = encoder.fit_transform(
    train_categories
)

# Apply the SAME encoder to Validation.
validation_encoded = encoder.transform(
    validation_categories
)

# --------------------------------------------------
# 3.3.6 - Create encoded DataFrames
# --------------------------------------------------

encoded_column_names = encoder.get_feature_names_out(
    categorical_columns
)

train_encoded_df = pd.DataFrame(
    train_encoded,
    columns=encoded_column_names,
    index=train_df.index
)

validation_encoded_df = pd.DataFrame(
    validation_encoded,
    columns=encoded_column_names,
    index=validation_df.index
)

# --------------------------------------------------
# 3.3.7 - Display encoding results
# --------------------------------------------------

print("\nOriginal categorical feature count:")
print(len(categorical_columns))

print("\nEncoded feature count:")
print(len(encoded_column_names))

print("\nFirst 10 encoded feature names:")
print(list(encoded_column_names[:10]))

print("\nTraining encoded shape:")
print(train_encoded_df.shape)

print("\nValidation encoded shape:")
print(validation_encoded_df.shape)

print("\nExample original categorical row:")
print(train_categories.head(1).to_string(index=False))

print("\nExample encoded row:")
print(train_encoded_df.head(1).to_string(index=False))

# --------------------------------------------------
# 3.3.8 - Save encoding summary
# --------------------------------------------------

summary = pd.DataFrame({
    "OriginalFeature": [
        name.split("_")[0]
        for name in encoded_column_names
    ],
    "EncodedFeature": encoded_column_names
})

summary_path = (
    output_folder
    / "categorical_encoding_summary.csv"
)

summary.to_csv(summary_path, index=False)

print("\nEncoding summary saved:")
print(summary_path)

# --------------------------------------------------
# 3.3.9 - Final notes
# --------------------------------------------------

print("\nIMPORTANT:")
print("Categories learned from Training only.")
print("Validation transformed using the same encoder.")
print("Test partition was not accessed.")
print("Original H2.csv was not modified.")
print("The final preprocessing pipeline is not built yet.")
    