
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer

# ==================================================
# STAGE 3: DATA CLEANING AND PREPROCESSING
# STEP 3.2 - MISSING VALUE HANDLING
# ==================================================

# --------------------------------------------------
# 3.2.1 - Locate our project files
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
)

output_folder.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 3.2.2 - Load the original dataset and split labels
# --------------------------------------------------

df = pd.read_csv(dataset_path)
assignments = pd.read_csv(split_path)

assert len(df) == len(assignments)
assert assignments["OriginalRowIndex"].tolist() == list(range(len(df)))

train_mask = assignments["Partition"] == "Train"
validation_mask = assignments["Partition"] == "Validation"

train_df = df.loc[train_mask].copy()
validation_df = df.loc[validation_mask].copy()

print("=" * 55)
print("STAGE 3.2 - MISSING VALUE HANDLING")
print("=" * 55)

print("\nMissing values before imputation:")

for name, subset in [
    ("Training", train_df),
    ("Validation", validation_df),
]:
    print(f"\n{name}:")
    print(subset[["Children", "Country"]].isna().sum())

# --------------------------------------------------
# 3.2.3 - Learn Children replacement from Training
# --------------------------------------------------

children_imputer = SimpleImputer(
    strategy="most_frequent"
)

train_df[["Children"]] = children_imputer.fit_transform(
    train_df[["Children"]]
)

validation_df[["Children"]] = children_imputer.transform(
    validation_df[["Children"]]
)

print("\nChildren replacement learned from Training:")
print(children_imputer.statistics_[0])

# --------------------------------------------------
# 3.2.4 - Replace missing Country with Unknown
# --------------------------------------------------

country_imputer = SimpleImputer(
    strategy="constant",
    fill_value="Unknown"
)

train_df[["Country"]] = country_imputer.fit_transform(
    train_df[["Country"]]
)

validation_df[["Country"]] = country_imputer.transform(
    validation_df[["Country"]]
)

# --------------------------------------------------
# 3.2.5 - Verify the results
# --------------------------------------------------

print("\nMissing values AFTER imputation:")

for name, subset in [
    ("Training", train_df),
    ("Validation", validation_df),
]:
    print(f"\n{name}:")
    print(subset[["Children", "Country"]].isna().sum())

    assert subset["Children"].isna().sum() == 0
    assert subset["Country"].isna().sum() == 0

# --------------------------------------------------
# 3.2.6 - Save a summary of our strategy
# --------------------------------------------------

summary = pd.DataFrame([
    {
        "Feature": "Children",
        "Strategy": "Most frequent (Training only)",
        "Replacement": children_imputer.statistics_[0],
    },
    {
        "Feature": "Country",
        "Strategy": "Constant",
        "Replacement": "Unknown",
    }
])

summary_path = (
    output_folder
    / "missing_value_strategy.csv"
)

summary.to_csv(summary_path, index=False)

print("\nImputation strategy saved:")
print(summary_path)

# --------------------------------------------------
# 3.2.7 - Final notes
# --------------------------------------------------

print("\nIMPORTANT:")
print("Only Training was used to learn replacement values.")
print("Validation was transformed using Training rules.")
print("The Test partition was not accessed.")
print("The original H2.csv was not modified.")
print("The final reusable ML pipeline is not built yet.")
