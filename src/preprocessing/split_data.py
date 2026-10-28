
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# ==================================================
# STAGE 3: DATA CLEANING AND PREPROCESSING
# STEP 3.1 - REPRODUCIBLE DATA SPLITTING
# ==================================================

# --------------------------------------------------
# 3.1.1 - Locate project files
# --------------------------------------------------

project_folder = Path(__file__).resolve().parents[2]

dataset_path = project_folder / "data" / "H2.csv"
output_folder = project_folder / "outputs" / "splits"

output_folder.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 3.1.2 - Load the original dataset
# --------------------------------------------------

df = pd.read_csv(dataset_path)

print("Original dataset loaded!")
print("Total bookings:", len(df))

# --------------------------------------------------
# 3.1.3 - Identify identical full rows
# --------------------------------------------------

# Assign the same group ID to rows whose
# values match across all 31 original columns.
#
# We KEEP the rows. We only group them.

row_groups = pd.util.hash_pandas_object(
    df,
    index=False
).to_numpy()

print("Exact duplicate rows:", df.duplicated().sum())

# --------------------------------------------------
# 3.1.4 - Create a reproducible grouped split
# --------------------------------------------------

# y is used here ONLY to balance the split.
# This does not train a prediction model.

y = df["IsCanceled"].to_numpy()

# First split:
# Approximately 15% test and 85% development.
#
# StratifiedGroupKFold creates grouped folds.
# With 7 folds, one fold is approximately 14.3%.

first_split = StratifiedGroupKFold(
    n_splits=7,
    shuffle=True,
    random_state=42
)

development_idx, test_idx = next(
    first_split.split(df, y, groups=row_groups)
)

# Second split:
# Divide development into training
# and validation, roughly 5:1.
#
# This gives an approximate final ratio
# of 71.4% / 14.3% / 14.3%.

development_df = df.iloc[development_idx]
development_groups = row_groups[development_idx]
development_y = y[development_idx]

second_split = StratifiedGroupKFold(
    n_splits=6,
    shuffle=True,
    random_state=43
)

train_relative, val_relative = next(
    second_split.split(
        development_df,
        development_y,
        groups=development_groups
    )
)

train_idx = development_idx[train_relative]
val_idx = development_idx[val_relative]

# --------------------------------------------------
# 3.1.5 - Verify that groups do not overlap
# --------------------------------------------------

train_groups = set(row_groups[train_idx])
val_groups = set(row_groups[val_idx])
test_groups = set(row_groups[test_idx])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)

# --------------------------------------------------
# 3.1.6 - Save row assignments
# --------------------------------------------------

assignments = pd.DataFrame({
    "OriginalRowIndex": np.arange(len(df)),
    "Partition": "Unassigned"
})

assignments.loc[train_idx, "Partition"] = "Train"
assignments.loc[val_idx, "Partition"] = "Validation"
assignments.loc[test_idx, "Partition"] = "Test"

assert not (assignments["Partition"] == "Unassigned").any()

assignment_path = output_folder / "split_assignments.csv"

assignments.to_csv(assignment_path, index=False)

# --------------------------------------------------
# 3.1.7 - Display split statistics
# --------------------------------------------------

print("\n" + "=" * 55)
print("STAGE 3.1 - DATA SPLIT RESULTS")
print("=" * 55)

for name, indices in [
    ("Training", train_idx),
    ("Validation", val_idx),
    ("Test", test_idx)
]:
    subset = df.iloc[indices]

    print(f"\n{name}:")
    print("Bookings:", len(subset))
    print("Percentage:", round(len(subset) / len(df) * 100, 2))
    print(
        "Cancellation rate:",
        round(subset["IsCanceled"].mean() * 100, 2),
        "%"
    )

print("\nIdentical-row groups overlap: NO")
print("Original dataset modified: NO")
print("Saved assignments:", assignment_path)

print("\nIMPORTANT:")
print("No ML model was trained.")
print("No cleaning or encoding was performed.")
print("The test set is reserved for final evaluation.")
