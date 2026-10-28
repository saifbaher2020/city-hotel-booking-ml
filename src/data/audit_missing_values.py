
from pathlib import Path

import pandas as pd

# ==========================================================
# STAGE 5: MILESTONE 1 PROJECT AUDIT
# STEP 5.1 - DATA QUALITY VERIFICATION
# ==========================================================

# ----------------------------------------------------------
# 5.1.1 - Locate project files
# ----------------------------------------------------------

project_folder = Path(__file__).resolve().parents[2]

dataset_path = project_folder / "data" / "H2.csv"

output_folder = project_folder / "outputs" / "audit"

output_folder.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# 5.1.2 - Load original dataset without changing it
# ----------------------------------------------------------

df = pd.read_csv(dataset_path)

print("=" * 60)
print("STAGE 5.1 - MISSING VALUE AUDIT")
print("=" * 60)

print("\nDataset rows:", len(df))
print("Dataset columns:", len(df.columns))

# ----------------------------------------------------------
# 5.1.3 - Count missing values recognized by pandas
# ----------------------------------------------------------

standard_missing = df.isna().sum()

# ----------------------------------------------------------
# 5.1.4 - Detect the text placeholder "NULL"
# ----------------------------------------------------------

null_text_counts = pd.Series(
    0,
    index=df.columns,
    dtype=int
)

for column in df.select_dtypes(
    include=["object", "string"]
).columns:

    values = df[column].astype("string")

    null_text_counts[column] = (
        values.str.strip()
        .str.upper()
        .eq("NULL")
        .fillna(False)
        .sum()
    )

# ----------------------------------------------------------
# 5.1.5 - Combine the two types of missing values
# ----------------------------------------------------------

total_missing = (
    standard_missing + null_text_counts
)

summary = pd.DataFrame({
    "Feature": df.columns,
    "PandasMissing": standard_missing.values,
    "TextNULL": null_text_counts.values,
    "TotalMissing": total_missing.values
})

summary["MissingPercentage"] = (
    summary["TotalMissing"] / len(df) * 100
)

summary = summary.sort_values(
    "TotalMissing",
    ascending=False
)

# ----------------------------------------------------------
# 5.1.6 - Display results
# ----------------------------------------------------------

affected_features = summary[
    summary["TotalMissing"] > 0
]

print("\nFeatures containing missing values:")
print(
    affected_features.round(2).to_string(
        index=False
    )
)

print("\nTotal missing cells:")
print(int(summary["TotalMissing"].sum()))

print("\nExact duplicate rows:")
print(int(df.duplicated().sum()))

# ----------------------------------------------------------
# 5.1.7 - Save verified audit results
# ----------------------------------------------------------

summary_path = (
    output_folder
    / "verified_missing_values.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print("\nAudit table saved:")
print(summary_path)

# ----------------------------------------------------------
# 5.1.8 - Final checks
# ----------------------------------------------------------

assert len(df) == 79330
assert len(df.columns) == 31

print("\n" + "=" * 60)
print("STAGE 5.1.1 - DATA QUALITY AUDIT COMPLETE")
print("=" * 60)

print("Original H2.csv modified: NO")
print("Missing values automatically replaced: NO")
print("Duplicate rows removed: NO")
print("Models retrained: NO")
print("Existing experiment results changed: NO")
