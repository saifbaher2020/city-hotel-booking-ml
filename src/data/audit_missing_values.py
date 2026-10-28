
from pathlib import Path

import pandas as pd

# ==========================================================
# STAGE 5.4 - FINAL MISSING VALUE AUDIT
# Identify literal NULL, NA, and genuinely empty cells
# ==========================================================

# ----------------------------------------------------------
# 1. Locate project files
# ----------------------------------------------------------

project_folder = Path(__file__).resolve().parents[2]

dataset_path = project_folder / "data" / "H2.csv"

output_folder = project_folder / "outputs" / "audit"
output_folder.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# 2. Read raw text without automatic missing-value parsing
# ----------------------------------------------------------

df_raw = pd.read_csv(
    dataset_path,
    dtype=str,
    keep_default_na=False,
    na_filter=False
)

print("=" * 65)
print("STAGE 5.4 - VERIFIED MISSING VALUE AUDIT")
print("=" * 65)

print("\nDataset rows:", len(df_raw))
print("Dataset columns:", len(df_raw.columns))

assert df_raw.shape == (79330, 31)

# ----------------------------------------------------------
# 3. Count each missing-value marker separately
# ----------------------------------------------------------

summary_rows = []

for column in df_raw.columns:

    values = df_raw[column].str.strip()
    normalized = values.str.upper()

    empty_count = int(values.eq("").sum())
    null_count = int(normalized.eq("NULL").sum())
    na_count = int(normalized.eq("NA").sum())

    total_flagged = (
        empty_count + null_count + na_count
    )

    summary_rows.append({
        "Feature": column,
        "EmptyCells": empty_count,
        "LiteralNULL": null_count,
        "LiteralNA": na_count,
        "TotalFlagged": total_flagged,
        "FlaggedPercentage": (
            total_flagged / len(df_raw) * 100
        )
    })

summary = pd.DataFrame(summary_rows)

summary = summary.sort_values(
    "TotalFlagged",
    ascending=False
)

# ----------------------------------------------------------
# 4. Document interpretations
# ----------------------------------------------------------

interpretations = {
    "Company": (
        "NULL may represent no associated company. "
        "Excluded from selected model features."
    ),
    "Agent": (
        "NULL may represent no associated agent. "
        "Excluded from selected model features."
    ),
    "Country": (
        "Country is unspecified (literal NULL). "
        "Categorical imputation is applied when modeling."
    ),
    "Children": (
        "Child count is unspecified (literal NA). "
        "Numerical imputation is applied when modeling."
    )
}

summary["Interpretation"] = (
    summary["Feature"]
    .map(interpretations)
    .fillna("No flagged values identified.")
)

# ----------------------------------------------------------
# 5. Display affected columns
# ----------------------------------------------------------

affected = summary[
    summary["TotalFlagged"] > 0
]

print("\nAffected columns:")
print(
    affected[
        [
            "Feature",
            "EmptyCells",
            "LiteralNULL",
            "LiteralNA",
            "TotalFlagged",
            "FlaggedPercentage"
        ]
    ].round(2).to_string(index=False)
)

total_empty = int(summary["EmptyCells"].sum())
total_null = int(summary["LiteralNULL"].sum())
total_na = int(summary["LiteralNA"].sum())
total_flagged = int(summary["TotalFlagged"].sum())

print("\n" + "-" * 65)
print("SUMMARY")
print("-" * 65)

print("Empty cells:", total_empty)
print("Literal NULL cells:", total_null)
print("Literal NA cells:", total_na)
print("Total flagged cells:", total_flagged)

# ----------------------------------------------------------
# 6. Verify the observed dataset counts
# ----------------------------------------------------------

expected_counts = {
    "Company": (0, 75641, 0),
    "Agent": (0, 8131, 0),
    "Country": (0, 24, 0),
    "Children": (0, 0, 4),
}

indexed = summary.set_index("Feature")

for feature, expected in expected_counts.items():

    actual = (
        int(indexed.loc[feature, "EmptyCells"]),
        int(indexed.loc[feature, "LiteralNULL"]),
        int(indexed.loc[feature, "LiteralNA"])
    )

    assert actual == expected, (
        f"{feature}: expected {expected}, got {actual}"
    )

assert total_empty == 0
assert total_null == 83796
assert total_na == 4
assert total_flagged == 83800

assert set(affected["Feature"]) == set(
    expected_counts
)

print("\nAll expected missing-value markers: VERIFIED")

# ----------------------------------------------------------
# 7. Compare with normal pandas missing-value handling
# ----------------------------------------------------------

df_default = pd.read_csv(dataset_path)

pandas_missing = df_default.isna().sum()

print("\nMissing values under default pandas parsing:")

for feature in expected_counts:
    print(
        f"{feature}: {int(pandas_missing[feature])}"
    )

assert int(pandas_missing.sum()) == 28

print("\nPandas missing-value count: VERIFIED")
print(
    "Explanation: Agent and Company contain leading spaces "
    "before NULL, so pandas does not automatically classify "
    "those entries as missing."
)

# ----------------------------------------------------------
# 8. Confirm the relevant selected model features
# ----------------------------------------------------------

for task in (
    "regression",
    "classification",
    "clustering"
):

    features_file = (
        project_folder
        / "outputs"
        / f"{task}_features.txt"
    )

    features = [
        line.strip()
        for line in features_file.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert "Agent" not in features
    assert "Company" not in features

    print(
        f"{task.title()}: "
        "Agent and Company excluded."
    )

# ----------------------------------------------------------
# 9. Save corrected audit
# ----------------------------------------------------------

summary_path = (
    output_folder
    / "missing_value_semantic_audit.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print("\nCorrected audit saved:")
print(summary_path)

# ----------------------------------------------------------
# 10. Final status
# ----------------------------------------------------------

print("\n" + "=" * 65)
print("STAGE 5.4 - MISSING VALUE AUDIT COMPLETE")
print("=" * 65)

print("Total flagged cells:", total_flagged)
print("Original H2.csv modified: NO")
print("Pipelines modified: NO")
print("Models retrained: NO")
print("Model metrics changed: NO")
print("Test predictions generated: NO")

print(
    "\nNOTE: The meaning of NULL in Agent and "
    "Company is a documented possibility, "
    "not a proven interpretation."
)
