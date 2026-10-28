
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# STAGE 5: MILESTONE 1 PROJECT AUDIT
# STEP 5.1.2 - VERIFIED MISSING-VALUE VISUALIZATION
# ==========================================================

# ----------------------------------------------------------
# 5.1.2.1 - Locate existing audit results
# ----------------------------------------------------------

project_folder = Path(__file__).resolve().parents[2]

audit_path = (
    project_folder
    / "outputs"
    / "audit"
    / "verified_missing_values.csv"
)

figure_folder = (
    project_folder
    / "outputs"
    / "eda"
    / "01_data_quality"
)

table_folder = (
    project_folder
    / "outputs"
    / "eda"
    / "tables"
)

figure_folder.mkdir(parents=True, exist_ok=True)
table_folder.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# 5.1.2.2 - Read verified missing-value statistics
# ----------------------------------------------------------

audit = pd.read_csv(audit_path)

required_columns = [
    "Feature",
    "PandasMissing",
    "TextNULL",
    "TotalMissing",
    "MissingPercentage"
]

assert all(
    column in audit.columns
    for column in required_columns
)

missing = audit.loc[
    audit["TotalMissing"] > 0
].copy()

missing = missing.sort_values(
    "TotalMissing",
    ascending=False
)

assert set(missing["Feature"]) == {
    "Company",
    "Agent",
    "Country",
    "Children"
}

assert int(missing["TotalMissing"].sum()) == 83800

print("=" * 60)
print("STAGE 5.1.2 - VERIFIED MISSING-VALUE VISUALIZATION")
print("=" * 60)

print("\nVerified missing features:")
print(missing.to_string(index=False))

# ----------------------------------------------------------
# 5.1.2.3 - Create a readable two-panel chart
# ----------------------------------------------------------

# Company and Agent have high missing percentages.
# Country and Children have much smaller percentages.
# Separate panels let us show both accurately.

high_missing = missing[
    missing["Feature"].isin(["Company", "Agent"])
].copy()

low_missing = missing[
    missing["Feature"].isin(["Country", "Children"])
].copy()

fig, axes = plt.subplots(
    2,
    1,
    figsize=(11, 8.5),
    gridspec_kw={"height_ratios": [1, 1]}
)

colors = {
    "Company": "#C26A52",
    "Agent": "#D49B45",
    "Country": "#3766A8",
    "Children": "#218B78"
}

for ax, subset, limit in [
    (axes[0], high_missing, 110),
    (axes[1], low_missing, 0.04)
]:

    bars = ax.barh(
        subset["Feature"],
        subset["MissingPercentage"],
        color=[
            colors[name]
            for name in subset["Feature"]
        ],
        height=0.55
    )

    ax.invert_yaxis()
    ax.set_xlim(0, limit)
    ax.set_xlabel("Missing Bookings (%)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="x",
        alpha=0.25
    )

    ax.set_axisbelow(True)

    for bar, (_, row) in zip(
        bars,
        subset.iterrows()
    ):

        percentage = row["MissingPercentage"]
        count = int(row["TotalMissing"])

        ax.annotate(
            f"{count:,} missing ({percentage:.3f}%)",
            xy=(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2
            ),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=10
        )

axes[0].set_title(
    "High-Missingness Features",
    fontsize=13,
    fontweight="bold"
)

axes[1].set_title(
    "Low-Missingness Features (Zoomed Scale)",
    fontsize=13,
    fontweight="bold"
)

fig.suptitle(
    "City Hotel Dataset: Verified Missing Values",
    fontsize=16,
    fontweight="bold"
)

fig.text(
    0.5,
    0.02,
    "Includes pandas missing values and text 'NULL' placeholders. "
    "Panels use different percentage scales.",
    ha="center",
    fontsize=9,
    color="#555555"
)

plt.tight_layout(rect=[0, 0.05, 1, 0.96])

# ----------------------------------------------------------
# 5.1.2.4 - Save corrected figure
# ----------------------------------------------------------

figure_path = (
    figure_folder
    / "01_missing_values_verified.png"
)

plt.savefig(
    figure_path,
    dpi=220,
    bbox_inches="tight"
)

plt.close()

print("\nVerified figure saved:")
print(figure_path)

# ----------------------------------------------------------
# 5.1.2.5 - Save corrected EDA table
# ----------------------------------------------------------

table_path = (
    table_folder
    / "missing_values_verified.csv"
)

missing.to_csv(
    table_path,
    index=False
)

print("\nVerified EDA table saved:")
print(table_path)

# ----------------------------------------------------------
# 5.1.2.6 - Final verification
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STAGE 5.1.2 - VISUALIZATION COMPLETE")
print("=" * 60)

print("Corrected figures generated: 1")
print("Corrected tables generated: 1")
print("Missing cells documented: 83800")
print("Original H2.csv modified: NO")
print("Existing models modified: NO")
print("Existing EDA graphs overwritten: NO")
