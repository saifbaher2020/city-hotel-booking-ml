
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# STAGE 5.4 - VERIFIED MISSING-VALUE VISUALIZATION
# Distinguish NULL placeholders, NA markers, and empty cells
# ==========================================================

# ----------------------------------------------------------
# 1. Locate the corrected audit results
# ----------------------------------------------------------

project_folder = Path(__file__).resolve().parents[2]

audit_path = (
    project_folder
    / "outputs"
    / "audit"
    / "missing_value_semantic_audit.csv"
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

assert audit_path.is_file(), (
    "Corrected audit file not found. "
    "Run audit_missing_values.py first."
)

# ----------------------------------------------------------
# 2. Read the corrected audit
# ----------------------------------------------------------

audit = pd.read_csv(audit_path)

required_columns = [
    "Feature",
    "EmptyCells",
    "LiteralNULL",
    "LiteralNA",
    "TotalFlagged",
    "FlaggedPercentage",
]

assert all(
    column in audit.columns
    for column in required_columns
)

flagged = audit.loc[
    audit["TotalFlagged"] > 0
].copy()

flagged = flagged.sort_values(
    "TotalFlagged",
    ascending=False
)

assert set(flagged["Feature"]) == {
    "Company",
    "Agent",
    "Country",
    "Children",
}

assert int(flagged["TotalFlagged"].sum()) == 83800
assert int(flagged["LiteralNULL"].sum()) == 83796
assert int(flagged["LiteralNA"].sum()) == 4
assert int(flagged["EmptyCells"].sum()) == 0

print("=" * 65)
print("STAGE 5.4 - VERIFIED MISSING-VALUE VISUALIZATION")
print("=" * 65)

print("\nVerified flagged values:")
print(
    flagged[required_columns]
    .round(3)
    .to_string(index=False)
)

# ----------------------------------------------------------
# 3. Separate high- and low-frequency markers
# ----------------------------------------------------------

high = flagged.loc[
    flagged["Feature"].isin(["Company", "Agent"])
].copy()

low = flagged.loc[
    flagged["Feature"].isin(["Country", "Children"])
].copy()

fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 9),
    gridspec_kw={"height_ratios": [1, 1]},
)

colors = {
    "Company": "#C26A52",
    "Agent": "#D49B45",
    "Country": "#3766A8",
    "Children": "#218B78",
}

# ----------------------------------------------------------
# 4. Draw the visualization
# ----------------------------------------------------------

for ax, subset, axis_limit in [
    (axes[0], high, 110),
    (axes[1], low, 0.045),
]:

    bars = ax.barh(
        subset["Feature"],
        subset["FlaggedPercentage"],
        color=[
            colors[name]
            for name in subset["Feature"]
        ],
        height=0.55,
    )

    ax.invert_yaxis()
    ax.set_xlim(0, axis_limit)
    ax.set_xlabel("Percentage of dataset rows (%)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, (_, record) in zip(
        bars,
        subset.iterrows()
    ):

        count = int(record["TotalFlagged"])
        percentage = float(
            record["FlaggedPercentage"]
        )

        marker_types = []

        if int(record["LiteralNULL"]) > 0:
            marker_types.append("NULL")

        if int(record["LiteralNA"]) > 0:
            marker_types.append("NA")

        if int(record["EmptyCells"]) > 0:
            marker_types.append("Empty")

        marker_label = " + ".join(marker_types)

        ax.annotate(
            f"{count:,} {marker_label} ({percentage:.3f}%)",
            xy=(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
            ),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
        )

axes[0].set_title(
    "High-Frequency NULL Placeholders",
    fontsize=13,
    fontweight="bold",
)

axes[1].set_title(
    "Low-Frequency NULL and NA Markers (Zoomed)",
    fontsize=13,
    fontweight="bold",
)

fig.suptitle(
    "City Hotel Dataset: Verified Missing-Value Markers",
    fontsize=16,
    fontweight="bold",
)

fig.text(
    0.5,
    0.025,
    "83,800 flagged cells: 83,796 NULL markers "
    "and 4 NA markers. NULL in Agent/Company "
    "may mean 'not applicable'. "
    "Panels use different percentage scales.",
    ha="center",
    fontsize=9,
    color="#555555",
)

plt.tight_layout(rect=[0, 0.07, 1, 0.95])

# ----------------------------------------------------------
# 5. Save updated figure
# ----------------------------------------------------------

figure_path = (
    figure_folder
    / "01_missing_values_verified.png"
)

fig.savefig(
    figure_path,
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)

print("\nUpdated figure saved:")
print(figure_path)

# ----------------------------------------------------------
# 6. Save updated EDA table
# ----------------------------------------------------------

table_path = (
    table_folder
    / "missing_values_verified.csv"
)

flagged.to_csv(table_path, index=False)

print("\nUpdated EDA table saved:")
print(table_path)

# ----------------------------------------------------------
# 7. Final verification
# ----------------------------------------------------------

assert figure_path.is_file()
assert table_path.is_file()

print("\n" + "=" * 65)
print("STAGE 5.4 - VISUALIZATION COMPLETE")
print("=" * 65)

print("Updated figures generated: 1")
print("Updated tables generated: 1")
print("Total flagged cells: 83800")
print("Literal NULL markers: 83796")
print("Literal NA markers: 4")
print("Original H2.csv modified: NO")
print("Model pipelines modified: NO")
print("Models retrained: NO")
