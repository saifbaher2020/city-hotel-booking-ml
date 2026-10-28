
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# 1. PROJECT SETUP
# ==================================================

PROJECT = Path(__file__).resolve().parents[2]
DATASET = PROJECT / "data" / "H2.csv"
OUTPUT = PROJECT / "outputs" / "eda"

FOLDERS = {
    "quality": OUTPUT / "01_data_quality",
    "pricing": OUTPUT / "02_pricing",
    "behavior": OUTPUT / "03_booking_behavior",
    "cancellations": OUTPUT / "04_cancellations",
    "seasonality": OUTPUT / "05_seasonality",
    "relationships": OUTPUT / "06_relationships",
}

for folder in FOLDERS.values():
    folder.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")

BLUE = "#2563A6"
TEAL = "#159895"

# ==================================================
# 2. LOAD ORIGINAL DATA
# ==================================================

df = pd.read_csv(DATASET)

print("=" * 55)
print("PROFESSIONAL EDA - CITY HOTEL BOOKINGS")
print("=" * 55)
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Missing cells:", int(df.isna().sum().sum()))
print("Exact duplicate rows:", int(df.duplicated().sum()))

generated = []


def save_chart(category, filename):
    """Save the current figure and close it."""
    path = FOLDERS[category] / filename
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    generated.append(path)
    print("Saved:", path.relative_to(PROJECT))


def make_figure(width=10, height=6):
    plt.figure(figsize=(width, height))


# ==================================================
# 3. DATA QUALITY
# ==================================================

# GRAPH 01 - Missing values

missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

make_figure(9, 5)

if not missing.empty:
    ax = sns.barplot(
        x=missing.index,
        y=missing.values,
        color=BLUE
    )

    for i, value in enumerate(missing.values):
        ax.text(i, value, str(value), ha="center", va="bottom")

    plt.ylabel("Number of Missing Values")
    plt.xlabel("Feature")
    plt.ylim(0, max(missing.values) * 1.2)
else:
    plt.text(
        0.5, 0.5, "No missing values",
        ha="center", va="center", fontsize=16
    )
    plt.xticks([])
    plt.yticks([])

plt.title("01 - Missing Values by Feature")
save_chart("quality", "01_missing_values.png")


# ==================================================
# 4. PRICING ANALYSIS
# ==================================================

# GRAPH 02 - ADR histogram, common range

adr_visible = df.loc[df["ADR"].between(0, 300), "ADR"]

make_figure()

sns.histplot(
    adr_visible,
    bins=50,
    color=BLUE
)

plt.title("02 - ADR Distribution (0 to 300)")
plt.xlabel("Average Daily Rate (ADR)")
plt.ylabel("Number of Bookings")
save_chart("pricing", "02_adr_distribution.png")


# GRAPH 03 - ADR boxplot, common range

make_figure(10, 4)

sns.boxplot(
    x=adr_visible,
    color=TEAL
)

plt.title("03 - ADR Boxplot (0 to 300)")
plt.xlabel("Average Daily Rate (ADR)")
save_chart("pricing", "03_adr_boxplot.png")


# ==================================================
# 5. BOOKING BEHAVIOR
# ==================================================

# GRAPH 04 - Lead time distribution

make_figure()

sns.histplot(
    df["LeadTime"],
    bins=50,
    color=BLUE
)

plt.title("04 - Booking Lead Time Distribution")
plt.xlabel("Days Before Arrival")
plt.ylabel("Number of Bookings")
save_chart("behavior", "04_lead_time.png")


# GRAPH 05 - Length of stay

stay_length = (
    df["StaysInWeekendNights"]
    + df["StaysInWeekNights"]
)

make_figure()

sns.histplot(
    stay_length[stay_length.between(0, 21)],
    bins=22,
    color=TEAL
)

plt.title("05 - Length of Stay Distribution (0 to 21 Nights)")
plt.xlabel("Total Nights")
plt.ylabel("Number of Bookings")
save_chart("behavior", "05_length_of_stay.png")


# ==================================================
# 6. CANCELLATION ANALYSIS
# ==================================================

# GRAPH 06 - Overall cancellation distribution

cancel_counts = (
    df["IsCanceled"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
)

make_figure(8, 6)

ax = sns.barplot(
    x=["Not Canceled", "Canceled"],
    y=cancel_counts.values,
    color=BLUE
)

for i, count in enumerate(cancel_counts.values):
    ax.text(
        i, count,
        f"{count:,}",
        ha="center", va="bottom"
    )

plt.ylim(0, cancel_counts.max() * 1.15)
plt.title("06 - Cancellation Distribution")
plt.xlabel("Booking Outcome")
plt.ylabel("Number of Bookings")
save_chart("cancellations", "06_cancellation_counts.png")


# GRAPH 07 - Cancellation rate by market segment

segment_stats = (
    df.groupby("MarketSegment")["IsCanceled"]
    .agg(["size", "mean"])
    .rename(columns={
        "size": "Bookings",
        "mean": "CancellationRate"
    })
)

segment_stats["CancellationRate"] *= 100
segment_stats = segment_stats.sort_values(
    "CancellationRate", ascending=False
)

make_figure(12, 6)

ax = sns.barplot(
    x=segment_stats.index,
    y=segment_stats["CancellationRate"],
    color=BLUE
)

for i, (_, row) in enumerate(segment_stats.iterrows()):
    ax.text(
        i,
        row["CancellationRate"],
        f'{row["CancellationRate"]:.1f}%\n(n={int(row["Bookings"]):,})',
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.ylim(0, max(110, segment_stats["CancellationRate"].max() * 1.18))
plt.title("07 - Cancellation Rate by Market Segment")
plt.xlabel("Market Segment")
plt.ylabel("Cancellation Rate (%)")
plt.xticks(rotation=25, ha="right")
save_chart("cancellations", "07_market_segment.png")


# GRAPH 08 - Cancellation rate by deposit type

deposit_stats = (
    df.groupby("DepositType")["IsCanceled"]
    .agg(["size", "mean"])
    .rename(columns={
        "size": "Bookings",
        "mean": "CancellationRate"
    })
    .sort_values("CancellationRate", ascending=False)
)

deposit_stats["CancellationRate"] *= 100

make_figure(9, 6)

ax = sns.barplot(
    x=deposit_stats.index,
    y=deposit_stats["CancellationRate"],
    color=TEAL
)

for i, (_, row) in enumerate(deposit_stats.iterrows()):
    ax.text(
        i,
        row["CancellationRate"],
        f'{row["CancellationRate"]:.1f}%\n(n={int(row["Bookings"]):,})',
        ha="center",
        va="bottom"
    )

plt.ylim(0, max(110, deposit_stats["CancellationRate"].max() * 1.2))
plt.title("08 - Cancellation Rate by Deposit Type")
plt.xlabel("Deposit Type")
plt.ylabel("Cancellation Rate (%)")
save_chart("cancellations", "08_deposit_type.png")


# ==================================================
# 7. SEASONALITY
# ==================================================

month_order = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

# GRAPH 09 - Bookings by arrival month

monthly_counts = (
    df["ArrivalDateMonth"]
    .value_counts()
    .reindex(month_order, fill_value=0)
)

make_figure(12, 6)

sns.barplot(
    x=monthly_counts.index,
    y=monthly_counts.values,
    color=BLUE
)

plt.title("09 - Booking Records by Arrival Month")
plt.xlabel("Arrival Month")
plt.ylabel("Number of Bookings")
plt.xticks(rotation=35, ha="right")
save_chart("seasonality", "09_monthly_bookings.png")


# GRAPH 10 - Average ADR by arrival month

monthly_adr = (
    df.groupby("ArrivalDateMonth")["ADR"]
    .mean()
    .reindex(month_order)
)

make_figure(12, 6)

sns.lineplot(
    x=month_order,
    y=monthly_adr.values,
    marker="o",
    color=TEAL,
    linewidth=2.5
)

plt.title("10 - Mean ADR by Arrival Month")
plt.xlabel("Arrival Month")
plt.ylabel("Mean ADR")
plt.xticks(rotation=35, ha="right")
save_chart("seasonality", "10_monthly_adr.png")


# ==================================================
# 8. FEATURE RELATIONSHIPS
# ==================================================

# GRAPH 11 - Numerical correlation heatmap

correlation_columns = [
    "IsCanceled",
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "BookingChanges",
    "ADR",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
]

correlations = df[correlation_columns].corr()

make_figure(13, 10)

sns.heatmap(
    correlations,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    annot_kws={"size": 8}
)

plt.title("11 - Numerical Feature Correlation Heatmap")
save_chart("relationships", "11_correlation_heatmap.png")


# ==================================================
# 9. EXPORT SUMMARY TABLES
# ==================================================

summary_folder = OUTPUT / "tables"
summary_folder.mkdir(exist_ok=True)

quality_summary = pd.DataFrame({
    "Feature": df.columns,
    "DataType": [str(t) for t in df.dtypes],
    "MissingValues": df.isna().sum().values,
    "UniqueValues": df.nunique(dropna=True).values
})

quality_summary.to_csv(
    summary_folder / "data_quality_summary.csv",
    index=False
)

df.describe().to_csv(
    summary_folder / "numerical_statistics.csv"
)

segment_stats.to_csv(
    summary_folder / "market_segment_statistics.csv"
)

deposit_stats.to_csv(
    summary_folder / "deposit_type_statistics.csv"
)

# ==================================================
# 10. FINAL RESULTS
# ==================================================

print("\n" + "=" * 55)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 55)

print("Total graphs generated:", len(generated))
print("Summary tables saved:", 4)
print("Output folder:", OUTPUT)

print("\nImportant notes:")
print("- Original H2.csv was not modified.")
print("- ADR plots use a display range of 0 to 300.")
print("- All original records remain available.")
print("- EDA associations do not prove causation.")
print("- Some variables may cause prediction leakage.")
print("- Monthly totals cover unequal year/month exposure.")
