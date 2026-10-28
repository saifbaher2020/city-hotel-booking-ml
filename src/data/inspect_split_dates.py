
from pathlib import Path
import pandas as pd

# ==================================================
# STAGE 2: FEATURE SELECTION AND LEAKAGE AUDIT
# ==================================================

# --------------------------------------------------
# 2.14.1 - Load the original dataset
# --------------------------------------------------

project_folder = Path(__file__).resolve().parents[2]
dataset_path = project_folder / "data" / "H2.csv"

df = pd.read_csv(dataset_path)

# --------------------------------------------------
# 2.14.2 - Construct scheduled arrival dates
# --------------------------------------------------

arrival_dates = pd.to_datetime(
    df["ArrivalDateYear"].astype(str)
    + "-"
    + df["ArrivalDateMonth"]
    + "-"
    + df["ArrivalDateDayOfMonth"].astype(str),
    format="%Y-%B-%d",
    errors="coerce"
)

# --------------------------------------------------
# 2.14.3 - Inspect date coverage
# --------------------------------------------------

print("=" * 55)
print("STAGE 2 - DATE INVESTIGATION")
print("=" * 55)

print("\nTotal bookings:", len(df))
print("Earliest scheduled arrival:", arrival_dates.min())
print("Latest scheduled arrival:", arrival_dates.max())
print("Invalid arrival dates:", arrival_dates.isna().sum())

print("\nBookings by arrival year:")
print(df["ArrivalDateYear"].value_counts().sort_index())

# --------------------------------------------------
# 2.14.4 - Inspect monthly coverage
# --------------------------------------------------

monthly_counts = (
    df.assign(ArrivalMonth=arrival_dates.dt.to_period("M"))
    .groupby("ArrivalMonth")
    .size()
)

print("\nBookings by year and month:")
print(monthly_counts.to_string())

# --------------------------------------------------
# 2.14.5 - Identify date limitation
# --------------------------------------------------

print("\nIMPORTANT:")
print("These are scheduled arrival dates.")
print("They are not necessarily reservation creation dates.")
print("No dataset splitting has been performed.")
print("The original H2.csv was not modified.")
