
from pathlib import Path
import pandas as pd

# ==========================================
# STEP 1: Locate and load the dataset
# ==========================================

project_folder = Path(__file__).resolve().parents[2]
dataset_path = project_folder / "data" / "H2.csv"

df = pd.read_csv(dataset_path)

# ==========================================
# STEP 2: Basic dataset information
# ==========================================

print("Dataset loaded successfully!")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nFirst 5 bookings:")
print(df.head())

# ==========================================
# STEP 3: Missing values
# ==========================================

print("\nMissing values in each column:")
print(df.isnull().sum())

# ==========================================
# STEP 4: Duplicate rows
# ==========================================

print("\nNumber of exact duplicate rows:")
print(df.duplicated().sum())

# ==========================================
# STEP 5: Data types
# ==========================================

print("\nData types of all columns:")
print(df.dtypes)

# ==========================================
# STEP 6: Numerical statistics
# ==========================================

print("\nNumerical statistics:")
print(df.describe().to_string())

# ==========================================
# STEP 7: Investigate ADR values
# ==========================================

zero_adr = (df["ADR"] == 0).sum()
high_adr = (df["ADR"] > 300).sum()
negative_adr = (df["ADR"] < 0).sum()

print("\nADR Investigation:")
print("Bookings with ADR = 0:", zero_adr)
print("Bookings with ADR > 300:", high_adr)
print("Bookings with ADR < 0:", negative_adr)

print("\nTop 10 highest ADR values:")
print(df["ADR"].nlargest(10).to_string(index=False))

# ==========================================
# STEP 8: Examine suspicious bookings
# ==========================================

columns_to_examine = [
    "ADR",
    "IsCanceled",
    "Adults",
    "Children",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "ReservedRoomType",
    "AssignedRoomType",
    "MarketSegment",
]

print("\nBooking with the highest ADR:")
highest_adr_booking = df.nlargest(1, "ADR")
print(highest_adr_booking[columns_to_examine].to_string(index=False))

print("\nFirst 5 bookings with ADR = 0:")
zero_adr_bookings = df[df["ADR"] == 0]
print(zero_adr_bookings[columns_to_examine].head(5).to_string(index=False))

print("\nCancellation counts for bookings with ADR = 0:")
print(zero_adr_bookings["IsCanceled"].value_counts().to_string())
