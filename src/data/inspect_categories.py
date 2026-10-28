
from pathlib import Path
import pandas as pd

# ==========================================
# STEP 1: Load the dataset
# ==========================================

project_folder = Path(__file__).resolve().parents[2]
dataset_path = project_folder / "data" / "H2.csv"

df = pd.read_csv(dataset_path)

# ==========================================
# STEP 2: Select categorical columns
# ==========================================

columns_to_inspect = [
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "ReservedRoomType",
    "AssignedRoomType",
    "DepositType",
    "CustomerType",
    "ReservationStatus",
]

# ==========================================
# STEP 3: Show category counts
# ==========================================

for column in columns_to_inspect:
    print("\n" + "=" * 50)
    print("COLUMN:", column)
    print("=" * 50)

    print("Number of unique categories:")
    print(df[column].nunique(dropna=True))

    print("\nTop 10 most common values:")
    print(
        df[column]
        .value_counts(dropna=False)
        .head(10)
        .to_string()
    )

# ==========================================
# STEP 4: Finish
# ==========================================

print("\nCategorical inspection completed!")
