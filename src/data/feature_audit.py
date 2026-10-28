
from pathlib import Path
import pandas as pd

# ==================================================
# STAGE 2: FEATURE SELECTION AND DATA LEAKAGE AUDIT
# STAGE 5.1.4: UPDATE AFTER LEAKAGE REVIEW
# ==================================================

# --------------------------------------------------
# 5.1.4.1 - Locate project files
# --------------------------------------------------

project_folder = Path(__file__).resolve().parents[2]
dataset_path = project_folder / "data" / "H2.csv"
output_folder = project_folder / "outputs"

# --------------------------------------------------
# 5.1.4.2 - Verify backups before making changes
# --------------------------------------------------

required_backups = [
    output_folder / "baselines_before_leakage_review",
    output_folder / "preprocessing" / "pipelines_before_leakage_review",
    output_folder / "regression_features_before_leakage_review.txt",
    output_folder / "classification_features_before_leakage_review.txt",
]

missing_backups = [
    str(path)
    for path in required_backups
    if not path.exists()
]

if missing_backups:
    raise FileNotFoundError(
        "Required backup is missing:\n"
        + "\n".join(missing_backups)
    )

print("All required backups found.")

# --------------------------------------------------
# 5.1.4.3 - Load the original dataset
# --------------------------------------------------

df = pd.read_csv(dataset_path)

print("\nDataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# --------------------------------------------------
# 5.1.4.4 - Define the prediction point
# --------------------------------------------------

# Regression and Classification:
# Predictions are made when the booking is created.
#
# Clustering:
# Group bookings using their booking characteristics.

prediction_point = "Initial reservation creation"

# --------------------------------------------------
# 5.1.4.5 - Define supervised input features
# --------------------------------------------------

# DepositType has been excluded after our
# preliminary leakage review.
#
# Its stored value may reflect payment information
# recorded after initial reservation creation.
#
# Other retained historical/customer features
# still have documented timing assumptions.

common_features = [
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "CustomerType",
]

# --------------------------------------------------
# 5.1.4.6 - Define Regression
# --------------------------------------------------

regression_features = common_features.copy()
regression_target = "ADR"

# --------------------------------------------------
# 5.1.4.7 - Define Classification
# --------------------------------------------------

classification_features = common_features.copy()
classification_target = "IsCanceled"

# --------------------------------------------------
# 5.1.4.8 - Keep Clustering unchanged
# --------------------------------------------------

clustering_features = [
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "MarketSegment",
    "ReservedRoomType",
    "CustomerType",
]

# --------------------------------------------------
# 5.1.4.9 - Document excluded features
# --------------------------------------------------

excluded_reasons = {
    "IsCanceled":
        "Classification target; not an input feature",

    "ADR":
        "Regression target; excluded from other initial tasks",

    "DepositType":
        "Excluded after leakage review: booking-time value "
        "cannot be established reliably from the dataset",

    "AssignedRoomType":
        "Room assignment may occur after reservation",

    "BookingChanges":
        "May include changes made after reservation",

    "Agent":
        "Identifier omitted from initial baseline",

    "Company":
        "Identifier omitted from initial baseline",

    "DaysInWaitingList":
        "May reflect post-reservation information",

    "RequiredCarParkingSpaces":
        "Timing of requests requires verification",

    "TotalOfSpecialRequests":
        "May reflect requests added later",

    "ReservationStatus":
        "Reveals final reservation outcome",

    "ReservationStatusDate":
        "Date associated with final reservation status",
}

# --------------------------------------------------
# 5.1.4.10 - Verify the selected features
# --------------------------------------------------

task_features = {
    "Regression": regression_features,
    "Classification": classification_features,
    "Clustering": clustering_features,
}

all_columns = set(df.columns)

for task_name, features in task_features.items():

    assert len(features) == len(set(features))
    assert set(features).issubset(all_columns)

assert len(regression_features) == 19
assert len(classification_features) == 19
assert len(clustering_features) == 10

assert "DepositType" not in regression_features
assert "DepositType" not in classification_features
assert "DepositType" not in clustering_features

assert "ADR" not in regression_features
assert "IsCanceled" not in classification_features

assert "ADR" not in clustering_features
assert "IsCanceled" not in clustering_features

for features in task_features.values():

    assert "ReservationStatus" not in features
    assert "ReservationStatusDate" not in features

# --------------------------------------------------
# 5.1.4.11 - Create updated feature audit table
# --------------------------------------------------

audit_rows = []

for column in df.columns:

    audit_rows.append({
        "Feature": column,
        "DataType": str(df[column].dtype),
        "MissingValues": int(df[column].isna().sum()),

        "RegressionRole": (
            "Target" if column == regression_target
            else "Input" if column in regression_features
            else "Excluded"
        ),

        "ClassificationRole": (
            "Target" if column == classification_target
            else "Input" if column in classification_features
            else "Excluded"
        ),

        "ClusteringRole": (
            "Input" if column in clustering_features
            else "Excluded"
        ),

        "Reason": excluded_reasons.get(
            column,
            "Candidate booking-time feature; "
            "timing assumptions remain provisional"
        ),
    })

audit_df = pd.DataFrame(audit_rows)

# This audit's MissingValues column records pandas NaNs.
# The separate Stage 5.1 missing-value audit also
# documents literal 'NULL' placeholders.

# --------------------------------------------------
# 5.1.4.12 - Save updated feature lists
# --------------------------------------------------

output_folder.mkdir(parents=True, exist_ok=True)

audit_path = (
    output_folder / "feature_leakage_audit.csv"
)

audit_df.to_csv(audit_path, index=False)

for task_name, features in task_features.items():

    filename = (
        task_name.lower() + "_features.txt"
    )

    path = output_folder / filename

    path.write_text(
        "\n".join(features) + "\n",
        encoding="utf-8"
    )

# --------------------------------------------------
# 5.1.4.13 - Display results
# --------------------------------------------------

print("\n" + "=" * 60)
print("STAGE 5.1.4 - FEATURE UPDATE COMPLETE")
print("=" * 60)

print("\nPrediction point:", prediction_point)

print("\nRegression features:", len(regression_features))
print("Classification features:", len(classification_features))
print("Clustering features:", len(clustering_features))

print("\nDepositType:")
print(
    audit_df.loc[
        audit_df["Feature"] == "DepositType",
        [
            "RegressionRole",
            "ClassificationRole",
            "ClusteringRole"
        ]
    ].to_string(index=False)
)

print("\nUpdated feature lists saved.")
print("Updated leakage audit saved:", audit_path)

print("\nOriginal H2.csv modified: NO")
print("Old results backed up: YES")
print("Preprocessing pipelines rebuilt: NO")
print("Models retrained: NO")
print("Test partition accessed: NO")

print("\nNEXT: Rebuild supervised preprocessing pipelines.")
