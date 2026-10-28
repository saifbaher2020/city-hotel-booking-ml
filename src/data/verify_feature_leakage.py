
from pathlib import Path
import pandas as pd

# ==========================================================
# STAGE 5: MILESTONE 1 PROJECT AUDIT
# STEP 5.1.3 - FEATURE LEAKAGE VERIFICATION
# ==========================================================

# ----------------------------------------------------------
# 5.1.3.1 - Locate project files
# ----------------------------------------------------------

project_folder = Path(__file__).resolve().parents[2]

outputs_folder = project_folder / "outputs"
audit_folder = outputs_folder / "audit"

audit_folder.mkdir(parents=True, exist_ok=True)

feature_audit_path = (
    outputs_folder / "feature_leakage_audit.csv"
)

missing_audit_path = (
    audit_folder / "missing_value_semantic_audit.csv"
)

# ----------------------------------------------------------
# 5.1.3.2 - Load existing audit information
# ----------------------------------------------------------

audit = pd.read_csv(feature_audit_path)
missing = pd.read_csv(missing_audit_path)

missing_counts = dict(
    zip(
        missing["Feature"],
       missing["TotalFlagged"]
    )
)

audit["VerifiedMissingValues"] = (
    audit["Feature"]
    .map(missing_counts)
    .fillna(0)
    .astype(int)
)

# ----------------------------------------------------------
# 5.1.3.3 - Load actual selected feature lists
# ----------------------------------------------------------

tasks = [
    "regression",
    "classification",
    "clustering"
]

feature_lists = {}

for task in tasks:

    path = outputs_folder / f"{task}_features.txt"

    feature_lists[task] = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

# ----------------------------------------------------------
# 5.1.3.4 - Define known leakage-risk columns
# ----------------------------------------------------------

# These columns are excluded because they either
# reveal an outcome or may describe later events.

excluded_features = {
    "ReservationStatus",
    "ReservationStatusDate",
    "AssignedRoomType",
    "BookingChanges",
    "DaysInWaitingList",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests"
}

# These selected features require confirmation that
# their values were available at reservation creation.

timing_review_features = {
    "MarketSegment",
    "DistributionChannel",
    "DepositType",
    "CustomerType",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled"
}

# ----------------------------------------------------------
# 5.1.3.5 - Verify feature selection
# ----------------------------------------------------------

assert set(audit["Feature"]) == set(
    missing["Feature"]
)

assert len(audit) == 31

assert len(feature_lists["regression"]) == 19
assert len(feature_lists["classification"]) == 19
assert len(feature_lists["clustering"]) == 10

assert "ADR" not in feature_lists["regression"]

assert "IsCanceled" not in feature_lists[
    "classification"
]

for task in tasks:

    features = feature_lists[task]

    assert len(features) == len(set(features))

    assert set(features).isdisjoint(
        excluded_features
    )

    if task == "clustering":

        assert "ADR" not in features
        assert "IsCanceled" not in features

    print(
        f"{task.capitalize()}: "
        f"{len(features)} features checked."
    )

# ----------------------------------------------------------
# 5.1.3.6 - Assign preliminary risk-review status
# ----------------------------------------------------------

statuses = []
notes = []

for feature in audit["Feature"]:

    selected_for = [
        task
        for task in tasks
        if feature in feature_lists[task]
    ]

    if feature in excluded_features:

        status = "Excluded - potential leakage"
        note = (
            "May reveal the outcome or information "
            "recorded after the initial reservation."
        )

    elif feature in timing_review_features and selected_for:

        status = "Selected - timing verification needed"
        note = (
            "Confirm that the stored value was known "
            "at initial reservation time and was not "
            "updated using future events."
        )

    elif selected_for:

        status = "Selected - provisional"
        note = (
            "Candidate initial-reservation feature. "
            "Availability and data provenance must "
            "be confirmed."
        )

    else:

        status = "Excluded - other reason"
        note = (
            "Not included in the current baseline "
            "feature selection."
        )

    statuses.append(status)
    notes.append(note)

audit["FinalReviewStatus"] = statuses
audit["AuditNote"] = notes

# ----------------------------------------------------------
# 5.1.3.7 - Save updated audit separately
# ----------------------------------------------------------

output_path = (
    audit_folder
    / "milestone1_feature_leakage_review.csv"
)

audit.to_csv(output_path, index=False)

# ----------------------------------------------------------
# 5.1.3.8 - Display summary
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STAGE 5.1.3 - FEATURE LEAKAGE REVIEW")
print("=" * 60)

print("\nPrediction point:")
print("Initial reservation creation")

print("\nFeature review:")
print(
    audit["FinalReviewStatus"]
    .value_counts()
    .to_string()
)

print("\nSelected features requiring timing review:")

for feature in sorted(timing_review_features):

    if any(
        feature in feature_lists[task]
        for task in tasks
    ):
        print("-", feature)

print("\nAudit saved:", output_path)

print("\nIMPORTANT:")
print("Known high-risk features excluded: YES")
print("Booking-time provenance fully verified: NO")
print("Feature lists changed: NO")
print("Preprocessing pipelines changed: NO")
print("Models retrained: NO")
print("Original dataset modified: NO")

print("\nSTAGE 5.1.3 - REVIEW COMPLETE")
