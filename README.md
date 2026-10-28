# City Hotel Booking Demand — Machine Learning Project

## 1. Project Overview

This project analyzes the **City Hotel Booking Demand (H2)** dataset and develops baseline machine learning models for three tasks:

1. **Regression:** Predict the Average Daily Rate (ADR).
2. **Classification:** Predict whether a hotel booking will be canceled.
3. **Clustering:** Group bookings with similar characteristics.

The project is developed for **CSE382 — Introduction to Machine Learning**, Milestone 1.

Milestone 1 focuses on exploratory data analysis, preprocessing, leakage-aware feature selection, reproducible splitting, and initial untuned model experiments.

## 2. Dataset

**Dataset file:** `data/H2.csv`

| Property | Value |
|---|---|
| Dataset | City Hotel (H2) |
| Number of records | 79,330 |
| Original variables | 31 |
| Booking arrival period | July 2015 – August 2017 |

The canonical H2 dataset is preserved without modification by the project's preprocessing and modeling scripts.

The dataset is examined for:

- Missing and unspecified values
- Duplicate records
- Numerical and categorical distributions
- Outliers and unusual values
- Cancellation patterns
- Seasonal patterns
- Potential feature leakage

**Missing-value interpretation:** The dataset contains 28 pandas-recognized missing values in `Country` (24) and `Children` (4). The raw data also contains literal `NULL` placeholders associated with `Agent` and `Company`. These identifiers are excluded from the current baseline models. Placeholder values are audited separately because they may represent absent associations rather than unknown information.

## 3. Machine Learning Tasks

### 3.1 Regression

**Target:** `ADR` — Average Daily Rate.

Initial models:

- Dummy Mean Regressor
- Linear Regression

Evaluation metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

### 3.2 Classification

**Target:** `IsCanceled`.

Initial models:

- Dummy Majority Classifier
- Logistic Regression

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

### 3.3 Clustering

**Initial algorithm:** MiniBatchKMeans.

Clustering analysis includes:

- Cluster assignments and sizes
- Silhouette score
- Davies–Bouldin index
- Cluster-profile summaries
- Cluster visualization

Clustering is unsupervised; the cancellation and ADR targets are not used as clustering features.

## 4. Prediction Point and Feature Leakage Policy

For regression and classification, predictions are assumed to be made at **initial reservation creation**.

This prediction point determines which variables are acceptable inputs.

The project excludes features that reveal outcomes or may contain information recorded after the reservation was created.

Examples include:

- `ReservationStatus`
- `ReservationStatusDate`
- `AssignedRoomType`
- `BookingChanges`
- `DaysInWaitingList`

`DepositType` is also excluded because the timing of its recorded value cannot be established reliably for the assumed prediction point.

`Agent` and `Company` are omitted from the current baseline because they are identifiers requiring additional handling.

Some selected features remain subject to timing verification:

- `CustomerType`
- `DistributionChannel`
- `IsRepeatedGuest`
- `MarketSegment`
- `PreviousBookingsNotCanceled`
- `PreviousCancellations`

These features are retained under provisional assumptions about their availability at reservation creation.

**Limitation:** The current audit identifies and excludes known high-risk variables, but it does not establish that every selected feature is conclusively free from temporal or indirect leakage.

## 5. Data Splitting

The project uses a reproducible, grouped, approximately stratified train/validation/test split.

| Partition | Records |
|---|---:|
| Training | 56,664 |
| Validation | 11,333 |
| Test | 11,333 |
| **Total** | **79,330** |

Identical full dataset rows are assigned to the same partition. The records themselves are retained rather than deleted.

The splitting algorithm uses `StratifiedGroupKFold` with fixed random seeds:

- First split: `random_state=42`
- Second split: `random_state=43`

The split is generated from the canonical `data/H2.csv`, using hashes of the original row values.

**Reproducibility verification:** An independent in-memory recalculation matched the existing saved assignments for all 79,330 rows, with **zero partition mismatches**.

The final supervised test partition is reserved for final model evaluation. Baseline model performance is reported using validation data, not final test predictions.

## 6. Preprocessing

The project uses reusable scikit-learn preprocessing pipelines.

**Numerical preprocessing:**

- Missing-value imputation using the most frequent training value
- Standardization with `StandardScaler`

**Categorical preprocessing:**

- Missing-value imputation using an `Unknown` category
- One-hot encoding with support for previously unseen categories

All fitted preprocessing parameters are learned from the **training partition only**.

Validation observations are transformed using those fitted parameters, without fitting again.

### Current feature dimensions

| Task | Original features | Numerical | Categorical | Prepared features |
|---|---:|---:|---:|---:|
| Regression | 19 | 12 | 7 | 211 |
| Classification | 19 | 12 | 7 | 211 |
| Clustering | 10 | 6 | 4 | 30 |

For the supervised tasks, the seven categorical inputs expand into 199 one-hot encoded columns. Together with 12 numerical columns, this produces 211 prepared features.

The feature lists have been checked against the feature lists stored alongside the existing fitted preprocessing pipelines.

## 7. Project Structure

The main project directories and files are:

```text
Project/
├── data/
│   └── H2.csv
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── regression/
│   ├── classification/
│   └── clustering/
├── outputs/
├── results/
├── reports/
├── requirements.txt
├── .gitignore
└── README.md
```

`src/` contains the implementation.

`outputs/` contains generated intermediate files, fitted preprocessing objects, model artifacts, and experiment outputs.

`results/` contains selected tables and figures intended for reporting.

`reports/` is intended for the milestone progress report and related documentation.

**Note:** Generated outputs and fitted artifacts may not be present in a fresh GitHub clone. They must be regenerated using the appropriate scripts.

## 8. Installation

**Recommended Python version:** 3.12.

Open PowerShell in the project root directory.

### Create a virtual environment

```powershell
python -m venv .venv
```

### Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

Keep the dependency versions consistent when reproducing the reported results.

## 9. Current Baseline Results

All supervised results below were calculated on the **validation partition**.

### 9.1 Regression — Linear Regression

| Metric | Value |
|---|---:|
| MAE | 17.1785 |
| RMSE | 23.6941 |
| R² | 0.6142 |

For comparison, the Dummy Mean Regressor produced an MAE of approximately 28.8927 and an RMSE of 38.1470.

Linear Regression therefore provides a substantial improvement over predicting the training-set mean ADR.

### 9.2 Classification — Logistic Regression

| Metric | Value |
|---|---:|
| Accuracy | 0.7608 |
| Precision | 0.7587 |
| Recall | 0.6257 |
| F1-score | 0.6858 |
| ROC-AUC | 0.8389 |

The Dummy Majority Classifier produced approximately 58.27% validation accuracy.

Logistic Regression improves prediction accuracy and ranking performance, but its recall indicates that some canceled bookings are still incorrectly classified as not canceled.

### 9.3 Clustering — MiniBatchKMeans

| Metric | Value |
|---|---:|
| Number of clusters | 3 |
| Silhouette score | 0.1733 |
| Davies–Bouldin index | 1.6367 |

The observed silhouette score suggests relatively weak separation between the initial booking clusters.

The three-cluster configuration is a preliminary baseline, not a final optimized selection.

## 10. Reproducing Milestone 1 from the Original Dataset

This section describes the intended execution order for generating the Milestone 1 baselines from the canonical dataset.

**Prerequisites:**

- Python 3.12
- Installed dependencies from `requirements.txt`
- Canonical City Hotel dataset located at `data/H2.csv`
- Commands executed from the project root

**Important:** Run this procedure in a separate fresh project copy if you need to preserve existing generated artifacts. The scripts create or overwrite files under `outputs/`.

### Step 1 — Generate train/validation/test assignments

```powershell
python src/preprocessing/split_data.py
```

This generates:

`outputs/splits/split_assignments.csv`

It uses grouped splitting and fixed random seeds.

### Step 2 — Generate selected feature lists and leakage audit

```powershell
python src/data/feature_audit.py
```

This generates the selected features for regression, classification, and clustering and an accompanying feature-leakage audit.

Expected original feature counts:

| Task | Features |
|---|---:|
| Regression | 19 |
| Classification | 19 |
| Clustering | 10 |

This script does not require historical leakage-review backup files.

### Step 3 — Build preprocessing pipelines

```powershell
python src/preprocessing/build_pipelines.py
```

This constructs and fits the three preprocessing pipelines using training data only.

Expected prepared dimensions are 211 for regression, 211 for classification, and 30 for clustering.

### Step 4 — Run initial baseline experiments

```powershell
python src/regression/regression_baseline.py
python src/classification/classification_baseline.py
python src/clustering/clustering_baseline.py
```

These scripts perform the initial experiments and generate model-related outputs.

### Step 5 — Generate baseline visualizations

```powershell
python src/regression/visualize_regression.py
python src/classification/visualize_classification.py
python src/clustering/visualize_clustering.py
```

### Step 6 — Run additional verification and visualizations

```powershell
python src/data/verify_feature_leakage.py
python src/preprocessing/visualize_encoding.py
python src/preprocessing/visualize_pipelines.py
```

These commands provide supporting verification and visualizations for the current preprocessing configuration.

### Reproducibility limitations

- The saved split has been independently reproduced with zero partition mismatches.
- Current selected feature lists match the stored fitted-pipeline feature lists.
- Current baseline model metrics have been independently checked against the saved validation predictions.
- The preprocessing-building script does not require the historical leakage-review backups.
- The complete command sequence has **not yet been executed and verified end-to-end on a fresh clone**.
- Successful execution of individual scripts does not establish complete fresh-environment reproducibility.
- The canonical dataset representation and installed library versions must remain consistent.

## 11. Running the Existing Project

If the saved split assignments, preprocessing pipelines, and other required generated outputs already exist locally, the following commands can be used.

### Regression

```powershell
python src/regression/regression_baseline.py
python src/regression/visualize_regression.py
```

### Classification

```powershell
python src/classification/classification_baseline.py
python src/classification/visualize_classification.py
```

### Clustering

```powershell
python src/clustering/clustering_baseline.py
python src/clustering/visualize_clustering.py
```

### Verify feature selection

```powershell
python src/data/verify_feature_leakage.py
```

Running the baseline scripts may regenerate or overwrite saved model files and results. Preserve existing artifacts before repeating experiments.

## 12. Important Notes

- The current models are **untuned Milestone 1 baselines**.
- The final supervised test set has not been used for baseline model evaluation.
- Some selected feature-timing assumptions remain provisional.
- Duplicate bookings are retained; exact full-row duplicates are grouped during splitting.
- The project uses the canonical H2 dataset for its current experiments.
- Generated outputs should be regenerated when working from a fresh project copy.
- No hyperparameter optimization has been performed yet.
- Current results should not be interpreted as final model performance.

## 13. Future Work — Milestone 2

The next milestone will expand the project beyond the initial baselines.

Planned work includes:

- Evaluating at least three regression models and three classification models
- Comparing baseline and hyperparameter-tuned configurations
- Testing at least three clustering algorithms and optimizing their parameters
- Applying PCA and LDA where appropriate
- Comparing models using consistent evaluation procedures
- Strengthening the feature leakage audit and documenting prediction-time assumptions
- Performing final supervised test-set evaluation after model selection
- Developing the required end-to-end CLI or graphical application
- Preparing the final project report and live demonstration

## 14. Dataset Source and Attribution

The project uses the City Hotel dataset (`H2.csv`) from:

António, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41–49.

**DOI:** https://doi.org/10.1016/j.dib.2018.11.126

**Dataset source:** https://github.com/afaq-ahmad/Hotel-booking-demand-Case-Study

**Original research article:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/

The dataset is attributed to its original authors and used for educational machine learning research.

License reference: https://creativecommons.org/licenses/by/4.0/

## 15. GitHub Repository and Submission Version

**Project GitHub repository:**

https://github.com/saifbaher2020/city-hotel-booking-ml

**Branch:** `main`

**Last verified pushed commit before this README update:**

`800198a`

**Milestone 1 submitted commit SHA:** To be recorded after final review, completion of the progress report, and preparation of the submission package.

The commit listed above documents an earlier verified project state. It is not claimed to contain this updated README or represent the final Milestone 1 submission.

The repository history should show meaningful contributions from the project's individual team members, using their own GitHub accounts.

**Team member names, student IDs, GitHub usernames, and contribution descriptions:** To be completed with the confirmed team roster before Milestone 1 submission.