
# City Hotel Booking Demand — Machine Learning Project

## 1. Project Overview

This project analyzes the City Hotel Booking Demand dataset
and develops baseline machine learning models for three tasks:

1. Regression: Predict the Average Daily Rate (ADR).
2. Classification: Predict whether a hotel booking will be canceled.
3. Clustering: Group bookings with similar characteristics.

The project is developed for the Introduction to Machine
Learning (CSE382) course, Milestone 1.

## 2. Dataset

Dataset file: `data/H2.csv`

- Number of records: 79,330
- Number of original columns: 31
- Dataset: City Hotel bookings

The original dataset is preserved without modification.

The project includes exploratory data analysis, missing-value
inspection, duplicate-row analysis, and feature-leakage review.

## 3. Machine Learning Tasks

### Regression

Target: `ADR`

Models:
- Dummy Mean Regressor
- Linear Regression

Evaluation metrics:
- MAE
- RMSE
- R-squared

### Classification

Target: `IsCanceled`

Models:
- Dummy Majority Classifier
- Logistic Regression

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

### Clustering

Model:
- MiniBatchKMeans

Evaluation includes:
- Cluster sizes
- Silhouette score
- Davies-Bouldin index
- Cluster visualization

## 4. Prediction Point and Leakage Policy

Supervised predictions are intended to be made when the
customer creates the hotel reservation.

The feature `DepositType` is excluded from the current
supervised models because its recorded timing cannot
be established reliably for this prediction point.

Other historical and booking-related features are retained
under documented assumptions about their availability.

The leakage checks verify consistency with this policy.
They do not prove that every feature was available at
reservation creation.

## 5. Data Splitting

The saved data split contains:

- Training: 56,664 rows
- Validation: 11,333 rows
- Test: 11,333 rows

Identical full rows are grouped to prevent them from
appearing in different partitions.

The Test set is reserved and has not been used for
baseline model evaluation.

## 6. Preprocessing

The project uses scikit-learn preprocessing pipelines.

Numerical features:
- Missing-value imputation
- StandardScaler

Categorical features:
- Missing-value imputation
- One-Hot Encoding

The preprocessing transformations are fitted on
the Training data only.

Current feature counts:

| Task | Original Features | Prepared Features |
|---|---:|---:|
| Regression | 19 | 211 |
| Classification | 19 | 211 |
| Clustering | 10 | 30 |

## 7. Project Structure

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
├── requirements.txt
├── .gitignore
└── README.md
```

## 8. Installation

Recommended Python version: 3.12

Create a virtual environment in the project directory:

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

## 9. Current Baseline Results

Results below are from the Validation partition.

### Regression — Linear Regression

| Metric | Value |
|---|---:|
| MAE | 17.1785 |
| RMSE | 23.6941 |
| R-squared | 0.6142 |

### Classification — Logistic Regression

| Metric | Value |
|---|---:|
| Accuracy | 0.7608 |
| Precision | 0.7587 |
| Recall | 0.6257 |
| F1 | 0.6858 |
| ROC-AUC | 0.8389 |

### Clustering — MiniBatchKMeans

- Number of clusters: 3
- Silhouette score: approximately 0.1733
- Davies-Bouldin index: approximately 1.6367

The clustering metrics indicate limited separation
between the identified booking groups.

## 10. Running the Existing Project

The following commands work with the saved split
assignments and fitted preprocessing pipelines.

Run Regression:

```powershell
python src/regression/regression_baseline.py
python src/regression/visualize_regression.py
```

Run Classification:

```powershell
python src/classification/classification_baseline.py
python src/classification/visualize_classification.py
```

Run Clustering:

```powershell
python src/clustering/clustering_baseline.py
python src/clustering/visualize_clustering.py
```

Verify the current preprocessing setup:

```powershell
python src/preprocessing/visualize_pipelines.py
python src/data/verify_feature_leakage.py
```

These commands require the relevant generated files
inside `outputs/`.

IMPORTANT: A full clean-machine execution from
the original CSV has not yet been verified.
The complete execution order and dependencies
will be finalized during the reproducibility audit.

## 11. Important Notes

- The original H2.csv dataset is not modified.
- The Test set is reserved for later final evaluation.
- The current models are untuned Milestone 1 baselines.
- No hyperparameter optimization has been performed.
- Some feature-timing assumptions remain provisional.
- Generated files are stored under `outputs/`.

## 12. Future Work

Planned work for Milestone 2 includes:

- Hyperparameter tuning
- Additional model evaluation
- Feature engineering and selection
- PCA and LDA experiments where appropriate
- Model comparison and final application development
Get-ChildItem -Force -File -Name

## 13. Dataset Source and Attribution

This project uses the City Hotel dataset (`H2.csv`).

Original academic publication:

António, N., de Almeida, A., & Nunes, L. (2019).
Hotel booking demand datasets.
Data in Brief, 22, 41–49.

DOI: https://doi.org/10.1016/j.dib.2018.11.126

Dataset downloaded from:
https://github.com/afaq-ahmad/Hotel-booking-demand-Case-Study

The original research publication and dataset are
distributed under Creative Commons Attribution 4.0
(CC BY 4.0).

License: https://creativecommons.org/licenses/by/4.0/

The dataset is used for educational machine learning
research. The original H2.csv file is preserved
without modification.
