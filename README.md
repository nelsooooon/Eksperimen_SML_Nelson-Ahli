# Telco Customer Churn Analysis & Preprocessing

A machine learning project focused on analyzing and preprocessing the Telco Customer Churn dataset from Kaggle. This project includes comprehensive data exploration, preprocessing pipeline automation, and feature engineering for churn prediction.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Key Components](#key-components)
- [Requirements](#requirements)
- [Author](#author)

## Overview

This project implements a complete data preprocessing pipeline for customer churn analysis. It demonstrates best practices in data cleaning, feature engineering, and automated preprocessing using scikit-learn pipelines. The goal is to prepare the Telco Customer Churn dataset for machine learning models that can predict customer churn.

## Dataset

**Source:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle

The dataset contains customer information from a telecommunications company including:

- **Customer Demographics:** Gender, Senior Citizen status, Partner, Dependents
- **Service Details:** Phone service, Multiple lines, Internet service, Online security, etc.
- **Account Information:** Tenure, Contract type, Payment method, Monthly charges, Total charges
- **Target Variable:** Churn (Yes/No) - whether the customer left the company

**Dataset Size:** 7,043 customers with 21 features

## Project Structure

```
Eksperimen_SML_Nelson-Ahli/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv          # Original dataset
│
└── preprocessing/
    ├── Eksperimen_Nelson_Ahli.ipynb              # Jupyter notebook with full EDA
    ├── automate_nelson_ahli.py                    # Automated preprocessing script
    ├── data.csv                                   # Column headers file
    ├── preprocessor_pipeline.joblib               # Saved preprocessing pipeline
    └── WA_Fn-UseC_-Telco-Customer-Churn_preprocessing.csv  # Preprocessed dataset
```

## Features

### Exploratory Data Analysis (EDA)

- Comprehensive data visualization with correlation heatmaps
- Distribution analysis for all features
- Target variable correlation analysis
- Missing value detection and analysis

### Data Preprocessing

- **Missing Value Handling:** Median imputation for numerical features
- **Duplicate Removal:** Automatic detection and removal
- **Feature Engineering:** Tenure binning into meaningful categories
- **Encoding:**
  - Label Encoding for binary target variable
  - One-Hot Encoding for categorical features
  - Ordinal Encoding for ordered categories
- **Scaling:** StandardScaler for numerical features
- **Pipeline Automation:** Reusable scikit-learn pipeline

### Automation

- Fully automated preprocessing script
- Serialized pipeline for production use
- Easy-to-use functions for preprocessing new data

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Setup

1. **Clone the repository:**

```bash
git clone https://github.com/nelsooooon/Eksperimen_SML_Nelson-Ahli.git
cd Eksperimen_SML_Nelson-Ahli
```

2. **Install required packages:**

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

3. **For Jupyter Notebook (optional):**

```bash
pip install notebook
```

## Usage

### Option 1: Run Automated Preprocessing Script

```bash
python preprocessing/automate_nelson_ahli.py
```

This will:

- Load the original dataset
- Apply all preprocessing steps
- Save the preprocessed data to `preprocessing/WA_Fn-UseC_-Telco-Customer-Churn_preprocessing.csv`
- Save the preprocessing pipeline to `preprocessing/preprocessor_pipeline.joblib`
- Generate column headers file at `preprocessing/data.csv`

### Option 2: Interactive Exploration with Jupyter Notebook

```bash
jupyter notebook preprocessing/Eksperimen_Nelson_Ahli.ipynb
```

The notebook includes:

1. Dataset introduction and loading
2. Library imports
3. Comprehensive EDA with visualizations
4. Step-by-step preprocessing
5. Data quality checks

### Option 3: Use the Saved Pipeline

```python
from joblib import load
import pandas as pd

# Load the preprocessing pipeline
preprocessor = load('preprocessing/preprocessor_pipeline.joblib')

# Load new data
new_data = pd.read_csv('your_new_data.csv')

# Apply preprocessing
processed_data = preprocessor.transform(new_data)
```

## Preprocessing Pipeline

The automated preprocessing pipeline includes:

### 1. Data Cleaning

- Remove duplicate records
- Drop `customerID` column (non-predictive)
- Convert `TotalCharges` to numeric, handling errors

### 2. Feature Engineering

**Tenure Binning:**

- `< 1 Year`: 0-12 months
- `1-2 Years`: 13-24 months
- `2-4 Years`: 25-48 months
- `4-5 Years`: 49-60 months
- `> 5 Years`: 61+ months

### 3. Feature Transformation

**Numerical Features:** `tenure`, `MonthlyCharges`, `TotalCharges`

- Median imputation for missing values
- StandardScaler normalization

**Categorical Features:** Service-related and demographic features

- Constant imputation ('missing' for unknown)
- One-Hot Encoding

**Ordinal Features:** `tenure_binning`

- Most frequent value imputation
- Ordinal Encoding with predefined order

### 4. Target Encoding

- Label Encoding for `Churn` (Yes → 1, No → 0)

## Key Components

### `automate_nelson_ahli.py`

Main function: `preprocess_data(data, target_column, save_path, file_path, final_dataset_path)`

**Parameters:**

- `data`: Input DataFrame
- `target_column`: Name of the target variable (default: 'Churn')
- `save_path`: Path to save the preprocessing pipeline
- `file_path`: Path to save column headers
- `final_dataset_path`: Path to save preprocessed data

**Outputs:**

- Preprocessed dataset (saved to CSV)
- Serialized preprocessing pipeline (joblib file)
- Column headers file (CSV)

**Returns:**

- None

## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
```

## Author

**Nelson Ahli**

- GitHub: [@nelsooooon](https://github.com/nelsooooon)
- Repository: [Eksperimen_SML_Nelson-Ahli](https://github.com/nelsooooon/Eksperimen_SML_Nelson-Ahli)

## Notes

- The notebook was originally designed for Google Colab and includes Kaggle API integration
- The preprocessing pipeline is reusable and can be applied to similar datasets
- All transformations are reversible for model interpretation
- The saved pipeline ensures consistency between training and production environments

## Future Enhancements

- Machine learning model training and evaluation
- Hyperparameter tuning
- Feature importance analysis
- Model deployment pipeline
- API endpoint for real-time predictions

---

**License:** This project is available for educational and research purposes.
