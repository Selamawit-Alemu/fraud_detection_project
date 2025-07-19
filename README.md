# Fraud_detection_project

## 📌 Project Overview

This project focuses on detecting fraudulent transactions in e-commerce and credit card datasets. It involves comprehensive data cleaning, exploratory data analysis (EDA), feature engineering, and handling of class imbalance to prepare the data for robust fraud detection modeling. The goal is to uncover fraud patterns and build a strong foundation for accurate and explainable machine learning models.

## Project Milestones & Completed Work

### 1. Data Loading and Initial Inspection
- Loaded datasets: `fraud_data.csv`, `ip_to_country.csv`, and `creditcard.csv`.
- Conducted initial inspection of data structure and content.

### 2. Data Cleaning
- **Duplicate Removal**: Successfully eliminated duplicates from `creditcard.csv`.
- **Missing Value Handling**: Verified no missing values remain across all datasets after cleaning.

### 3. Data Type Conversion
- Converted timestamp columns (`signup_time`, `purchase_time`) to `datetime` objects.
- Converted `ip_address` columns to `int64` to facilitate merging.
- Ensured consistency of numerical feature types.

### 4. Exploratory Data Analysis (EDA)
- **Class Imbalance Check**: Identified severe class imbalance in both fraud datasets.
- **Summary Statistics**: Analyzed distributions of `purchase_value` and `age`.
- **Categorical Feature Analysis**: Examined value counts for `source`, `browser`, and `sex`.
- **Visualizations**:
  - Fraud Rate by Browser
  - Age Distribution by Fraud Class
  - Purchase Value Distribution by Fraud Class
  - Transaction Amount by Fraud Class (Credit Card Data)
  - Log-Transformed Transaction Amount Distribution
  - Time from Signup to Purchase (Fraud vs Non-Fraud) — *Strong predictor identified*
  - Transactions by Hour of Day
  - Top 10 Countries by Transaction Volume

### 5. Feature Engineering
- Derived `hours_since_signup` (time difference between signup and purchase).
- Created `hour_of_day` and `day_of_week` features from purchase timestamps.
- Implemented IP-to-Country mapping by merging `fraud_data` with `ip_to_country.csv` using `pd.merge_asof`. Unmatched IPs labeled as `'Unknown'`.
- Engineered binary flags: 
  - `new_account` (purchases within 1 hour of signup)
  - `off_hour_purchase` (transactions between 8 PM and 6 AM)
- **Redundant Features** identified: `user_transaction_count` and `time_since_last_purchase` were constant per user and removed.

### 6. Data Preprocessing for Modeling
- Applied `StandardScaler` for numerical feature scaling (`Amount`, `Time`) in `credit_data`.
- Used one-hot encoding (with `drop_first=True`) for categorical variables: `source`, `browser`, `sex`, and `country`.
- Performed stratified train-test split to maintain class distribution.
- Addressed class imbalance with resampling techniques (e.g., minority class upsampling) and computed class weights.

### 7. Code Structure & Reproducibility
- Developed in Jupyter Notebook with clear, well-documented sections.
- Saved cleaned datasets (`cleaned_fraud_data.csv`, `cleaned_credit_data.csv`) and processed checkpoints using `pickle` for reproducibility.

## 📁 Folder Structure

fraud_detection_project/
│
├── data/ # Contains raw and cleaned data files
├── docs/ # Contains detailed reports
├── notebooks/1_eda.ipynb # Main Jupyter notebook for Task 1
├── README.md # Project documentation
└── requirements.txt # Python environment dependencies 

## ⚙️ How to Run

1. Clone this repository:

```

git clone https://github.com/Selamawit-Alemu/fraud_detection_project.git
cd fraud_detection_project

```

2. Install dependencies:

```

pip install -r requirements.txt

```

3. Launch the notebook and run all cells:

```

jupyter notebook 1_eda.ipynb

```

🛠️ Dependencies

    Python 3.8+

    pandas

    numpy

    matplotlib / seaborn

    scikit-learn

    imbalanced-learn

    ipaddress (for IP conversion)


## 📌 Assumptions Made

    IP addresses are assumed to be correctly mapped to countries using the ip_to_country.csv reference via merge_asof. Unmatched IPs are categorized as 'Unknown'.

    Timestamps such as signup_time and purchase_time are assumed to be in the same timezone and accurately reflect user behavior.

    Features like user_transaction_count and time_since_last_purchase were assumed redundant since they showed constant values per user.

    Categorical values like source, browser, and sex are treated as complete and meaningful despite their imbalance or limited categories.

    Fraud labels are assumed to be correctly annotated with no labeling noise or bias.