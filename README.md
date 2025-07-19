# fraud_detection_project

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

---

## ⚙️ How to Run

1. Clone this repository:

```

git clone [https://github.com/Selamawit-Alemu/fraud\_detection\_project.git](https://github.com/Selamawit-Alemu/fraud_detection_project.git)
cd fraud\_detection\_project

```

2. Install dependencies:

```

pip install -r requirements.txt

```

3. Launch the notebook and run all cells:

```

jupyter notebook 1\_eda.ipynb

```

