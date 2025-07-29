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

### 7. Fraud Detection Modeling


2. Data Preparation

2.1. Fraud Data (processed_fraud_data.csv)

    Target Variable: 'class'

    Dropped Columns: 'user_id', 'signup_time', 'purchase_time'

    Feature Types Identified:

        Categorical: 'device_id', 'source', 'browser', 'sex', 'country'

        Numerical: 'purchase_value', 'age', 'ip_address', 'hours_to_purchase', 'user_transaction_count', 'time_since_last_purchase', 'hour_of_day', 'day_of_week', 'time_since_signup', 'time_to_purchase', 'purchase_hour', 'purchase_dayofweek', 'device_transaction_count', 'user_unique_ip_count', 'ip_unique_device_count', 'log_purchase_value'

    Class Distribution: Highly imbalanced, with 'class 0' (non-fraud) representing approximately 90.64% and 'class 1' (fraud) representing approximately 9.36%.

    Preprocessing:

        Numerical features were scaled using StandardScaler.

        Categorical features were One-Hot Encoded using OneHotEncoder (sparse output for efficiency).

    Train-Test Split: Data was split into training and testing sets with a test_size=0.2 and stratify=y to maintain class proportions.

    Class Imbalance Handling: SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training set to balance the classes before model training.

2.2. Credit Card Fraud Data (creditcard.csv)

    Target Variable: 'Class'

    Features: All columns except 'Class' were used as features.

    Class Distribution: Extremely imbalanced, with 'Class 0' (non-fraud) representing approximately 99.83% and 'Class 1' (fraud) representing approximately 0.17%.

    Preprocessing: All numerical features were scaled using StandardScaler.

    Train-Test Split: Data was split into training and testing sets with a test_size=0.2 and stratify=y.

    Class Imbalance Handling: SMOTE was applied to the training set to balance the classes before model training.

3. Model Selection and Training

Two models were selected for comparison:

    Logistic Regression: Chosen as a simple and interpretable baseline.

    XGBoost (Extreme Gradient Boosting): Chosen as a powerful ensemble model, known for its performance on structured data.

3.1. Logistic Regression

    Configuration: max_iter=1000, class_weight='balanced', n_jobs=-1 (for Fraud Data, class_weight='balanced' was implicitly handled by SMOTE, but explicitly set for Credit Card dataset).

    Training: Trained on the preprocessed and SMOTE-resampled training data.

3.2. XGBoost

    Configuration:

        objective='binary:logistic'

        eval_metric='aucpr'

        scale_pos_weight: Set to balance the positive and negative classes.

        Hyperparameter tuning was performed for the Credit Card Fraud dataset across max_depth, eta, subsample, and colsample_bytree.

    Training: Trained on the preprocessed and SMOTE-resampled training data.

4. Model Evaluation and Comparison

Evaluation was performed using F1-Score and Average Precision (AUC-PR), which are critical metrics for imbalanced datasets as they focus on the performance of the minority class. Confusion matrices and full classification reports were also generated.

4.1. Fraud Data (processed_fraud_data.csv) Results
Model	F1 Score (Fraud Class)	AUC-PR (Fraud Class)
Logistic Regression	0.6593	0.6547
XGBoost	0.6989	0.7173

Justification for Best Model (Fraud Data):
XGBoost performed better than Logistic Regression on both F1-Score (0.6989 vs. 0.6593) and AUC-PR (0.7173 vs. 0.6547). This indicates that XGBoost is more effective at identifying fraudulent transactions while maintaining a good balance between precision and recall, making it the preferred model for this dataset.

4.2. Credit Card Fraud Data (creditcard.csv) Results
Model	F1 Score (Fraud Class)	AUC-PR (Fraud Class)	Optimal Threshold (XGBoost)
Logistic Regression	0.1088	0.7245	N/A
XGBoost (Tuned)	0.8444	0.8426	0.9790

Justification for Best Model (Credit Card Data):
XGBoost, particularly after hyperparameter and threshold tuning, significantly outperformed Logistic Regression on this highly imbalanced dataset. XGBoost achieved a much higher F1-Score (0.8444 vs. 0.1088) and AUC-PR (0.8426 vs. 0.7245). The optimal threshold of 0.9790 further improved its ability to maximize F1-score. Therefore, the tuned XGBoost model is clearly the preferred choice for detecting credit card fraud.

5. Conclusion 

For both fraud detection tasks, XGBoost proved to be the superior model compared to Logistic Regression, exhibiting higher F1-Scores and AUC-PR values. This demonstrates its effectiveness in handling imbalanced datasets and complex fraud patterns.

The trained models and their respective preprocessors (or scalers) have been saved for future use:

    preprocessor.pkl and logistic_regression_pipeline.pkl for general fraud data.

    xgboost_model.json for general fraud data.

    xgboost_credit_model_xgb.json and scaler_credit.pkl for credit card fraud data.

### 8. Model Explainability (Task 3)

    Global Explainability with SHAP Summary Plot:

        Used SHAP (SHapley Additive exPlanations) to interpret the best-performing models.

        Identified key features influencing fraud predictions globally for both e-commerce and credit card datasets.

        Features like cat_country_Japan, device_id, and browser had the highest impact in the fraud dataset.

        For credit card data, V14, V10, and V12 were the most influential contributors to fraud detection.

    Local Explainability with SHAP Force Plot:

        Generated force plots for individual predictions to understand the contribution of each feature to the final model output.

        Visualized how specific features push predictions toward fraud or non-fraud.

        Clear separation observed between features that increase fraud likelihood and those that reduce it.

    Insights:

        SHAP analysis validated the relevance of time-related and categorical features.

        Helped establish trust and transparency in model predictions by visually communicating decision logic.

### 🔁 Reproducibility & Code Structure

    Project developed in modular Jupyter Notebooks:

        1_eda.ipynb – Data Cleaning, EDA, Feature Engineering

        2_model_training.ipynb – Modeling, Evaluation, Explainability

        3_model_explainability.ipynb- Model Explainability

    Checkpoints and transformed data saved using joblib and pickle for reproducibility.

    All dependencies listed in requirements.txt.

## 📁 Folder Structure

    fraud_detection_project/
        │
        ├── data/ # Contains raw and cleaned data files
        ├── docs/ # Contains detailed reports
        ├── notebooks
                 ├──1_eda.ipynb # Main Jupyter notebook for Task 1
                 ├──2_model_training.ipynb – Modeling, Evaluation, Explainability
                 └──3_model_explainability.ipynb- Model Explainability
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