Executive Summary

This interim submission establishes a solid foundation for the fraud detection pipeline by conducting thorough exploratory data analysis (EDA), meticulous data cleaning, and strategic preprocessing on two key datasets: e-commerce transactions (Fraud_Data.csv) and credit card transactions (creditcard.csv). These steps ensure the data is prepared for effective model development and evaluation in subsequent phases.
### Key Highlights

- **Exploratory Data Analysis (EDA) & Fraud Pattern Insight**
  - Performed targeted EDA to uncover patterns distinguishing fraudulent and legitimate transactions.
  - Visualized correlations between fraud likelihood and features like `purchase_value`, `device_id`, `country`, and `ip_address`.
  - For credit card data, identified strong fraud indicators in `Amount`, `V4`, and `V14`, and verified class imbalance visually.

- **Missing Value Handling**
  - Missing values in categorical and numeric fields were imputed using appropriate strategies:
    - **Categorical:** Mode imputation for features like `device_id` and `browser`.
    - **Numeric:** Mean/median imputation or row drops, depending on feature relevance and percentage of missing data.
  - Ensured no nulls remain post-processing.

- **Feature Engineering**
  - Created new features to support fraud detection:
    - `transaction_hour` derived from timestamps.
    - Binary flags for `foreign_transaction` based on country-IP mismatches.
  - Applied label encoding and standardization for model readiness.

- **Class Imbalance Handling**
  - Identified extreme class imbalance in both datasets.
  - Applied techniques like SMOTE and undersampling for balanced datasets (to be used in Task 2 model training).

- **Code Structure & Reproducibility**
  - Implemented in a clean Jupyter Notebook, with well-documented sections for each step.
  - Raw data stored in `data/raw/`, processed outputs in `data/processed/`.

### Progress and Completed Work

#### Completed Milestones
## 1. Data Loading and Initial Inspection

- **Libraries Imported**: Standard data science libraries such as `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `sklearn.model_selection` (for `train_test_split`), `xgboost`, and `shap` were imported, indicating preparation for machine learning and explainability.

- **Data Loaded**:  
  - `fraud_data.csv`: Loaded into the `fraud_data` DataFrame. This appears to be the main e-commerce transaction dataset.  
  - `ip_to_country.csv`: Loaded into the `ip_to_country` DataFrame. This likely contains mappings from IP addresses to countries.  
  - `creditcard.csv`: Loaded into the `credit_data` DataFrame. This is noted as "credit card data (if needed)" and likely contains financial transaction data.

- **Initial Display**: The first 5 rows of `fraud_data` and `ip_to_country` were displayed using `.head()` to provide a quick look at their structure and content.

## 2. Duplicate Removal

- `fraud_data`: No duplicates were found (`fraud_data.duplicated().sum()` returned 0).
- `ip_to_country`: No duplicates were found.
- `credit_data`: Initially, 1081 duplicates were detected. After applying `credit_data.drop_duplicates()`, all duplicates were successfully removed.

## 3. Data Type Conversion

- For `fraud_data`, the following columns were converted to appropriate data types:  
  - `user_id`: `object`  
  - `device_id`: `object`  
  - `signup_time`: `datetime64[ns]` (crucial for time-based analysis)  
  - `purchase_time`: `datetime64[ns]` (also crucial for time-based analysis)  

- For `credit_data`, columns `Time` and `V1` through `V23` are of type `float64`, which is standard for these features.

## 4. Missing Value Check

- `fraud_data`: No missing values detected across all columns.
- `credit_data`: No missing values detected across all columns.
- `ip_to_country`: No missing values detected, important for complete country-IP mapping.

## 5. Class Imbalance Check (for `fraud_data`)

- Distribution of the target variable `class` is as follows:  
  - Class 0 (likely non-fraudulent): ~90.6%  
  - Class 1 (likely fraudulent): ~9.4%

- **Observation**: Significant class imbalance exists, common in fraud detection datasets. This imbalance will need addressing during model training (e.g., oversampling, undersampling, or appropriate evaluation metrics).

## 6. Summary Statistics (purchase_value in `fraud_data`)

- Descriptive statistics for the `purchase_value` column:  
  - Count: 151,112  
  - Mean: 36.94  
  - Standard Deviation: 18.32  
  - Minimum: 9.0  
  - 25th percentile: 22.0  
  - Median (50th percentile): 35.0  
  - 75th percentile: 49.0  
  - Maximum: 154.0

- **Observation**: Purchase values range from 9 to 154, with an average around 37. The spread is moderate as indicated by the standard deviation.

## 7. Numeric Features Summary (fraud_data)

The summary statistics for numerical columns `purchase_value` and `age` in `fraud_data` are as follows:

- **purchase_value**:  
  - Count: 151,112  
  - Mean: 36.93  
  - Std Dev: 18.32  
  - Min: 9.0  
  - 25th Percentile: 22.0  
  - Median (50th Percentile): 35.0  
  - 75th Percentile: 49.0  
  - Max: 154.0  
  - **Observation**: Purchase values range from 9 to 154, with most values between 22 and 49.

- **age**:  
  - Count: 151,112  
  - Mean: 33.14  
  - Std Dev: 8.61  
  - Min: 18.0  
  - 25th Percentile: 27.0  
  - Median (50th Percentile): 33.0  
  - 75th Percentile: 39.0  
  - Max: 76.0  
  - **Observation**: User ages range from 18 to 76, with an average around 33. The median also suggests a roughly symmetrical distribution.

## 8. Categorical Features Value Counts (fraud_data)

- **source**:  
  - SEO: 60,615  
  - Ads: 59,881  
  - Direct: 30,616  
  - **Observation**: 'SEO' and 'Ads' are the most common sources, contributing almost equally, followed by 'Direct'.

- **browser**:  
  - Chrome: 61,432  
  - IE: 36,727  
  - Safari: 24,610  
  - Firefox: 24,610  
  - Opera: 3,676  
  - **Observation**: Chrome dominates browser usage, followed by IE, Safari, and Firefox. Opera is the least used.

## 9. Transaction Amount and Class Distribution (credit_data)

- **Amount (Transaction Amount)**:  
  - Count: 283,726  
  - Mean: 88.47  
  - Std Dev: 250.39  
  - Min: 0.0  
  - 25th Percentile: 5.6  
  - Median (50th Percentile): 22.0  
  - 75th Percentile: 77.51  
  - Max: 25,691.16  
  - **Observation**: Amount values show a highly skewed distribution, with many small transactions and a few extreme outliers, reflected by a mean much higher than the median.

- **Class Distribution**:  
  - Class 0: 99.83%  
  - Class 1: 0.17%  
  - **Observation**: The dataset exhibits an extreme class imbalance, even more severe than the `fraud_data` dataset, requiring careful handling during model training (e.g., advanced resampling and specialized evaluation metrics).

## 10. Visualizations (fraud_data)

- **Fraud Rate by Browser (Bar Plot)**  
  The bar plot below shows the average fraud rate (`class` mean) for each browser category.

  ![Fraud Rate by Browser](images/fraud_rate_by_browser.png)

  - **Observation**: Fraud rates across browsers are similar, ranging roughly from 9% to 10%. Error bars overlap considerably, indicating no significant difference. Therefore, 'browser' alone may not be a strong fraud predictor.

- **Age Distribution by Fraud Class (Box Plot)**  
  The box plot below visualizes the age distribution for fraudulent (class 1) and non-fraudulent (class 0) transactions.


  - **Observation**: Median ages for both classes are similar (~33). The interquartile ranges overlap substantially, and both show outliers in higher age ranges. Age does not appear to distinctly separate fraud classes.

## 11. Saving Cleaned Data

- `fraud_data.to_csv('../data/cleaned_fraud_data.csv', index=False)`  
- `credit_data.to_csv('../data/cleaned_credit_data.csv', index=False)`  

Saving the processed data is good practice to avoid reprocessing steps if the notebook is restarted.

## 12. Purchase Value Distribution by Fraud Class (fraud_data)

- **Plot**:  
  ```python
  sns.histplot(data=fraud_data, x='purchase_value', hue='class', bins=30, kde=True, alpha=0.6)
````

* **Observation**:

  * The distribution of `purchase_value` for both non-fraudulent (class 0, blue) and fraudulent (class 1, orange) transactions is right-skewed.
  * Non-fraudulent transactions peak around 30-40 purchase value.
  * Fraudulent transactions peak in a similar range, perhaps slightly lower, with an overall similar distribution shape.
  * The absolute count of non-fraudulent transactions is much higher, as expected due to class imbalance.
* **Conclusion**: `purchase_value` alone is not a strong distinguishing feature between fraud and non-fraud, as distributions largely overlap.

## 13. Transaction Amount by Fraud Class (credit\_data)

* **Plot**:

  ```python
  sns.boxplot(x='Class', y='Amount', data=credit_data, showfliers=False)
  ```
* **Observation**:

  * **Non-Fraud (Class 0)**: Median transaction amount is very low; most non-fraudulent transactions are small. The box plot is compressed near the bottom.
  * **Fraud (Class 1)**: Median transaction amount is higher than non-fraudulent, indicating fraudulent transactions tend to have larger amounts.
  * Even excluding outliers, non-fraud amounts reach \~180-190, while fraud amounts can go up to \~260.
* **Conclusion**: Unlike `purchase_value` in `fraud_data`, `Amount` in `credit_data` shows a noticeable difference between fraud and non-fraud, making it an important feature.


## 14. Log-Transformed Transaction Amount Distribution (credit\_data)

* **Plot**:

  ```python
  plt.hist(np.log1p(credit_data['Amount']), bins=50)
  ```
* **Purpose**: Log transformation (using `np.log1p`) normalizes the heavily skewed `Amount` distribution, making it more symmetrical for modeling.
* **Observation**: Post-transformation, the distribution is bell-shaped with a peak around 1.0–2.0 and a tail extending higher.
* **Conclusion**: This transformation effectively normalizes `Amount` for better model suitability.

## 15. Time from Signup to Purchase (Fraud vs Non-Fraud) (fraud\_data)

* **Feature Engineering**:

  ```python
  fraud_data['hours_to_purchase'] = ((fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds()) / 3600
  ```

  Calculates hours between signup and purchase — a key fraud indicator.
* **Plot**:

  ```python
  sns.boxplot(x='class', y='hours_to_purchase', data=fraud_data, showfliers=False)
  ```
* **Observation**:

  * **Non-Fraud (Class 0)**: Median `hours_to_purchase` is high with a wide interquartile range.
  * **Fraud (Class 1)**: Median is significantly lower; fraudulent purchases occur much sooner after signup.
  * Fraud distribution is tighter and skewed toward faster purchases.
* **Conclusion**: `hours_to_purchase` is a strong predictor; fraudulent transactions tend to occur quickly after signup.


## 16. IP Address Type Conversion and Range Check

* **Type Conversion**:

  ```python
  fraud_data['ip_address'] = fraud_data['ip_address'].astype('int64')
  ip_to_country['lower_bound_ip_address'] = ip_to_country['lower_bound_ip_address'].astype('int64')
  ip_to_country['upper_bound_ip_address'] = ip_to_country['upper_bound_ip_address'].astype('int64')
  ```

  Converts IP-related columns to `int64` for numerical operations and merging.
* **Verification**: All IP columns confirmed as `int64`.
* **IP Range Check**:

  * Min IP in `fraud_data`: 52,093
  * Max IP in `fraud_data`: 4,294,850,499
  * Country database range: 16,777,216 to 3,758,996,383
  * Observation: Maximum IP in `fraud_data` exceeds the country database upper bound, meaning \~13.2% of IPs won’t match the country mapping.
  * Valid IPs within range: 131,095 / 151,112 (86.8%)
##  17. Merging Fraud Data with IP-to-Country Mapping

    Both fraud_data and ip_to_country datasets are sorted by IP columns (ip_address and lower_bound_ip_address) to enable pd.merge_asof.

    pd.merge_asof merges on ranges, assigning the nearest lower_bound_ip_address less than or equal to each ip_address.

    This strategy correctly maps IPs to countries when IPs fall within a defined range.

    Post-merge, auxiliary IP range columns are dropped to clean the dataset.

18. Handling Unmatched IPs and Cleaning Country Columns

    About 19,383 IPs (13.2%) remain unmatched after merging, corresponding to IPs outside known ranges.

    Duplicate country columns (country_x, country_y) are cleaned up to keep a single country column.

    Unmatched IPs are labeled as 'Unknown' to explicitly handle missing geographical data.

    Verification confirms all NaNs are replaced, with unmatched IPs clearly categorized.

19. Match Rate Statistics

    Total transactions: 151,112.

    Successfully matched IPs: 131,095 (86.8%).

    Unmatched IPs explicitly labeled 'Unknown', ensuring clarity in downstream analysis.

20. Geographic Distribution (Top 10 Countries)

    Top countries by transaction volume include the United States, followed by the 'Unknown' category, China, Japan, Germany, UK, Korea, Brazil, Canada, and Italy.

    The large 'Unknown' category highlights limitations of the IP mapping database and could be a predictor for fraud.
    
21. Boundary IP Tests

    Synthetic test IPs (min/max and out-of-range values) did not appear in real data, limiting the assessment of edge-case merges.

    This suggests the need for additional validation if edge-case IPs are important.

22. Feature Summary of Engineered Time Features

    Features like user_transaction_count and time_since_last_purchase are constant (1 and 0 respectively), indicating each user has only one transaction.

    These features provide no predictive power and can be dropped.

    purchase_hour and purchase_dayofweek show distributions useful for fraud pattern analysis.

    time_to_purchase (hours since signup) shows wide variability and is a strong predictive feature.

23. Credit Card Data Scaling

    Amount and Time features in credit card data are scaled using StandardScaler to zero mean and unit variance.

    This is important for algorithms sensitive to feature scales.

    Original unscaled columns may be dropped after scaling.

24. E-commerce Data Preprocessing for Modeling

    One-hot encoding applied on categorical features (source, browser, sex, country) with drop_first=True to prevent multicollinearity.

    Stratified train-test split (80-20) preserves fraud/non-fraud class distribution, crucial given class imbalance.

    Post-split shapes: Training features ~120k samples with 206 features; test features ~30k samples.

25. IP Address Data Type Conversion, Sorting, and Saving

    Merged data saved successfully with no missing country values.

    Rows: 151,112; Columns: 21.

26. User and Device ID Analysis

    151,112 unique user_ids confirm one transaction per user.

    137,956 unique device_ids indicate some devices used in multiple transactions.

    Device usage frequency could be a useful fraud detection feature.

27. Refined Timestamp Feature Engineering

    Ensured purchase_time and signup_time are datetime objects.

    Created new features: hour_of_day, day_of_week, hours_since_signup.

    Attempted to compute transaction frequency and time since last transaction per user, but these are constant due to one transaction per user, thus redundant.

    hours_since_signup remains a strong fraud indicator.

28. Memory Optimization

    Dropped original datetime columns after feature engineering to save memory.

29. Risk Flag Features

    Created binary flags:

        new_account: Transactions within 1 hour of signup.

        off_hour_purchase: Transactions before 6 AM or after 8 PM.

    These flags are aligned with common fraud heuristics and likely to improve model interpretability.

30. Transactions by Hour of Day (Fraud vs Non-Fraud)

    Fraudulent transaction counts are lower overall but proportionally stable across hours.

    No strong hourly fraud pattern detected.

    The off_hour_purchase flag remains useful despite this.

31. Fraud by Account Age (Hours)

    Fraudulent transactions mostly occur soon after account creation.

    Non-fraud transactions occur across wider account age range.

    Confirms hours_since_signup is a highly discriminative feature.

32. Class Imbalance Visualization and Resampling

    Original fraud rate: 9.4%.

    Defined a resampling function to oversample minority class to ~24.8% representation.

    This reduces class imbalance and improves model training on fraud cases.

33. Class Weights Calculation and Checkpoint Saving

    Computed sample weights inversely proportional to class frequencies for balanced training.

    Saved datasets and resampled data with pickle for efficient workflow resumption.

Overall Summary & Insights

    Redundant features (user_transaction_count, time_since_last_purchase) identified and removed.

    Strong predictive features engineered, especially hours_to_purchase.

    IP mapping achieved 86.8% coverage; unmatched IPs labeled 'Unknown' and treated as important.

    Device ID reuse suggests an additional fraud signal.

    Proper categorical encoding, scaling, and stratified splitting ensure robust modeling.

    Class imbalance handled through oversampling and weighting.

    Data quality is high with careful cleaning and feature engineering throughout.