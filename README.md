# Fraud Detection Model

This project aims to detect fraudulent transactions using machine learning techniques. Two models, Random Forest and XGBoost, were trained and evaluated on a highly imbalanced dataset. The goal is to accurately predict fraudulent transactions while minimizing false positives and false negatives.

## Table of Contents

- [Dataset Description](#dataset-description)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Outliers Handling](#outliers-handling)
- [Class Balancing](#class-balancing)
- [Precision-Recall Trade-off](#precision-recall-trade-off)
- [Confusion Matrix Analysis](#confusion-matrix-analysis)
- [Future Improvements](#future-improvements)
- [How to Run](#how-to-run)

## Dataset Description

The dataset contains 6,362,620 transactions and includes the following columns:

| Column            | Description                                     |
|-------------------|-------------------------------------------------|
| `step`            | Unit of time at which the transaction occurred  |
| `type`            | Type of transaction (e.g., PAYMENT, TRANSFER)   |
| `amount`          | Amount of the transaction                       |
| `nameOrig`        | Customer ID for the origin account              |
| `oldbalanceOrg`   | Initial balance of the origin account           |
| `newbalanceOrig`  | Balance of the origin account after transaction |
| `nameDest`        | Customer ID for the destination account         |
| `oldbalanceDest`  | Initial balance of the destination account      |
| `newbalanceDest`  | Balance of the destination account after transaction |
| `isFraud`         | Whether the transaction was fraudulent (1) or not (0) |
| `isFlaggedFraud`  | Flag for large fraud attempts (1 if flagged, 0 otherwise) |

## Preprocessing

### Steps Taken:
1. **Handling Outliers:**
   - Outliers in the `amount` column were detected using the Interquartile Range (IQR) method.
   - Outliers were handled by capping values below the lower bound and above the upper bound.
   
   ```python
   Q1 = df['amount'].quantile(0.25)
   Q3 = df['amount'].quantile(0.75)
   IQR = Q3 - Q1
   lower_bound = Q1 - 1.5 * IQR
   upper_bound = Q3 + 1.5 * IQR
   df['amount'] = np.where(df['amount'] < lower_bound, lower_bound, df['amount'])
   df['amount'] = np.where(df['amount'] > upper_bound, upper_bound, df['amount'])
   ```

2. **Feature Engineering:**
   - **Transaction Type Encoding:** Categorical `type` column was one-hot encoded.
   - **Merchant Flagging:** A flag was added to identify if the destination account was a merchant account.
   - **Balance Discrepancy:** New columns were created to capture discrepancies between old and new balances for both origin and destination accounts.
   - Unnecessary columns like `nameOrig` and `nameDest` were dropped.

   ```python
   df = pd.get_dummies(df, columns=['type'])
   df['isMerchantDest'] = df['nameDest'].apply(lambda x: 1 if x.startswith('M') else 0)
   df['origBalanceDiscrepancy'] = df['oldbalanceOrg'] - df['newbalanceOrig']
   df['destBalanceDiscrepancy'] = df['oldbalanceDest'] - df['newbalanceDest']
   df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)
   ```

3. **Train-Test Split:**
   - The dataset was split into training (80%) and testing (20%) sets with stratification based on the target variable (`isFraud`).

   ```python
   from sklearn.model_selection import train_test_split
   X = df.drop('isFraud', axis=1)
   y = df['isFraud']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
   ```

## Modeling

Two machine learning models were used for fraud detection:

1. **RandomForestClassifier:**
   - 100 estimators
   - Balanced class weights to handle the imbalanced dataset

   ```python
   from sklearn.ensemble import RandomForestClassifier
   rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
   rf.fit(X_train, y_train)
   ```

2. **XGBoostClassifier:**
   - 100 estimators
   - Learning rate of 0.1

   ```python
   from xgboost import XGBClassifier
   xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
   xgb.fit(X_train, y_train)
   ```

## Evaluation

### Classification Reports:

#### RandomForest:
```plaintext
              precision    recall  f1-score   support
           0       1.00      1.00      1.00   1270881
           1       0.97      0.80      0.88      1643
    accuracy                           1.00   1272524
   macro avg       0.99      0.90      0.94   1272524
weighted avg       1.00      1.00      1.00   1272524
```

#### XGBoost:
```plaintext
              precision    recall  f1-score   support
           0       1.00      1.00      1.00   1270881
           1       0.97      0.80      0.88      1643
    accuracy                           1.00   1272524
   macro avg       0.99      0.90      0.94   1272524
weighted avg       1.00      1.00      1.00   1272524
```

## Outliers Handling

The IQR method was applied to detect and cap outliers in the `amount` column, ensuring that extreme values do not skew the model.

## Class Balancing

Due to the imbalanced nature of the dataset, class balancing was handled in two ways:

1. **Balanced Class Weights:** Applied in the RandomForestClassifier to assign higher weight to minority class (fraudulent transactions).
2. **Stratified Sampling:** Ensured that both training and test sets maintained the same proportion of fraud cases as the original dataset.

## Precision-Recall Trade-off

The Precision-Recall curve was plotted to evaluate the trade-off between precision and recall, particularly for the minority class (`isFraud`).

```python
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, rf.predict_proba(X_test)[:,1])
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
```

## Confusion Matrix Analysis

### RandomForest:
- True Negatives (TN): High accuracy in identifying legitimate transactions.
- False Positives (FP): Few legitimate transactions were incorrectly flagged as fraud.
- False Negatives (FN): A moderate number of fraudulent transactions were missed, indicating an area for improvement.
- True Positives (TP): Successfully identified fraudulent cases.

### XGBoost:
- Similar to RandomForest, with slightly lower sensitivity to fraudulent cases.

## Future Improvements

1. **Class Balancing Techniques:** Applying SMOTE or undersampling the majority class could help further balance the dataset and improve recall for fraud cases.
2. **Hyperparameter Tuning:** Fine-tuning model hyperparameters, especially for XGBoost, could improve performance.
3. **Model Ensemble:** Combining multiple models may enhance the accuracy of fraud detection.

## How to Run

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   ```
   
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook:**
   - Open the Jupyter Notebook and execute the cells sequentially.
   
4. **Dataset:**
   - Place the `Fraud.csv` dataset in the appropriate directory or adjust the file path in the notebook.
