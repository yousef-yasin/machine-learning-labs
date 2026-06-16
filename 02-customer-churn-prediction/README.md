# Telco Customer Churn Prediction

## Overview

This project predicts whether a customer is likely to leave a telecommunications company (Customer Churn) using Machine Learning techniques.

The goal is to analyze customer information and identify customers who may cancel their services in the future.

---

## Dataset

Dataset: Telco Customer Churn Dataset

Number of Records: 7043 customers

Target Variable:

- Churn
  - 0 = Customer stays
  - 1 = Customer leaves

Features include:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

---

## Data Preprocessing

The following preprocessing steps were applied:

### 1. Data Exploration

- Displayed dataset information
- Checked data types
- Checked missing values

### 2. Handling Missing Values

The `TotalCharges` column contained non-numeric values.

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
```

Rows containing missing values were removed:

```python
df = df.dropna()
```

Dataset size changed from:

```text
7043 rows
```

to:

```text
7032 rows
```

---

### 3. Removing Unnecessary Features

The `customerID` column was removed because it does not contribute to predicting customer churn.

```python
df = df.drop("customerID", axis=1)
```

---

### 4. Encoding Categorical Variables

Categorical columns were converted into numerical values using Label Encoding.

```python
from sklearn.preprocessing import LabelEncoder
```

---

## Feature and Target Separation

Features:

```python
X = df.drop("Churn", axis=1)
```

Target:

```python
y = df["Churn"]
```

---

## Train-Test Split

The dataset was split into:

- 80% Training Data
- 20% Testing Data

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Training Set:

```text
(5625, 19)
```

Testing Set:

```text
(1407, 19)
```

---

# Machine Learning Models

## 1. Decision Tree Classifier

```python
from sklearn.tree import DecisionTreeClassifier
```

### Results

Accuracy:

```text
72.49%
```

Confusion Matrix:

```text
[[826 207]
 [180 194]]
```

---

## 2. Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier
```

### Results

Accuracy:

```text
79.25%
```

Confusion Matrix:

```text
[[932 101]
 [191 183]]
```

---

## Model Comparison

| Model | Accuracy |
|---------|---------|
| Decision Tree | 72.49% |
| Random Forest | 79.25% |

Best Model:

```text
Random Forest Classifier
```

---

## Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Libraries:

```python
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
```

---

## Technologies Used

- Python
- Pandas
- Scikit-Learn
- NumPy
- Matplotlib

---

## Project Structure

```text
02-customer-churn-prediction
│
├── Telco-Customer-Churn.csv
├── main.py
├── requirements.txt
└── README.md
```

---

## Conclusion

This project demonstrates a complete machine learning workflow including:

- Data Exploration
- Data Cleaning
- Feature Engineering
- Label Encoding
- Model Training
- Model Evaluation
- Model Comparison

The Random Forest model achieved the best performance with an accuracy of approximately 79.25%.