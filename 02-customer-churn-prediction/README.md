# Customer Churn Prediction

## Overview

This project predicts whether a customer will leave a telecom company (Churn) using Machine Learning.

The goal is to identify customers who are likely to cancel their services based on customer information, contract details, billing information, and service usage.

---

## Dataset

Dataset: Telco Customer Churn Dataset

Number of Records: 7,043 customers

Features include:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Online Security
- Tech Support
- Streaming Services
- Contract Type
- Payment Method
- Monthly Charges
- Total Charges

Target Variable:

- Churn
  - 0 = Customer Stays
  - 1 = Customer Leaves

---

## Data Preprocessing

The following preprocessing steps were performed:

### 1. Missing Value Handling

The `TotalCharges` column contained invalid values.

- Converted the column to numeric format.
- Replaced invalid values with NaN.
- Removed rows containing missing values.

### 2. Feature Selection

Removed:

- customerID

because it does not contribute to prediction.

### 3. Encoding

Categorical variables were converted into numerical values using:

- LabelEncoder

Example:

| Original | Encoded |
|-----------|----------|
| Yes | 1 |
| No | 0 |
| Female | 0 |
| Male | 1 |

---

## Train-Test Split

The dataset was divided into:

- Training Set: 80%
- Testing Set: 20%

Result:

Training Shape:

```python
(5625, 19)
```

Testing Shape:

```python
(1407, 19)
```

---

## Models Used

### Decision Tree Classifier

A Decision Tree model was trained and evaluated.

### Random Forest Classifier

A Random Forest model was trained and compared against the Decision Tree model.

---

## Results

### Decision Tree

Accuracy:

```text
72.49%
```

### Random Forest

Accuracy:

```text
79.25%
```

Random Forest achieved better performance than Decision Tree.

---

## Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Random Forest Confusion Matrix

| Actual / Predicted | No Churn | Churn |
|-------------------|----------|--------|
| No Churn | 932 | 101 |
| Churn | 191 | 183 |

---

## Libraries Used

```python
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Project

```bash
python main.py
```

---

## Project Structure

```text
02-customer-churn-prediction/
│
├── main.py
├── Telco-Customer-Churn.csv
├── requirements.txt
└── README.md
```

---

## Future Improvements

- Hyperparameter Tuning
- Feature Importance Analysis
- Cross Validation
- XGBoost Model
- Interactive Dashboard

---

## Author

Yousef Yasin

Machine Learning Practice Project