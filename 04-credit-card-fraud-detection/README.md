# Credit Card Fraud Detection

## Overview

This project uses Machine Learning to detect fraudulent credit card transactions. The dataset contains anonymized transaction features and a target variable indicating whether a transaction is legitimate or fraudulent.

The project demonstrates the complete machine learning workflow:

* Data loading and exploration
* Data preprocessing
* Train-test split
* Random Forest classification
* Model evaluation
* Accuracy visualization
* Confusion matrix analysis

---

## Dataset

Dataset: Credit Card Fraud Detection Dataset

Features:

* Time
* V1 - V28 (anonymized PCA features)
* Amount

Target:

* Class

  * 0 = Legitimate Transaction
  * 1 = Fraudulent Transaction

---

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* Matplotlib
* Seaborn

---

## Project Workflow

### 1. Data Exploration

The dataset is loaded using Pandas and inspected using:

```python
df.head()
df.info()
df.isnull().sum()
```

### 2. Feature Selection

The target column is separated from the features:

```python
X = df.drop("Class", axis=1)
y = df["Class"]
```

### 3. Train-Test Split

The dataset is divided into training and testing sets:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### 4. Model Training

A Random Forest classifier is trained:

```python
RandomForestClassifier(random_state=42)
```

### 5. Evaluation

The model is evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report

---

## Results

Example performance:

* Accuracy ≈ 99%

Metrics used:

* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## Visualization

The project generates:

1. Model Accuracy Bar Chart
2. Confusion Matrix Heatmap

These visualizations help evaluate model performance and fraud detection capability.

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

## Future Improvements

* Logistic Regression comparison
* XGBoost implementation
* Class imbalance handling using SMOTE
* ROC Curve visualization
* Feature importance analysis
* Hyperparameter tuning

---

## Author

Yousef Yasin
Machine Learning Lab Projects
