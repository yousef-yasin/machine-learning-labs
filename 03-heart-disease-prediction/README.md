# ❤️ Heart Disease Prediction

## Overview

This project predicts whether a patient is likely to have heart disease using Machine Learning techniques.

The dataset contains medical information such as age, cholesterol level, blood pressure, heart rate, and other health indicators.

The goal is to build predictive models that can help identify potential heart disease cases based on patient data.

---

## Dataset

Dataset: Heart Disease UCI Dataset

Features include:

- Age
- Sex
- Chest Pain Type (cp)
- Resting Blood Pressure (trestbps)
- Cholesterol (chol)
- Fasting Blood Sugar (fbs)
- Resting ECG Results (restecg)
- Maximum Heart Rate Achieved (thalach)
- Exercise Induced Angina (exang)
- ST Depression (oldpeak)
- Slope
- Number of Major Vessels (ca)
- Thalassemia (thal)

Target:

- 0 = No Heart Disease
- 1 = Heart Disease

---

## Data Preprocessing

The following preprocessing steps were performed:

- Loaded the dataset using Pandas
- Checked for missing values
- Separated features and target variable
- Split the dataset into training and testing sets
- Prepared data for machine learning models

---

## Models Used

### Logistic Regression

A linear classification algorithm used as a baseline model.

Result:

- Accuracy: 79.5%

---

### Random Forest Classifier

An ensemble machine learning algorithm that combines multiple decision trees to improve prediction performance.

Result:

- Accuracy: 98.5%

---

## Model Evaluation

Evaluation metrics used:

- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1-Score

---

## Random Forest Results

Accuracy:

```text
98.5%
```

Confusion Matrix:

```text
[[102   0]
 [  3 100]]
```

Classification Report:

```text
Precision: 0.99
Recall: 0.99
F1-Score: 0.99
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

---

## Project Structure

```text
03-heart-disease-prediction/
│
├── heart.csv
├── main.py
├── requirements.txt
└── README.md
```

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

## Author

Yousef Yasin

Machine Learning Labs Collection
