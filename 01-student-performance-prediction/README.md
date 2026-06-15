# Student Performance Prediction

## Overview

This project predicts student academic performance using Machine Learning.

A Decision Tree Classifier was trained on student-related data such as study time, absences, parental support, extracurricular activities, and GPA to predict the student's final grade class.

---

## Dataset

The dataset contains information about students including:

- Age
- Gender
- Ethnicity
- Parental Education
- Study Time Weekly
- Absences
- Tutoring
- Parental Support
- Extracurricular Activities
- Sports
- Music
- Volunteering
- GPA

Target Variable:

- GradeClass

---

## Technologies Used

- Python
- Pandas
- Scikit-Learn

---

## Machine Learning Workflow

1. Load Dataset
2. Data Exploration
3. Feature Selection
4. Train/Test Split
5. Decision Tree Training
6. Prediction
7. Model Evaluation

---

## Model

Algorithm:

- Decision Tree Classifier

```python
model = DecisionTreeClassifier(random_state=42)
```

---

## Results

Dataset Size:

- 2392 Students

Train Set:

- 1913 Records

Test Set:

- 479 Records

Model Accuracy:

- 93%

---

## Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report

Example Metrics:

- Accuracy: 0.93
- Weighted F1-Score: 0.93

---

## Project Structure

```
01-student-performance-prediction/
│
├── README.md
├── main.py
├── requirements.txt
└── Student_performance_data_.csv
```

---

## Author

Yousef Yasin

