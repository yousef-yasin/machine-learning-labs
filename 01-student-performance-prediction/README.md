# 🎓 Student Performance Prediction

## 📌 Project Overview

This project uses Machine Learning to predict student academic performance based on various educational, social, and personal factors.

A Decision Tree Classifier was trained to classify students into performance categories using real student data.

---

## 🎯 Objective

The goal of this project is to analyze student-related factors and predict the final academic performance class (GradeClass).

---

## 📊 Dataset Features

The dataset includes:

* Age
* Gender
* Ethnicity
* Parental Education
* Study Time Weekly
* Absences
* Tutoring
* Parental Support
* Extracurricular Activities
* Sports Participation
* Music Participation
* Volunteering
* GPA

### Target Variable

* GradeClass

---

## 🛠 Technologies Used

* Python
* Pandas
* Scikit-Learn
* Matplotlib

---

## 🤖 Machine Learning Workflow

### 1. Load Dataset

The dataset was loaded using Pandas.

### 2. Data Exploration

The dataset structure and columns were examined.

### 3. Feature Selection

Features (X) and target variable (y) were separated.

### 4. Train-Test Split

The dataset was divided into:

* Training Set: 1913 records
* Testing Set: 479 records

### 5. Model Training

A Decision Tree Classifier was used:

```python
model = DecisionTreeClassifier(random_state=42)
```

### 6. Prediction

The trained model predicted student performance on unseen test data.

### 7. Evaluation

The model was evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report

---

## 📈 Results

### Model Accuracy

**93% Accuracy**

This means the model correctly classified approximately 93% of the students in the testing dataset.

### Classification Performance

| Metric            | Score |
| ----------------- | ----- |
| Accuracy          | 0.93  |
| Weighted F1-Score | 0.93  |

---

## 📷 Visualization

The project includes an Accuracy Chart generated using Matplotlib.

---

## 📂 Project Structure

```
01-student-performance-prediction/
│
├── README.md
├── main.py
├── requirements.txt
└── Student_performance_data_.csv
```

---

## 🚀 Future Improvements

* Compare multiple machine learning algorithms.
* Hyperparameter tuning.
* Feature importance analysis.
* Interactive dashboard for predictions.

---

## 👨‍💻 Author

Yousef Yasin
