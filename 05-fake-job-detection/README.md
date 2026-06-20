# Fake Job Detection Using Machine Learning

## Overview

This project uses Natural Language Processing (NLP) and Machine Learning to detect fraudulent job postings.

The model analyzes job advertisements and predicts whether a job posting is legitimate or fraudulent.

This project introduces text processing techniques and demonstrates how machine learning can be applied to real-world cybersecurity and recruitment challenges.

---

## Dataset

Dataset: Fake Job Postings Dataset

Total Records:

```text
17,880 Job Postings
```

Target Variable:

```text
fraudulent
```

Classes:

```text
0 = Real Job Posting
1 = Fake Job Posting
```

---

## Features Used

The project combines multiple text fields into a single text feature:

* title
* company_profile
* description
* requirements

Example:

```text
Python Developer

We are a software company...

Requirements:
Python, SQL, Django
```

---

## Data Preprocessing

### Missing Values Handling

Missing values were replaced with empty strings:

```python
df = df.fillna("")
```

### Text Feature Creation

Multiple text columns were merged into one feature:

```python
df["text"] = (
    df["title"] +
    " " +
    df["company_profile"] +
    " " +
    df["description"] +
    " " +
    df["requirements"]
)
```

### Feature and Target Selection

```python
X = df["text"]
y = df["fraudulent"]
```

---

## TF-IDF Vectorization

Text data cannot be processed directly by machine learning models.

TF-IDF (Term Frequency – Inverse Document Frequency) was used to convert text into numerical features.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

Configuration:

```python
TfidfVectorizer(
    stop_words="english",
    max_features=5000
)
```

Result:

```text
Training Matrix: (14304, 5000)
Testing Matrix: (3576, 5000)
```

---

## Train-Test Split

Dataset Split:

```text
80% Training Data
20% Testing Data
```

Training Samples:

```text
14,304
```

Testing Samples:

```text
3,576
```

---

# Models Used

## Logistic Regression

Accuracy:

```text
97.18%
```

Confusion Matrix:

```text
[[3395    0]
 [ 101   80]]
```

Classification Performance:

```text
Precision (Fake Jobs): 1.00
Recall (Fake Jobs): 0.44
F1-Score (Fake Jobs): 0.61
```

---

## Random Forest Classifier

Accuracy:

```text
98.07%
```

Confusion Matrix:

```text
[[3394    1]
 [  68  113]]
```

Classification Performance:

```text
Precision (Fake Jobs): 0.99
Recall (Fake Jobs): 0.62
F1-Score (Fake Jobs): 0.77
```

---

## Model Comparison

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 97.18%   |
| Random Forest       | 98.07%   |

Best Model:

```text
Random Forest Classifier
```

---

## Evaluation Metrics

The following metrics were used:

* Accuracy Score
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## Visualizations

The project includes:

* Accuracy Comparison Chart
* Confusion Matrix Heatmap

---

## Custom Prediction Example

Example Input:

```text
Work from home.
Earn $5000 per week.
No experience required.
Immediate hiring.
```

Model Output:

```text
Fake Job
```

---

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* Matplotlib
* Seaborn
* NLP
* TF-IDF

---

## Project Structure

```text
05-fake-job-detection
│
├── fake_job_postings.csv
├── main.py
├── README.md
└── requirements.txt
```

---

## Future Improvements

* XGBoost Implementation
* Hyperparameter Tuning
* Streamlit Web Application
* Real-Time Job Screening Tool
* Deep Learning Models (LSTM, BERT)

---

## Author

Yousef Yasin

Machine Learning Labs Collection
