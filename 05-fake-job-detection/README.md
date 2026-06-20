# Fake Job Detection Using Machine Learning

## Overview

This project uses Natural Language Processing (NLP) and Machine Learning techniques to detect fraudulent job postings.

Online job platforms contain thousands of job advertisements, and some of them are fake or scam postings designed to collect personal information or deceive applicants. This project analyzes job descriptions and predicts whether a job posting is legitimate or fraudulent.

The dataset contains real and fake job advertisements with textual and categorical information. Text data is transformed into numerical features using TF-IDF Vectorization before training machine learning models.

---

## Dataset

**Dataset:** Fake Job Postings Dataset

**Target Variable:**

* `fraudulent`

  * 0 = Real Job Posting
  * 1 = Fake Job Posting

The dataset contains:

* Job title
* Company profile
* Job description
* Requirements
* Benefits
* Employment type
* Experience level
* Education level
* Industry
* Function
* Other job-related information

---

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* Matplotlib
* Seaborn
* Natural Language Processing (NLP)

---

## Machine Learning Workflow

### 1. Data Loading

The dataset is loaded using Pandas and inspected for missing values and data types.

### 2. Text Preprocessing

The job descriptions are cleaned and prepared for machine learning.

### 3. Feature Extraction

TF-IDF Vectorization converts text into numerical feature vectors.

### 4. Data Splitting

The dataset is divided into:

* 80% Training Data
* 20% Testing Data

### 5. Model Training

#### Logistic Regression

Used as a baseline text classification model.

#### Random Forest Classifier

Used to improve prediction performance and detect fraudulent job postings more accurately.

### 6. Model Evaluation

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## Results

### Logistic Regression

* Accuracy: **97.18%**
* Precision (Fraudulent Jobs): **100%**
* Recall (Fraudulent Jobs): **44%**
* F1-Score: **61%**

### Random Forest

* Accuracy: **98.07%**
* Precision (Fraudulent Jobs): **99%**
* Recall (Fraudulent Jobs): **62%**
* F1-Score: **77%**

### Best Model

**Random Forest Classifier** achieved the best overall performance and provided better detection of fraudulent job postings.

---

## Project Structure

```text
05-fake-job-detection/
│
├── main.py
├── requirements.txt
├── README.md
└── fake_job_postings.csv
```

---

## Future Improvements

* Advanced text preprocessing
* Hyperparameter tuning
* XGBoost implementation
* LightGBM implementation
* Deep Learning models (LSTM, BERT)
* Interactive prediction interface using Streamlit

---

## Skills Demonstrated

* Data Cleaning
* Data Exploration
* NLP
* TF-IDF Vectorization
* Feature Engineering
* Machine Learning Classification
* Model Evaluation
* Random Forest
* Logistic Regression
* Python Programming
* Data Visualization

---

## Author

**Yousef Yasin**

Machine Learning and Artificial Intelligence Projects Repository.
