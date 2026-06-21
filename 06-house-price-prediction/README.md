# House Price Prediction

## Overview

This project predicts house prices using Machine Learning regression models.

The goal is to estimate the price of a house based on property features such as bedrooms, bathrooms, living area, lot size, floors, condition, renovation year, city, and zip code.

This project introduces Regression, which is different from Classification because the model predicts a continuous numerical value instead of a class label.

---

## Dataset

Dataset file:

- data.csv

Target Variable:

- price

Features include:

- bedrooms
- bathrooms
- sqft_living
- sqft_lot
- floors
- waterfront
- view
- condition
- sqft_above
- sqft_basement
- yr_built
- yr_renovated
- city
- statezip

---

## Data Preprocessing

The following preprocessing steps were applied:

- Loaded the dataset using Pandas
- Checked dataset information and data types
- Removed unnecessary text columns:
  - date
  - street
  - country
- Encoded categorical columns:
  - city
  - statezip
- Split the data into training and testing sets

---

## Models Used

### Linear Regression

Used as the baseline regression model.

### Random Forest Regressor

Used as a stronger model to compare performance.

---

## Evaluation Metrics

The models were evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R-squared Score (R²)

---

## Results

### Linear Regression

- MAE: 207,555.79
- MSE: 985,909,494,893.81
- R² Score: 0.03

### Random Forest Regressor

- MAE: 165,833.19
- MSE: 976,122,952,671.24
- R² Score: 0.04

---

## Notes

The model performance was low because house prices are strongly affected by location-based features such as city, zip code, and street. These features need more advanced preprocessing such as One-Hot Encoding, feature engineering, and outlier handling.

This project was used to practice regression models and understand the difference between classification and regression problems.

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
06-house-price-prediction/
│
├── data.csv
├── main.py
├── README.md
└── requirements.txt

Future Improvements
One-Hot Encoding for city and statezip
Outlier removal
Feature scaling
Feature engineering
XGBoost Regressor
Random Forest tuning
Price distribution visualization


Author

Yousef Yasin

Machine Learning Labs Collection