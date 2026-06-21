import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib#to save the trained model to a file for later use, allowing you to load the model without retraining it each time you want to make predictions. This is useful for deploying the model in production or sharing it with others.


df = pd.read_csv('creditcard.csv')
print(df.head())
print(df.info())


print(df['Class'].value_counts())

print("\nPercentage:")
print(df['Class'].value_counts(normalize=True)*100)  # Percentage of each class
#to check the distribution of the target variable 'Class' in the dataset. The value_counts() function counts the occurrences of each unique value in the 'Class' column, which indicates whether a transaction is fraudulent (1) or not (0). The normalize=True parameter calculates the relative frequencies, and multiplying by 100 converts these frequencies into percentages. This helps to understand the class imbalance in the dataset, which is crucial for building effective machine learning models for fraud detection.

x=df.drop(['Class'], axis=1)
y=df['Class']
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,stratify=y)  # Stratified sampling to maintain class distribution in train and test sets
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


model = RandomForestClassifier(
    n_estimators=30,
    random_state=42,
    class_weight="balanced"
)
model.fit(x_train, y_train)

predictions = model.predict(x_test)
probabilities = model.predict_proba(x_test)[:, 1]

print("Training Finished")
print(predictions[:10])
print(probabilities[:10])

metrics = classification_report(y_test, predictions)
print(metrics)

roc_auc = roc_auc_score(y_test, probabilities)
print("ROC AUC Score:", roc_auc)

specificity = confusion_matrix(y_test, predictions)[0, 0] / (confusion_matrix(y_test, predictions)[0, 0] + confusion_matrix(y_test, predictions)[0, 1])
print("Specificity:", specificity)


joblib.dump(model, "fraud_model.pkl")
print("Model saved as fraud_model.pkl")