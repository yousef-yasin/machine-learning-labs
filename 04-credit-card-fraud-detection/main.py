import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns


# Read dataset
df = pd.read_csv("creditcard.csv")

# Explore dataset
print(df.head())
print(df.info())
print(df.isnull().sum())

print("Class Distribution:")
print(df["Class"].value_counts())

# Features and target
x = df.drop("Class", axis=1)
y = df["Class"]

print(x.head(10))
print("&&&&&&&&&&&&&&&&")
print(y.head(10))

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print("Training:", x_train.shape)
print("Testing:", x_test.shape)

# Random Forest Model
rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(x_train, y_train)

y_pred = rf_model.predict(x_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("First 20 Predictions:", y_pred[:20])

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Accuracy chart
plt.bar(["Random Forest"], [accuracy * 100])
plt.title("Model Accuracy")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.show()

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()