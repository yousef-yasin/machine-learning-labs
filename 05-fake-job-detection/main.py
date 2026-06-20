import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('fake_job_postings.csv')
print(df.head())
print(df.info())

df= df.fillna('')  # Fill missing values with empty strings

df['text'] = df['title'] + ' ' + df['location'] + ' ' + df['department'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + df['benefits']

X = df["text"]
y = df["fraudulent"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(["X_train shape:", x_train.shape])
print(["X_test shape:", x_test.shape])


tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = tfidf.fit_transform(x_train)
X_test_tfidf = tfidf.transform(x_test)

print(X_train_tfidf.shape)
print(X_test_tfidf.shape)


lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)
y_pred = lr.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_tfidf, y_train)
rf_pred = rf.predict(X_test_tfidf)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_accuracy)
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

plt.bar(
    ["Logistic Regression", "Random Forest"],
    [accuracy * 100, rf_accuracy * 100]
)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.show()


rf_cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(6,4))

sns.heatmap(
    rf_cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

new_job = """
Work from home.
Earn $5000 per week.
No experience required.
Immediate hiring.
"""
new_job_tfidf = tfidf.transform([new_job])
prediction = rf.predict(new_job_tfidf)
if prediction[0] == 1:
    print("Fake Job")
else:
    print("Real Job")