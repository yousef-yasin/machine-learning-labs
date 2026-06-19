import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score #to evaluate the accuracy of the model by comparing the predicted labels with the true labels
from sklearn.metrics import confusion_matrix ,classification_report #to evaluate the performance of a classification model by comparing the predicted
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('heart.csv')#to read the dataset from a CSV file and store it in a DataFrame called df

print(df.head(10))
print(df.isnull().sum())
print("************")
x=df.drop("target", axis=1)#to create a new DataFrame x by dropping the "target" column from the original DataFrame df, which will be used as the feature set for training the model
y=df["target"]
print(x.head(10))
print("&&&&&&&&&&&&&&&&")
print(y.head(10))

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)#to split the dataset into training and testing sets, where 20% of the data is used for testing and the random state is set to 42 for reproducibility
print("Training:", X_train.shape)
print("Testing:", X_test.shape)

lr = LogisticRegression(max_iter=1000)#to create a logistic regression model for classification and set the maximum number of iterations to 1000 to ensure convergence during training
lr.fit(X_train, Y_train)
y_pred = lr.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)
print("Predictions:", y_pred)

print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))

rf = RandomForestClassifier(random_state=42)#to create a random forest classifier model for classification and set the random state to 42 for reproducibility
rf.fit(X_train, Y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(Y_test, y_pred_rf)

print("Random Forest Accuracy:", accuracy_rf)
print("Random Forest Predictions:", y_pred_rf)
print(confusion_matrix(Y_test, y_pred_rf))
print(classification_report(Y_test, y_pred_rf))