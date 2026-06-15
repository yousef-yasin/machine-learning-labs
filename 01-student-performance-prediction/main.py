import pandas as pd
from sklearn.model_selection import train_test_split #to split the data into training and testing sets
from sklearn.tree import DecisionTreeClassifier #to create a decision tree model for classification
from sklearn.metrics import accuracy_score #to evaluate the accuracy of the model by comparing the predicted labels with the true labels
from sklearn.metrics import confusion_matrix #to evaluate the performance of a classification model by comparing the predicted labels with the true labels and summarizing the results in a matrix format
from sklearn.metrics import classification_report #to evaluate the performance of a classification model by providing a detailed report that includes precision, recall, F1-score, and support for each class in the dataset
import matplotlib.pyplot as plt #to create visualizations and plots for data analysis and model evaluation
import seaborn as sns #to create visualizations and plots for data analysis and model evaluation



df=pd.read_csv("Student_performance_data _.csv") #to read the dataset from a CSV file and store it in a DataFrame called df


print(df.head(10))
print(df.columns)
print(df.info())

x=df.drop("GradeClass",axis=1)
y=df["GradeClass"]
print(x.head(10))
print("&&&&&&&&&&&&&&&&")
print(y.head(10))


X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("Training:", X_train.shape)
print("Testing:", X_test.shape)


model=DecisionTreeClassifier(random_state=42)
model.fit(X_train,Y_train)#fit use to train the model
y_pred=model.predict(X_test)#to check if the model is working or not
accuracy=accuracy_score(Y_test,y_pred) #to check the accuracy of the model

print("Accuracy:", accuracy)    #print the accuracy of the model
print("Predictions:", y_pred)   #print the predicted labels for the test set

print(confusion_matrix(Y_test, y_pred))

print(classification_report(Y_test, y_pred))


accuracy_percent = accuracy * 100

plt.bar(["Decision Tree"], [accuracy_percent])  #the bar chart will have one bar representing the accuracy of the Decision Tree model

plt.title("Model Accuracy") #display the title of the plot
plt.ylabel("Accuracy (%)") #display the label for the y-axis

plt.show()#i have one bar in the chart