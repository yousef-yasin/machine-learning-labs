import pandas as pd
from sklearn.preprocessing import LabelEncoder  #to convert categorical variables into numerical format for machine learning models
from sklearn.model_selection import train_test_split #to split the data into training and testing sets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #to evaluate the performance of the model by comparing the predicted labels with the true labels and summarizing the results in a matrix format
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("Telco-Customer-Churn.csv")

print(df.head())
print(df.columns)
print(df.info()) #display the summary of the DataFrame, including the number of non-null values and the data types of each column
print(df.isnull().sum()) #to check for missing values in the dataset and count the number of null values in each column



#now we start to clean the data and make it ready for analysis and modeling
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")#make the "TotalCharges" column numeric


print(df["TotalCharges"].isnull().sum())#to check for missing values in the "TotalCharges"

df = df.dropna()
print(df.shape)

print(df["TotalCharges"].isnull().sum())#to print the null in TOtalCharges column

df=df.drop("customerID", axis=1) #to drop the "customerID" column from the DataFrame, as it is not relevant for the analysis
print(df.columns)
print(df.head())#here we delted the customerID column 

le=LabelEncoder() #to create an instance of the LabelEncoder class, which will be used to convert categorical variables into numerical format
for col in df.columns: #to make all coloum numeric
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

print(df.head())

x=df.drop("Churn", axis=1) #to create a new DataFrame x by dropping the "Churn" column from the original DataFrame df, which will be used as the feature set for machine learning models
y=df["Churn"] #to create a new Series y by selecting the "Churn" column from the original DataFrame df, which will be used as the target variable for machine learning models
print(x.head())
print("&&&&&&&&&&&&&&&&")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) #to split the feature set x and target variable y into training and testing sets, with 20% of the data used for testing and a random state of 42 for reproducibility
print("Training:", x_train.shape) 
print("Testing:", x_test.shape)

model = DecisionTreeClassifier(random_state=42) #to create an instance of the DecisionTreeClassifier class, which will be used to train a decision tree model for classification
model.fit(x_train, y_train) #to train the decision tree model using the training data  
y_pred = model.predict(x_test) #to make predictions on the test set using the trained decision tree model
accuracy = accuracy_score(y_test, y_pred) #to evaluate the accuracy of the model by
print("Accuracy:", accuracy) #to print the accuracy of the model
print("Predictions:", y_pred) #to print the predicted labels for the test set
print(confusion_matrix(y_test, y_pred)) #to print the confusion matrix, which summarizes the performance of the classification model by comparing the predicted labels with the true labels in a matrix format
print(classification_report(y_test, y_pred)) #to print the classification report, which provides

rf = RandomForestClassifier( #to create an instance of the RandomForestClassifier class, which will be used to train a random forest model for classification
    n_estimators=100,
    random_state=42
)

rf.fit(x_train,y_train)

rf_pred = rf.predict(x_test)

rf_accuracy = accuracy_score(y_test,rf_pred) #to evaluate the accuracy of the random forest model by comparing the predicted labels with the true labels in the test set and calculating the proportion of correct predictions

print("Random Forest Accuracy:",rf_accuracy)

print(confusion_matrix(y_test,rf_pred))

print(classification_report(y_test,rf_pred))