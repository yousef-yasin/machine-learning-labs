import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data.csv')

print(df.head())
print(df.info())


le_city = LabelEncoder()
le_statezip = LabelEncoder()

df["city"] = le_city.fit_transform(df["city"])
df["statezip"] = le_statezip.fit_transform(df["statezip"])

x = df.drop([
    "price",
    "date",
    "street",
    "country"
], axis=1)

y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)#the average of the absolute differences between the predicted values and the actual values. It is a measure of how well the model fits the data, with lower values indicating a better fit. The MAE is calculated by taking the absolute difference between each predicted value and its corresponding actual value, summing these differences, and then dividing by the total number of predictions.
mse = mean_squared_error(y_test, y_pred)#the average of the squares of the errors, which is a measure of how well the model fits the data. It is calculated by taking the average of the squared differences between the predicted values and the actual values. A lower MSE indicates a better fit of the model to the data.
r2 = r2_score(y_test, y_pred) #the coefficient of determination, which is a measure of how well the model fits the data. It ranges from 0 to 1, where 1 indicates a perfect fit and 0 indicates no fit at all.

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)