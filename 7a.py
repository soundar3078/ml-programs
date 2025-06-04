import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

housing_data_df = pd.read_csv('HousingData.csv')
housing_data_df.fillna(housing_data_df.mean(), inplace=True)

def linear_regression_housing():
    X = housing_data_df[['RM', 'LSTAT', 'PTRATIO']]
    y = housing_data_df['MEDV']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Linear Regression on Boston Housing Dataset")
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Prices (in $1000s)")
    plt.ylabel("Predicted Prices (in $1000s)")
    plt.title("Actual vs Predicted Prices")
    plt.show()

linear_regression_housing()
