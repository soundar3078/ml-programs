import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

auto_mpg_df = pd.read_csv('auto_mpg.csv')

def polynomial_regression_auto_mpg():
    X = auto_mpg_df[['displacement', 'horsepower', 'weight', 'acceleration']]
    y = auto_mpg_df['mpg']

    X['horsepower'] = pd.to_numeric(X['horsepower'], errors='coerce')
    X.fillna(X.mean(), inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Polynomial Regression on Auto MPG Dataset")
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual MPG")
    plt.ylabel("Predicted MPG")
    plt.title("Actual vs Predicted MPG")
    plt.show()

polynomial_regression_auto_mpg()
