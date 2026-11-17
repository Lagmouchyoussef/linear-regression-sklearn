import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score


def main():
    # Exercise 0: data preparation
    df = pd.DataFrame({
        "Experience": [1, 2, 3, 4, 5, 6, 7],
        "Salary": [22, 27, 34, 40, 45, 52, 57],
    })
    print("Data:")
    print(df)
    print()

    X = df[["Experience"]]
    y = df["Salary"]

    # Exercise 1: optimal parameters (analytical method)
    model = LinearRegression()
    model.fit(X, y)

    theta0 = model.intercept_
    theta1 = model.coef_[0]
    print("Optimal parameters:")
    print(f"theta0 (intercept) = {theta0:.3f}")
    print(f"theta1 (slope) = {theta1:.3f}")
    print(f"Equation: h(x) = {theta0:.3f} + {theta1:.3f} * x")

    x_3_5 = pd.DataFrame({"Experience": [3.5]})
    pred_3_5 = model.predict(x_3_5)[0]
    print(f"Prediction for 3.5 years of experience: {pred_3_5:.1f} kDH")
    print()

    # Exercise 2: model evaluation
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    j_cost = mse / 2
    r2 = r2_score(y, y_pred)

    print("Model evaluation:")
    print(f"MSE = {mse:.3f}")
    print(f"J(theta) = MSE / 2 = {j_cost:.3f}")
    print(f"R^2 = {r2:.3f}")
    print()

    # Exercise 3: visualization
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color="blue", label="Actual data")
    x_line = np.linspace(X["Experience"].min(), X["Experience"].max(), 100)
    y_line = theta0 + theta1 * x_line
    plt.plot(x_line, y_line, color="red", label="Regression line")
    plt.title("Simple Linear Regression")
    plt.xlabel("Experience (years)")
    plt.ylabel("Salary (kDH)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    print("Graph displayed with plt.show()")

    x_10 = pd.DataFrame({"Experience": [10]})
    pred_10 = model.predict(x_10)[0]
    print(f"Prediction for 10 years of experience: {pred_10:.1f} kDH")
    print("This forecast is an extrapolation and is less reliable than a prediction within the range of the training data.")
    print()

    # Exercise 4: gradient descent
    print("Gradient descent (SGDRegressor):")
    learning_rates = [1e-4, 1e-2, 1e-1]
    for eta0 in learning_rates:
        sgd_model = SGDRegressor(
            loss="squared_error",
            penalty=None,
            learning_rate="constant",
            eta0=eta0,
            max_iter=10000,
            tol=None,
            random_state=42,
        )
        sgd_model.fit(X, y)
        print(f"eta0 = {eta0} -> theta0 ≈ {sgd_model.intercept_[0]:.3f}, theta1 ≈ {sgd_model.coef_[0]:.3f}")

    sgd_01 = SGDRegressor(
        loss="squared_error",
        penalty=None,
        learning_rate="constant",
        eta0=0.01,
        max_iter=10000,
        tol=None,
        random_state=42,
    )
    sgd_01.fit(X, y)
    print()
    print("Comparison with the analytical method (eta0 = 0.01):")
    print(f"theta0 (SGD) = {sgd_01.intercept_[0]:.3f} ; theta0 (analytical) = {theta0:.3f}")
    print(f"theta1 (SGD) = {sgd_01.coef_[0]:.3f} ; theta1 (analytical) = {theta1:.3f}")
    print("The difference comes from the fact that gradient descent uses an iterative approximation, while the analytical method gives the exact solution for a simple linear problem.")
    print("Gradient descent is particularly useful when the number of parameters is very large, because analytical resolution becomes expensive or impossible.")


if __name__ == "__main__":
    main()