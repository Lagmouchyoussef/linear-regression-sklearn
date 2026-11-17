# 📊 Simple Linear Regression with scikit-learn

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas" alt="pandas" />
  <img src="https://img.shields.io/badge/Matplotlib-3.7%2B-11557C?style=for-the-badge&logo=matplotlib" alt="matplotlib" />
</div>

A complete machine learning project that demonstrates the use of simple linear regression to model the relationship between years of experience and annual salary. This project combines theoretical understanding with practical implementation using Python libraries such as `pandas`, `matplotlib`, and `scikit-learn`.

## 🎯 Objective

The aim of this project is to:
- understand the concept of simple linear regression
- estimate the model parameters θ₀ and θ₁
- evaluate model performance using MSE and R²
- visualize the regression line
- compare the analytical solution with gradient descent
- interpret the results in a realistic business context

## 🧠 Mathematical Model

The relationship is modeled as:

h(x) = θ₀ + θ₁x

where:
- x = years of experience
- y = salary in kDH
- θ₀ = intercept
- θ₁ = slope

The cost function is:

J(θ₀, θ₁) = (1 / 2m) Σ (hθ(xᵢ) - yᵢ)²

This project applies both:
- the analytical method using the normal equation
- the numerical method using gradient descent (`SGDRegressor`)

## 📊 Dataset

The data used in this exercise is:

| Experience (years) | Salary (kDH) |
| --- | ---: |
| 1 | 22 |
| 2 | 27 |
| 3 | 34 |
| 4 | 40 |
| 5 | 45 |
| 6 | 52 |
| 7 | 57 |

## 🖼️ Regression Plot

![Simple Linear Regression Plot](./regression_lineaire_simple.png)

## 📁 Project Structure

```text
linear-regression-sklearn/
├── main.py                  # Main Python script for the full exercise
├── README.md                # Project documentation
├── regression_lineaire_simple.png  # Generated plot
├── .venv/                   # Virtual environment (if created)
└── requirements.txt         # Optional dependency file (can be added if needed)
```

## ⚙️ Requirements

Before running the project, install the required Python libraries:

```bash
pip install pandas matplotlib scikit-learn
```

## 🚀 Step-by-Step Guide

### 1) Open the project folder

Open the folder in VS Code or in your preferred Python IDE.

### 2) Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install pandas matplotlib scikit-learn
```

### 4) Run the script

From the project root:

```bash
python main.py
```

Or on Windows PowerShell:

```powershell
cd "D:\Travaille EMSI\4IIR G9 - G92 ( 1er Semestre )\Machine Learning\TP\TP 2\linear-regression-sklearn"
.\.venv\Scripts\python.exe main.py
```

### 5) Observe the results

The script prints:
- the dataset
- the optimal model parameters
- the regression equation
- predictions for 3.5 and 10 years of experience
- the MSE and R² values
- the comparison between analytical and gradient descent methods

## ✅ Expected Results

The trained model gives the following approximate results:

- θ₀ ≈ 15.857
- θ₁ ≈ 5.929
- Equation: h(x) = 15.857 + 5.929x
- MSE ≈ 0.224
- J(θ) = MSE / 2 ≈ 0.112
- R² ≈ 0.998
- Prediction at 3.5 years ≈ 36.6 kDH
- Prediction at 10 years ≈ 75.1 kDH

## 🔍 Interpretation of the Results

### Intercept θ₀
The intercept indicates the expected salary when experience is zero. In this case:
- θ₀ ≈ 15.857
- This means the model estimates a base salary of about 15.9 kDH at 0 years of experience.

### Slope θ₁
The slope measures how much the salary increases per additional year of experience.

- θ₁ ≈ 5.929
- This means a salary increase of approximately 5.9 kDH for each additional year of experience.

### Model quality
The coefficient of determination R² is very close to 1:
- R² ≈ 0.998
- This shows that the model fits the data very well.

### Prediction comments
- The prediction for 3.5 years is reliable because it lies within the observed experience range.
- The prediction for 10 years is an extrapolation, so it is less reliable than an in-range prediction.

## 🧪 Exercise Breakdown

### Exercise 0 — Data preparation
Create a DataFrame with the features and target variable:
- Experience
- Salary

### Exercise 1 — Find the optimal parameters analytically
Use `LinearRegression()` from scikit-learn:
- fit the model on X and y
- recover `intercept_` and `coef_`
- write the final equation

### Exercise 2 — Evaluate the model
Use:
- `mean_squared_error(y, y_pred)`
- `r2_score(y, y_pred)`

This gives a clear idea of the quality of the model.

### Exercise 3 — Visualization
Plot:
- the real data points
- the regression line
- titles, axis labels, and legend

### Exercise 4 — Gradient descent comparison
Train `SGDRegressor` with different learning rates and compare the coefficients with the analytical solution.

This helps understand how:
- the learning rate affects convergence
- gradient descent approximates the optimum iteratively
- large-scale optimization often relies on iterative methods

## 📈 Gradient Descent Analysis

The implementation compares multiple values:

- eta0 = 0.0001 → too small, slow convergence
- eta0 = 0.01 → close to the analytical solution
- eta0 = 0.1 → unstable or less accurate

This demonstrates that learning rate selection strongly affects the optimization result.

## 🏁 Why This Matters

Simple linear regression is one of the most important algorithms in machine learning because it provides:
- a clear mathematical foundation
- a basis for more advanced regression models
- interpretable results
- a practical approach for prediction and decision support

It is widely used in:
- business forecasting
- economic analysis
- scientific modeling
- performance prediction

## 🛠️ Troubleshooting

If the plot does not appear:
- ensure `matplotlib` is installed
- run the script in a GUI-capable environment if you want an interactive plot
- open the generated PNG file directly in the project folder

If dependencies are missing:

```bash
pip install pandas matplotlib scikit-learn
```

## 📌 Conclusion

This project successfully demonstrates how simple linear regression can be used to model the relationship between experience and salary. The analytical method and the gradient descent method both converge toward a strong predictive model, and the final visualization clearly confirms the linear trend in the data.

## 🏆 Final Note

This project is designed for educational and learning purposes and is a practical introduction to supervised learning, model fitting, optimization, and evaluation in Python.
