# Lesson 21: Introduction to Regression Analysis in Python

## Introduction to Regression Analysis

Regression analysis is a cornerstone of data science, particularly in the field of supervised learning. It allows us to predict the value of a continuous variable based on one or more other variables. For example, we might use regression to predict the price of a house based on its size, location, and age. Importantly, regression is not suitable for predicting categorical variables; such tasks are better handled by classification methods like logistic regression, which we will explore in a later lesson.

Understanding the difference between supervised and unsupervised learning is crucial. In supervised learning, our model learns from a labeled dataset, which means each example in the dataset is paired with the correct output. Regression is a form of supervised learning because it requires a dataset with known outputs to learn the relationship between variables. In contrast, unsupervised learning involves finding patterns in data where the outcomes are not known in advance.

## Mathematical Foundation of Regression

### Linear Equations

The simplest form of regression is simple linear regression, where we predict the outcome as a linear function of the input. The equation for a line in two dimensions is \(y = mx + c\), where:
- \(y\) is the value we want to predict (dependent variable).
- \(m\) is the slope of the line.
- \(x\) is our input variable (independent variable).
- \(c\) is the y-intercept, which tells us the value of \(y\) when \(x\) is 0.

### Hyperplanes in Higher Dimensions

When dealing with more than one input variable, the concept of a line extends to a hyperplane. In three dimensions, this hyperplane is a flat surface, and in higher dimensions, it's an n-dimensional subspace. These hyperplanes are the basis of multiple linear regression, where we predict an outcome based on several input variables.

### Best-Fit Line and Least Squares Method

In linear regression, we aim to find the line (or hyperplane in multiple linear regression) that best fits our data. The "best fit" is usually defined using the least squares method, which minimizes the sum of the squares of the differences (residuals) between the observed values and the values predicted by the line.

## Python Implementation

### Using numpy for Basic Statistical Calculations

Python's numpy library is a powerful tool for statistical calculations. Let's start with some basic statistics on a hypothetical dataset:

```python
import numpy as np

# Example data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Basic statistics
mean_x = np.mean(x)
mean_y = np.mean(y)
print("Mean of x:", mean_x)
print("Mean of y:", mean_y)
```

### Implementing Simple Linear Regression with scikit-learn

Scikit-learn is a go-to library for implementing machine learning algorithms. Here's how we can use it for simple linear regression:

```python
from sklearn.linear_model import LinearRegression

# Reshaping data for scikit-learn
x_reshaped = x.reshape(-1, 1)

# Create and train the model
model = LinearRegression()
model.fit(x_reshaped, y)

# Coefficients
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
```

### Visualizing Data and Regression Line Using matplotlib

We can use matplotlib to visualize our data points and the fitted regression line:

```python
import matplotlib.pyplot as plt

# Predict values
predicted_y = model.predict(x_reshaped)

# Plotting
plt.scatter(x, y, color='blue')
plt.plot(x, predicted_y, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simple Linear Regression')
plt.show()
```

## Example Dataset: Housing Price Prediction

For our example, we will use a housing dataset to predict house prices. We will import the dataset using pandas, visualize the relationship between variables using seaborn, and then build a simple linear regression model.

```python
import pandas as pd
import seaborn as sns

# Load dataset (replace with the path to your dataset)
housing_data = pd.read_csv('housing.csv')

# Pairplot to visualize relationships
sns.pairplot(housing_data)
plt.show()

# Simple Linear Regression on a chosen feature (e.g., 'size')
X = housing_data[['size']]  # Independent variable
y = housing_data['price']   # Dependent variable

# Building the model
model = LinearRegression()
model.fit(X, y)

# Plotting the regression
predicted_price = model.predict(X)
plt.scatter(X, y, color='blue')
plt.plot(X, predicted_price, color='red')
plt.xlabel('Size')
plt.ylabel('Price')
plt.title('Housing Price Prediction')
plt.show()
```

In this lesson, we've explored the basics of regression analysis, its mathematical foundation, and practical implementation in Python. We've seen how regression can be used to predict continuous variables and how Python's libraries make it straightforward to implement and visualize regression models. This foundational knowledge sets the stage for more complex regression techniques and applications in data science.