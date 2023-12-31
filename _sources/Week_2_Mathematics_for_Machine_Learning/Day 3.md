### Week 2: Mathematics for Machine Learning


#### Day 3: Calculus - Derivatives and Their Applications

**Freshman-Friendly Overview:**
Welcome to Day 3! Today, we're diving into the world of calculus, starting with derivatives. Now, if the word 'calculus' makes you a bit nervous, don't worry! Think of derivatives as a way to find out how things change. In machine learning, understanding how things change is crucial, especially when we're tweaking our models to make them better. We'll explore how derivatives help in optimizing machine learning algorithms. Let's break it down and have some fun with it!

**Activity: Exploring Derivatives in Python**

1. **Setting Up Your Environment**
   - We'll be using Python again, but this time with an additional library called `sympy`, which is great for symbolic mathematics.
     ```python
     import sympy as sp
     ```

2. **Understanding Derivatives**
   - Let's start by understanding what a derivative is. In simple terms, it tells us how a function is changing at any point.
     ```python
     x = sp.symbols('x')
     function = x**2 + 2*x + 1
     derivative = sp.diff(function, x)
     ```

3. **Applying Derivatives to a Function**
   - We'll apply the concept of derivatives to a simple function and see how it works.
     ```python
     # For example, find the derivative of x^2 + 2x + 1
     ```

4. **Visualizing the Function and Its Derivative**
   - Understanding derivatives is easier when we can see them. Let's plot our function and its derivative.
     ```python
     sp.plot(function, derivative, (x, -10, 10))
     ```

5. **Exploring Derivatives in Machine Learning**
   - Let's discuss how derivatives are used in machine learning, particularly in optimization algorithms like gradient descent.

**30-Minute Project: Implementing Gradient Descent**

- **Project Description:** You'll implement a simple version of the gradient descent algorithm, a cornerstone of optimization in machine learning. This algorithm uses derivatives to find the minimum value of a function - a critical step in training models.

- **Expected Outcome:** A basic understanding of how gradient descent uses derivatives for optimization, along with a simple Python implementation.

**Activity Code:**

```python
# Gradient Descent Implementation

import numpy as np

# Function and Derivative
def f(x):
    return x**2 + 2*x + 1

def df(x):
    return 2*x + 2

# Gradient Descent
learning_rate = 0.1
x = 0  # starting point
for i in range(25):
    grad = df(x)
    x = x - learning_rate * grad
    print(f"Step {i+1}: x = {x}, f(x) = {f(x)}")
```

**Wrap-Up:**
Nice work today! You've just scratched the surface of calculus in machine learning. Derivatives are all about understanding how changes in one quantity affect another, which is exactly what we need to do when training machine learning models. By understanding the concept of gradient descent, you're getting a glimpse into how machines learn and improve.

**Resources for Further Learning:**
- **Khan Academy's Calculus Course:** [Differential calculus](https://www.khanacademy.org/math/calculus-1) - A great resource for beginners to understand the basics of derivatives and calculus.
- **3Blue1Brown's YouTube Series:** [The Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) - Provides intuitive and visual explanations of calculus concepts.
- **Interactive Python Notebooks:** [Exploring Gradient Descent](https://github.com/jakevdp/PythonDataScienceHandbook) - Find Python Data Science Handbook by Jake VanderPlas for practical implementations.

Remember, practice makes perfect, especially with concepts like calculus. Keep experimenting with these ideas, and you'll find they become more intuitive over time. See you on Day 4 for more mathematical adventures in machine learning! 🚀📈🧮

### Day 3: Calculus - Gradient Descent and Ascent

#### Gradient Descent and Ascent Explained

**Gradient Descent:**
- **What It Is:** Gradient Descent is an optimization algorithm used in machine learning to minimize a function. Essentially, it's a method to find the lowest point (or minimum) of a function. Think of it as trying to find the lowest valley in a landscape of hills and valleys.
- **The Formula:**
  - Given a function \( f(x) \), the gradient descent formula is:
    \[ x_{\text{new}} = x_{\text{old}} - \eta \cdot \nabla f(x_{\text{old}}) \]
  - Here, \( x_{\text{old}} \) is the current position, \( \nabla f(x_{\text{old}}) \) is the gradient of the function at this position, and \( \eta \) is the learning rate, a small positive value that determines the size of the step to take.

**Gradient Ascent:**
- **What It Is:** Gradient Ascent is the opposite of Gradient Descent. It's used to maximize a function. Imagine climbing up to the peak of a hill instead of descending into a valley.
- **The Formula:**
  - For a function \( f(x) \), the gradient ascent formula is:
    \[ x_{\text{new}} = x_{\text{old}} + \eta \cdot \nabla f(x_{\text{old}}) \]
  - Similar to gradient descent, but here we add instead of subtract, moving upwards along the gradient.

#### 30-Minute Project: Implementing Gradient Descent

**Objective:**
- To implement the gradient descent algorithm step-by-step in Python and visualize how it finds the minimum of a function.

**Steps:**

1. **Choose a Function**
   - Let’s choose a simple quadratic function, \( f(x) = x^2 \), to minimize.
   - The derivative, \( f'(x) = 2x \), gives us the gradient.

2. **Set Up the Python Environment**
   - We'll need NumPy for numerical operations and Matplotlib for plotting.
     ```python
     import numpy as np
     import matplotlib.pyplot as plt
     ```

3. **Define the Function and its Derivative**
   - Write Python functions for \( f(x) \) and its derivative.
     ```python
     def f(x):
         return x**2

     def df(x):
         return 2*x
     ```

4. **Implementing Gradient Descent**
   - Start at a random point, e.g., \( x = 10 \).
   - Choose a learning rate, e.g., \( \eta = 0.1 \).
   - Update \( x \) iteratively using the gradient descent formula.
     ```python
     x = 10
     learning_rate = 0.1
     steps = 25
     for i in range(steps):
         grad = df(x)
         x = x - learning_rate * grad
     ```

5. **Visualizing the Descent**
   - Plot the function and each step of the descent.
     ```python
     x_vals = np.linspace(-10, 10, 100)
     y_vals = f(x_vals)
     plt.plot(x_vals, y_vals)
     plt.scatter(x, f(x), c='r')
     plt.show()
     ```

**Expected Outcome:**
- A visualization showing the path taken by the algorithm, descending to the minimum of the function.
- Understanding of how gradient descent moves step-by-step towards the minimum.

**Wrap-Up:**
With this project, you've experienced how gradient descent, a key algorithm in machine learning, iteratively moves towards minimizing a function. It's a powerful concept that underpins many optimization tasks in AI, from training neural networks to finding best-fit lines in data. Keep exploring these ideas, and you'll find a wealth of applications in machine learning!

**Resources for Further Learning:**
- **Online Tutorials:** Check out online platforms like Coursera or edX for courses on machine learning and optimization.
- **Interactive Notebooks:** Experiment with gradient descent using Jupyter Notebooks to deepen your understanding.
- **Reading Material:** "Deep Learning" by Goodfellow, Bengio, and Courville provides in-depth insights into these concepts.
