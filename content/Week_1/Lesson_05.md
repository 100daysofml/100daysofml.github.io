# Day 5: Functions and Modules

Functions are useful in two distinct scenarios: To repeat behavior without repeating an entire procedure, or to vary behavior based on the situation at hand. In programming, a function is a block of organized, reusable code used to perform a single, related action. They provide better modularity for your application and a high degree of code reusing.

## Basic Structure of a Function

In Python, a function is defined using the `def` keyword, followed by a function name and parentheses `()`. The code block within every function starts with a colon `:` and is indented.

*Note about code blocks* - Keep your eyes peeled for **blocks** of code in Python. Whitespace is syntax: a code block shares the same level of indentation. Blocks always start with a line ending in a colon. When the indentation resumes its previous level, the block has ended.

```python
def some_function():
    print("Hello world!")
```

*Parameters and Arguments*: Parameters are variables listed inside the parentheses in the function definition. Arguments are the values passed to these parameters. Arguments are specified after the function name, inside the parentheses.

*Return Values*: Functions can return values using the `return` statement. The return value can be any object, and a function can return multiple values. Function calls are expressions that evaluate to their return value.

```python
def do_math():
    return 2 * 2
    
x = do_math() # x takes the value of do_math()'s return value, 4.
```


## Writing Your First Function

*Using `def` Keyword*: To create a function, use the `def` keyword, followed by a chosen function name and parentheses. Example: `def my_function():`. Empty parentheses indicate that the function takes no arguments.

*Function Naming Conventions*: Function names should be lowercase, with words separated by underscores as necessary to improve readability.

*Calling a Function*: To call a function, use the function name followed by parentheses. Example: `my_function()`. Arguments go inside the parentheses, and the function call's arguments must match up with the definition.

A function can contain all the same variable definitions, mathematical operations, conditional statements, and loops that you're already familiar with. The only difference is that code written directly into your Python file or Jupyter notebook cell is executed immediately ("sequentially"); code inside a function is merely being defined. It won't execute until the function is called.

## Parameters and Arguments

*Understanding Local Variables*: Variables declared inside a function are called local variables. Their scope is limited to the function, meaning they can't be accessed outside of it.

*Positional vs. Keyword Arguments*: Positional arguments are arguments that need to be included in the proper position or order. Keyword arguments are arguments accompanied by an identifier (e.g., `name='John'`) and can be listed in any order.

*Default Parameter Values*: The value provided to a keyword argument in the function's definition is the **default value**, and it's not mandatory to provide a value for that argument to call the function. The value can be overridden by whatever value is provided when the function is called, though.

*Using Keywords to Reorder Arguments*: With keyword arguments, the order of arguments can be shuffled, allowing more flexibility. Arguments passed in with no keyword have to match up with a position. Positional arguments can't follow keyword arguments.

*Local vs. Global Variables*: Global variables are defined outside a function and can be accessed throughout the program, while local variables are confined within the function they are declared in.

*Variable Shadowing*: Don't Do This. Avoid using the same name for local and global variables as it can lead to confusing "shadowing" effects, where the local variable "shadows" the global variable. The local variable will be used inside the function without modifying the global. This can be especially confusing for beginners, who may struggle with the same name being used for two separate variables in different scopes.

Using the `global` keyword, you can explicitly name the global keyword, but this is generally regarded as a bad practice or anti-pattern. It's not necessary to call methods on objects from your enclosing scope, just to assign them -- see the examples below!

```python

x = 42 # global scope variable

def g():
  x = 50 # shadowing; don't do this. You didn't change global 'x'.
  
def h():
  global x
  x += 1 # modifies global variable 'x'

stuff = []

def i():
  stuff.append("thing") # allowed. 'stuff' will be found in the global scope that encloses this function.
```

### Test Your Understanding on Args

Consider this function:

```python

def f(x, y=4):
  z = 2 * x + y
  print(f"x = {x}, y = {y}, z = {z}")
  return z
```

Questions:

* What are the arguments?
* Which ones are required to call the function?
* What would the function print out if called with `f(6)`? How about `f(7, 2)`? How about `f(1, 2, 3)`? How about `f(y=2, x=3)`?

Answers:

* The arguments are `x` and `y`.
* `x` is mandatory, `y` is optional with a default value of 4.
* Function calls:
 - `f(6)` prints `x = 6, y = 4, z = 16`.
 - `f(7, 2)` prints `x = 7, y = 2, z = 16`.
 - `f(1, 2, 3)` causes a TypeError: you can't call a function with more arguments than it has defined.
 - `f(y=2, x=3)` prints `x = 3, y = 2, z = 8`. Note that the use of keywords has assigned the function's values by name, not by position.

## Return Values and `return` Statement

  - *How to Return Data from a Function*: To send back a result from a function to its caller, use the `return` statement.

  - *Multiple Return Values; Tuple Value Unpacking*: Functions in Python can return multiple values, usually in the form of a tuple, which can be unpacked into multiple variables.

  - *None: The Default Return Value*: If no return statement is used, or the return statement does not have an accompanying value, the function returns `None`.

### Test Your Understanding on Return Values

Questions:

* asdf
* If we defined the function from the previous section and ran `n = f(2, 4)`, what would `n`'s value be?

Answers:

* qwer
* Since functions evaluate to their return value, `n` is assigned the value that `f` returns (8).


xxxxxxxxx



## Lesson Project: FizzBuzz and a Similar Challenge

#### FizzBuzz Problem Statement
FizzBuzz is a classic programming task, often used in job interviews to assess basic programming skills. The challenge is as follows:
- Print numbers from 1 to 100.
- For multiples of 3, print "Fizz" instead of the number.
- For multiples of 5, print "Buzz" instead of the number.
- For numbers which are multiples of both 3 and 5, print "FizzBuzz".

#### FizzBuzz Solution in Python

```python
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
```

This solution uses a `for` loop to iterate through the numbers 1 to 100 and conditional statements (`if`, `elif`, `else`) to determine what to print for each number.

### Alternative Lesson Project: Temperature Analyzer

#### Temperature Analyzer Problem Statement
Create a program that analyzes temperature data and categorizes each temperature entry. The challenge is as follows:
- Iterate over a list of temperature readings (in Celsius).
- For each temperature:
  - If it's below 0, categorize it as "Freezing".
  - If it's between 0 and 15 (inclusive), categorize it as "Cold".
  - If it's between 16 and 25 (inclusive), categorize it as "Moderate".
  - If it's above 25, categorize it as "Hot".
- Print the temperature and its category.

#### Temperature Analyzer Solution in Python

```python
temperatures = [3, 16, -2, 21, 30, 15, 1, -5, 24, 27]

for temp in temperatures:
    if temp < 0:
        category = "Freezing"
    elif temp <= 15:
        category = "Cold"
    elif temp <= 25:
        category = "Moderate"
    else:
        category = "Hot"
    
    print(f"Temperature: {temp}°C, Category: {category}")
```

This solution uses a `for` loop to iterate through a predefined list of temperatures and conditional statements (`if`, `elif`, `else`) to determine the category for each temperature.

#### Your Challenge: Daily Step Counter

**Problem Statement**:
Write a program that interprets a list of daily step counts. For each day's step count:
- If the steps are less than 5,000, categorize as "Sedentary".
- If the steps are between 5,000 and 7,499 (inclusive), categorize as "Lightly Active".
- If the steps are between 7,500 and 9,999 (inclusive), categorize as "Moderately Active".
- If the steps are 10,000 or more, categorize as "Very Active".
- Print the day number and the corresponding activity category.

#### Further Resources


---

**Lesson 5: Functions and Modules**

- **Advanced Function Concepts**
  - *Variable Number of Arguments (`*args` and `**kwargs`)*: Use `*args` for a variable number of positional arguments and `**kwargs` for a variable number of keyword arguments. This allows functions to handle an unspecified number of arguments.
  - *Call Stack: Calling Functions from Within Functions*: Understanding how functions can call other functions and how the call stack works, which is crucial for understanding function execution flow.
  - *Reading a Traceback*: Learning to read and interpret traceback errors to debug issues in function calls.
  - *Recursion*: Understanding recursion, a method where the solution to a problem depends on solutions to smaller instances of the same problem. This involves a function calling itself.

- **Introduction to Python Modules**
  - *Definition and Importance of Modules*: Modules in Python are simply Python files with a `.py` extension containing Python definitions and statements. Modules help in dividing the code into smaller parts and organizing it logically.
  - *Standard Library Modules*: Overview of Python’s standard library modules, which provide a range of functionality.
  - *Importing Modules using `import`*: How to import modules into your Python script using the `import` statement.
  - *Accessing Functions from Modules*: Demonstrating how to access and use functions defined in imported modules.

- **Math Focus: Functions for Mathematical Formulas**
  - *Writing Functions for Basic Arithmetic Operations*: Creating functions for addition, subtraction, multiplication, and division.
  - *Functions for Common Mathematical Formulas (Area, Perimeter, Volume)*: Developing functions to calculate the area, perimeter, and volume of various shapes.
  - *Utilizing Math Module for Advanced Mathematical Functions*: Exploring Python's math module to perform more complex mathematical operations like trigonometric functions, exponential, square root, etc.

- **Practical Exercises and Projects**
  - *Creating Custom Functions for Real-World Problems*: Practical exercises where students create functions to solve specific problems.
  - *Group Activity: Developing a Small Library of Mathematical Functions*: Working in groups to create a library of reusable mathematical functions.
  - *Homework: Implement a Function to Solve a Specific Mathematical Problem*: Assigning homework that involves writing a function to solve a mathematical or real-world problem, reinforcing the concepts learned.
  


---

Sure, let's dive into a simple AI/ML example using NumPy, specifically demonstrating the matrix multiplication required in a linear regression calculation. We'll compute the loss, which is a measure of how far our model's predictions are from the actual outcomes.

Here's the step-by-step process:

1. **Import NumPy Library**
   - First, we import NumPy: `import numpy as np`

2. **Define Coefficients and Data**
   - Suppose we have a linear model with coefficients `a` and `b` (for a simple linear equation `y = ax + b`). We'll represent these as a NumPy array:
     ```python
     coefficients = np.array([a, b])
     ```
   - Let's also define our data points. In a simple linear regression, we often have input (`x`) and output (`y`) pairs. For this example, `x` and `y` are both arrays. Our `x` data needs an additional column of ones to account for the intercept `b`:
     ```python
     x_data = np.array([[x1, 1], [x2, 1], ..., [xn, 1]])
     y_data = np.array([y1, y2, ..., yn])
     ```

3. **Perform Matrix Multiplication for Prediction**
   - We compute the predicted `y` values (`y_pred`) using matrix multiplication between our data (`x_data`) and coefficients. In NumPy, matrix multiplication is done using the `@` operator or `np.dot()` function:
     ```python
     y_pred = x_data @ coefficients
     ```

4. **Compute the Loss**
   - The loss function quantifies how far our predictions are from the actual values. A common loss function in linear regression is Mean Squared Error (MSE), calculated as the average of the squares of the differences between actual (`y_data`) and predicted (`y_pred`) values:
     ```python
     loss = np.mean((y_data - y_pred) ** 2)
     ```

Here's the entire process in a cohesive code snippet:

```python
import numpy as np

# Example coefficients for the linear model (y = ax + b)
a, b = 2, 1  # Replace with actual values
coefficients = np.array([a, b])

# Example data (Replace with actual data points)
x_data = np.array([[1, 1], [2, 1], [3, 1]])  # Add a column of ones for the intercept
y_data = np.array([3, 5, 7])  # Actual y values

# Predict y using matrix multiplication
y_pred = x_data @ coefficients

# Calculate the loss (Mean Squared Error)
loss = np.mean((y_data - y_pred) ** 2)

print("Predicted y:", y_pred)
print("Loss (MSE):", loss)
```

In this example, `x_data` and `coefficients` are multiplied, resulting in predictions `y_pred`. The loss is then calculated as the MSE, which gives us a quantitative measure of the model's accuracy. This process is fundamental in linear regression and many other machine learning algorithms.
