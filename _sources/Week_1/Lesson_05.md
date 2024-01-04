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

* If we defined the function from the previous section and ran `n = f(2, 4)`, what would `n`'s value be?

Answers:

* Since functions evaluate to their return value, `n` is assigned the value that `f` returns (8).

## Get New Functionality Fast Via Modules

**Modules** are a way to bring complex capabilities into your Python programs with the use of the `import` keyword. The Python standard library is sometimes described as "batteries included", which includes functionality like: interacting with JSON data or CSV files, making web requests, mathematical functions, random numbers, interacting with dates and times, file compression, logging, cryptography, and the list goes on!

Modules are also very important to the performance of Python programs. You may have heard that Python is a "slow language". This is more or less true in many circumstances. However, when you import a popular data science or machine learning library like Pandas, Scikit, Keras, or PyTorch, the module itself is written in another language and has very few performance constraints. Modules written in C, C++, or Rust can run at the speed of highly efficient compiled languages, then we use Python as "glue code" to load these modules, provide them with data, and utilize their output.

### `import` Syntax

When you `import` something, that module will become available in your script's namespace.

```python
import math
```

After importing, the module is an object that you can interact with:

```
>>> math
<module 'math' (built-in)>
>>> type(math)
<class 'module'>
```

In fact, if you save any Python code in a `.py` file in the same directory as your notebook or additional files, you can import them as modules and access their contents. Just like using a function on any other object, you can use the dot operator (`.`) to access the contents of a module. Modules can contain any Python object.

```
>>> math.pi
3.141592653589793
>>> math.pow
<built-in function pow>
```

The name of the module is not forced upon you, though. You can change the name of something at the time of import, or simply import all of the contents into your current namespace. The `import ... as ...` syntax allows you to import a module with a specific new name. Here's one you'll see frequently:

```python
import matplotlib.pyplot as plt
```

The module `matplotlib` contains a module `pyplot`, and typing `matplotlib.pyplot` dozens of times in your code would be exhausting. When importing `as`, the given name `plt` becomes an alias for the module.

You can import specific parts of a module, leaving the rest behind:

```python
from math import pi
```

This would solely create a variable `pi`, no dots required, sourced from the `math` module.

Use this sparingly, but it's possible to import all of the contents of a module without using a dot operator.

```python
from math import *
```

This would bring all of the contents into your current file; variables or functions like `pi` or `pow` would now be available directly. It's often not recommended because it's difficult to trace back a variable to the module it comes from. If you are using your own `.py` files to encapsulate imports and definitions, it's occasionally OK to do this. Readability is greatly impacted, so use star imports with great caution.

### Standard Library Highlights

There's no way I could summarize the standard library adequately. As homework, peruse the [Python Standard Library documentation](https://docs.python.org/3/library/index.html).

* the `random` module: very handy for games or testing your functions with a lot of diverse input.
  ```python
  import random
  dice_roll = random.randint(1, 6)
  random.random() # float between 0.0 and 1.0
  ```

* the `json` module: important for sending and retrieving data over web services or saving to file.
  ```python
  import json
  data = {'a': 1, 'b': 2}
  txt = json.dumps(data) # dump to string
  loaded = json.loads(txt) # load from string
  ```

* the `time` module: for measuring elapsed time or deliberately slowing down your program.
  ```python
  import time
  start = time.time()
  for i in range(5):
    print(".", end='')
    time.sleep(1.5)
  print()
  end = time.time()
  print(f"Elapsed: {end - start:.2f} seconds")
  ```

* the `datetime` module: for interacting with calendars, timestamps, timezones, and durations.
  ```python
  from datetime import datetime
  print(datetime.now())
  ```


### Every AI/ML Developer Must Know NumPy

Let's introduce NumPy by demonstrating the matrix multiplication required in a linear regression calculation. We'll compute the loss, which is a measure of how far our model's predictions are from the actual outcomes.

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
