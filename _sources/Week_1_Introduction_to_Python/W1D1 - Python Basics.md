# Day 1: Python Basics - Syntax, Variables

## Setting Up Your Environment

**What will we use Python for in 100 Days of Machine Learning?**
What will we use Python for in 100 Days of Machine Learning?

In our 100 Days of Machine Learning challenge, Python will be our go-to programming language. It's widely used in the machine learning community due to its readability, simplicity, and rich ecosystem of data science libraries like NumPy, Pandas, and scikit-learn. These tools will be invaluable for data preprocessing, analysis, and building machine learning models.

**Google Colab**

For your coding environment, as long as you have a Google account, the best choice is likely Colab, accessible at [Google Colab](https://colab.research.google.com/). Colab is a free, cloud-based version of Jupyter Notebook. It allows you to write and execute Python code through your browser, without any setup required. Colab also provides free access to computing resources, including GPUs, which can be beneficial for some of the more computationally heavy machine learning tasks.

Working locally is an option, but if you don't have a CPU, you probably want to get acquainted with Colab anyway for the later challenges.

**Notebooks and Cells**

In Colab, you work with notebooks, which consist of a series of "cells". These cells can contain either executable code or text (markdown). A key point to remember is that cells should be run in order – since later cells might depend on code or variables defined in earlier cells. However, if you're experimenting or debugging, you can rerun individual cells in any order.

**Tips for Using Notebooks:**

* **Running Cells in Sequence**: Always run cells in the order they appear, especially when you're following a tutorial or working on a new concept.
* **Re-running Cells**: If you encounter errors or unexpected results, a useful troubleshooting step is to use the "Run all" feature to execute all cells from the beginning. This ensures that all your code and variables are up-to-date.
* **Dealing with Errors or Invalid Plots**: Errors or incorrect plots often occur due to changes in the code that haven't been propagated through the notebook. By using "Run all cells" (found in the Runtime menu), you can ensure that all cells reflect the latest code changes.
* **Saving Work**: Regularly save your notebooks. While Colab autosaves to Google Drive, it's a good habit to manually save important changes.

With Google Colab, you get a powerful, flexible, and easily accessible platform for coding in Python, perfect for our machine learning journey. 

## Syntax

**Expressions and Statements**

In Python, an expression is a combination of values and operators that can be evaluated to produce another value. For example, `3 + 4` is an expression where `3` and `4` are values and `+` is the operator. A statement, on the other hand, is an instruction that Python can execute. For example, `print(3 + 4)` is a statement that executes the `print()` function.

**Structure of Python Code**

After understanding expressions and statements, the next crucial aspect is the structure of Python code, particularly focusing on code blocks and the use of whitespace.

1. **Code Blocks and Indentation:**
   - In Python, code blocks are defined by indentation, which is unique compared to other programming languages that often use braces (`{}`).
   - A code block starts with an indentation and ends with the first unindented line. The amount of indentation is up to you, but it must be consistent throughout that block.
   - Commonly, code blocks are used in loops, functions, and conditions. For example:
     ```python
     if x > 0:
         print("x is positive")
     ```
     Here, `print("x is positive")` is a block of code that will execute if the condition `x > 0` is true.

2. **Whitespace as Syntax:**
   - Whitespace is meaningful in Python. Apart from indentation for code blocks, spaces and tabs are used to separate elements within a line. However, excessive whitespace within lines is ignored.
   - Good use of whitespace makes your code more readable and hence, maintainable. For example:
     ```python
     # Good use of whitespace
     sum = a + b
     # Harder to read due to lack of whitespace
     sum=a+b
     ```

3. **Line Continuation:**
   - Python allows for line continuation within parentheses `()`, brackets `[]`, and braces `{}`. This is useful for writing lengthy expressions:
     ```python
     total = (a + b + c +
              d + e + f)
     ```
   - For lines that are not naturally continued, you can use a backslash `\` to indicate that the line should continue:
     ```python
     total = a + b + c + \
             d + e + f
     ```

4. **Comments:**
   - Comments are essential for explaining code. They start with a `#` and are ignored by the Python interpreter.
   - Use comments to describe complex logic or to make notes for future reference.
   
**Operators**

Operators in Python are special symbols that carry out arithmetic or logical computation. The primary arithmetic operators include:

1. **Addition (`+`)**: Adds two operands. For example, `5 + 3` equals `8`.
2. **Subtraction (`-`)**: Subtracts the right operand from the left operand. For instance, `5 - 3` equals `2`.
3. **Multiplication (`*`)**: Multiplies two operands. For example, `5 * 3` equals `15`.
4. **True Division (`/`)**: Divides the left operand by the right operand and returns a floating-point number. For instance, `5 / 2` equals `2.5`.
5. **Floor Division (`//`)**: Divides and returns the largest whole number that is smaller than or equal to the result. For example, `5 // 2` equals `2`.
6. **Modulus (`%`)**: Returns the remainder when the left operand is divided by the right operand. For example, `5 % 2` equals `1`.
7. **Exponentiation (`**`)**: Raises the left operand to the power of the right operand. For instance, `5 ** 2` equals `25`.


## Variables

**Creating a variable**

Variables are names that represent a value. The operations of your program can change that value. 

In Python, the assignment operator `=` is used to assign a value to a variable. For example, `x = 10` assigns the value `10` to the variable `x`. This creates the variable `x`, or overwrites whatever value it previously held.

You can modify the value of a variable with these shorthand increment / decrement operators:

1. **Addition Assignment (`+=`)**: This operator adds the right operand to the left operand and then assigns the result to the left operand. For instance, `x += 3` is equivalent to `x = x + 3`. If `x` was `5`, it would now be `8`.

2. **Subtraction Assignment (`-=`)**: This operator subtracts the right operand from the left operand and then assigns the result to the left operand. For example, `x -= 2` is the same as `x = x - 2`. If `x` was initially `8`, it would now become `6`.

**Data Types**

Python variables can store different types of data. Today, we'll focus on three basic types:

1. **Integers (`int`)**: Whole numbers like `3`, `100`, `-1`.
2. **Floating Point Numbers (`float`)**: Numbers with a decimal point, like `3.14`, `-0.001`.
3. **Characters or Strings of Text (`str`)**: Text data enclosed in quotes, like `"Hello"`, `'Python'`.

Let's try creating variables of these types in a Jupyter Notebook.

```python
# Integer
my_integer = 10

# Float
my_float = 3.14

# String
my_string = "Hello, Python!"

# Displaying the values
print(my_integer)
print(my_float)
print(my_string)
```

use the built in function `type()` to ask Python what the current type of a variable is. There are no restrictions for a variable to always be a certain type - a Python variable has the type of whatever it's currently assigned to.

```python
type(my_integer) # returns `int`
type(my_float) # returns `float`
```


## Hands-On Project: Arithmetic in Python

Guess and then experiment on how operators and data types interact. Start with an int (`x = 4`), then a float (`y = 3.14`) then a string (`t = 'hi'`). Use each operator on the three variables with a constant or another variable of each of the types. If the operation is possible, note down what the data type of the result is. Save your results in new variables and convince yourself that you can store and retrieve values in Python.

**Integer tests**

```
| x = 4    | int | float | str |
|----------|-----|-------|-----|
| +        |     |       |     |
| -        |     |       |     |
| *        |     |       |     |
| /        |     |       |     |
| //       |     |       |     |
| **       |     |       |     |
| %        |     |       |     |
| =        |     |       |     |
| +=       |     |       |     |
| -=       |     |       |     |
```

**Floating point tests**

```
| y = 3.14 | int | float | str |
|----------|-----|-------|-----|
| +        |     |       |     |
| -        |     |       |     |
| *        |     |       |     |
| /        |     |       |     |
| //       |     |       |     |
| **       |     |       |     |
| %        |     |       |     |
| =        |     |       |     |
| +=       |     |       |     |
| -=       |     |       |     |
```

**String tests**

```
| t = 'hi' | int | float | str |
|----------|-----|-------|-----|
| +        |     |       |     |
| -        |     |       |     |
| *        |     |       |     |
| /        |     |       |     |
| //       |     |       |     |
| **       |     |       |     |
| %        |     |       |     |
| =        |     |       |     |
| +=       |     |       |     |
| -=       |     |       |     |
```

You can fill in each cell with specific examples or notes on how the operator behaves with each data type. For instance, the `+` operator can be used for addition with `int` and `float`, and for concatenation with `str`.


## Further Resources

Code Academy's introduction to Jupyter Notebooks: https://www.codecademy.com/article/introducing-jupyter-notebook

https://en.wikipedia.org/wiki/Python_syntax_and_semantics

