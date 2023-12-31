# Day 4: Control Structures - Conditionals

Conditional statements, commonly known as "if statements," are a fundamental aspect of programming, allowing the code to execute different actions depending on certain conditions. This concept, often referred to as "branching," enables programs to make decisions, thereby increasing their complexity and capability.

## If This, Then That

The `if` statement is the most basic form of conditional execution in Python. It tests a condition and executes a block of code if the condition is true. The syntax requires an indentation block, which defines the scope of the statement.

```python
if condition:
    # This code runs if condition is True
    do_something()
```

## Otherwise, Do This!

The `else` statement is used in conjunction with `if`. It defines a block of code that is executed when the `if` condition is not met.

```python
if condition:
    do_something()
else:
    # This code runs if condition is False
    do_something_else()
```

## Code Block of Last Resort

The `elif` (short for "else if") statement is used to check multiple expressions for truth value and execute a block of code as soon as one of the conditions evaluates to True. It's a way to handle multiple, mutually exclusive conditions.

```python
if condition1:
    do_something()
elif condition2:
    do_something_else()
else:
    # This runs if neither condition1 nor condition2 is True
    do_another_thing()
```

`if`/`elif` statements in Python can be likened to a C-style `switch` block, where different cases are checked, and corresponding actions are taken based on which case is true.

## Saving on Words: Succinct Versions

In Python, `if` statements can be written in a more compact form, especially when the action to be executed is short.

```python
if condition: do_something()
```

There is also the ternary expression in Python, which allows for a quick evaluation of a condition in a single line. It's a concise way of writing an `if-else` statement.

```python
result = a if condition else b
```

This expression assigns `a` to `result` if `condition` is True, and `b` otherwise. Understanding and using these forms of conditional statements enhance the readability and efficiency of your code, especially in scenarios where decision-making is a key aspect of the program logic.

## Lesson Project: FizzBuzz and a Similar Challenge

### FizzBuzz Problem Statement
FizzBuzz is a classic programming task, often used in job interviews to assess basic programming skills. The challenge is as follows:
- Print numbers from 1 to 100.
- For multiples of 3, print "Fizz" instead of the number.
- For multiples of 5, print "Buzz" instead of the number.
- For numbers which are multiples of both 3 and 5, print "FizzBuzz".

### FizzBuzz Solution in Python

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

## Alternative Lesson Project: Temperature Analyzer

### Temperature Analyzer Problem Statement
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

### Your Challenge: Daily Step Counter

**Problem Statement**:
Write a program that interprets a list of daily step counts. For each day's step count:
- If the steps are less than 5,000, categorize as "Sedentary".
- If the steps are between 5,000 and 7,499 (inclusive), categorize as "Lightly Active".
- If the steps are between 7,500 and 9,999 (inclusive), categorize as "Moderately Active".
- If the steps are 10,000 or more, categorize as "Very Active".
- Print the day number and the corresponding activity category.

## Further Resources






**Lesson 5: Functions and Modules**

- **Introduction to Functions**
  - *Definition and Purpose of Functions*: Functions are useful in two distinct scenarios: To repeat behavior without repeating an entire procedure, or to vary behavior based on the situation at hand. In programming, a function is a block of organized, reusable code used to perform a single, related action. They provide better modularity for your application and a high degree of code reusing.
  - *Basic Structure of a Function in Python*: In Python, a function is defined using the `def` keyword, followed by a function name and parentheses `()`. The code block within every function starts with a colon `:` and is indented.
  - *Parameters and Arguments*: Parameters are variables listed inside the parentheses in the function definition. Arguments are the values passed to these parameters. Arguments are specified after the function name, inside the parentheses.
  - *Return Values*: Functions can return values using the `return` statement. The return value can be any object, and a function can return multiple values.

- **Writing Your First Function**
  - *Using `def` Keyword*: To create a function, use the `def` keyword, followed by a chosen function name and parentheses. Example: `def my_function():`.
  - *Function Naming Conventions*: Function names should be lowercase, with words separated by underscores as necessary to improve readability.
  - *Calling a Function*: To call a function, use the function name followed by parentheses. Example: `my_function()`.

- **Parameters and Arguments**
  - *Understanding Local Variables*: Variables declared inside a function are called local variables. Their scope is limited to the function, meaning they can't be accessed outside of it.
  - *Positional vs. Keyword Arguments*: Positional arguments are arguments that need to be included in the proper position or order. Keyword arguments are arguments accompanied by an identifier (e.g., `name='John'`) and can be listed in any order.
  - *Using Keywords to Reorder Arguments*: With keyword arguments, the order of arguments can be shuffled, allowing more flexibility.
  - *Default Parameter Values*: Functions can have parameters with default values. If the argument is not provided, the function uses the default value.

- **Return Values and `return` Statement**
  - *How to Return Data from a Function*: To send back a result from a function to its caller, use the `return` statement.
  - *Multiple Return Values; Tuple Value Unpacking*: Functions in Python can return multiple values, usually in the form of a tuple, which can be unpacked into multiple variables.
  - *None: The Default Return Value*: If no return statement is used, or the return statement does not have an accompanying value, the function returns `None`.

- **Scope and Lifetime of Variables**
  - *Local vs. Global Variables*: Global variables are defined outside a function and can be accessed throughout the program, while local variables are confined within the function they are declared in.
  - *Variable Shadowing - Don't Do This*: Avoid using the same name for local and global variables as it can lead to confusing "shadowing" effects, where the local variable "shadows" the global variable.
