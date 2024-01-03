# Day 3: Control Structures - Loops

Loops are fundamental structures in programming that allow us to execute a block of code repeatedly, under certain conditions. They are essential for automating repetitive tasks, processing collections of data, and building the backbone of algorithms. Understanding loops and how to control them is crucial for writing efficient and effective code.

**This notebook is available for you to follow along in Colab:** [Link to Colab notebook](https://colab.research.google.com/drive/13h2TzSIfosqVcgljj7ps2zS7Bu8HH8O1?usp=sharing)

## Why Loop?

Loops are necessary for efficient programming as they enable us to avoid repetitive code, making our scripts shorter and more readable. They are particularly valuable when dealing with large datasets or when a task needs to be repeated multiple times under certain conditions. Effective use of loops leads to cleaner, more organized, and more maintainable code.

## Collection Data Types

- **List Type**: In Python, a list is an ordered collection of items which can be of varied types. Lists are mutable, meaning their elements can be changed after they are created.

  ```python
  my_list = [1, 2, 3, "Python", True]
  ```

- **Dictionary Type**: A dictionary in Python is an unordered collection of data in a key:value pair form. It's an efficient way to store and retrieve data based on a key.

  ```python
  my_dict = {"name": "Alice", "age": 25, "language": "Python"}
  ```

## The `while` Loop

- The `while` loop in Python executes a block of code as long as a specified condition is true. It's useful when the number of iterations is not known beforehand.

  ```python
  count = 0
  while count < 5:
      print(count)
      count += 1
  ```

- **Warning About Infinite Loops**: Be cautious with `while` loops, as they can lead to infinite loops if the condition never becomes false. Always ensure the loop has a clear end point.

  ```python
  # Warning: This is an example of an infinite loop
  # while True:
  #    print("Infinite loop")
  ```

- **Difficulty with Dictionaries**: Iterating over dictionaries with `while` loops is less straightforward because dictionaries are not inherently ordered, and the loop requires external manipulation of the keys or values.

## The `for` Loop

- The `for` loop is used for iterating over a sequence, such as a list, tuple, dictionary, or string. It's more concise and preferable when the number of iterations is known or defined by the collection.

  ```python
  for item in my_list:
      print(item)
  ```

- **Iterating Through a List**: Looping through a list is straightforward as it maintains order.

  ```python
  for element in my_list:
      print(element)
  ```

- **Iterating Through a Dictionary**: Iterating through a dictionary can be done over keys, values, or key-value pairs.

  ```python
  for key, value in my_dict.items():
      print(f"{key}: {value}")
  ```

- **Preferred Method**: `for` loops are often preferred over `while` loops because they are less prone to errors like infinite loops and are generally more readable, especially when iterating over objects.

### Range Objects

- The Python `range` object generates a sequence of numbers. It is often used in `for` loops to specify the number of iterations. The `range` function can take one, two, or three arguments: start, stop, and step.

  ```python
  for i in range(5):
      print(i)
  ```


## Understanding "break", "continue", and the "else" Clause in Loops

Loops in Python become even more powerful and flexible when combined with `break`, `continue`, and the `else` clause. These control statements allow for more nuanced and controlled execution of loops.

### The `break` Statement
- **Purpose**: The `break` statement is used to exit a loop prematurely, regardless of the iteration condition.
- **Usage in `for` and `while` Loops**:
  - In a `for` loop, `break` can be used to exit the loop if a certain condition is met.
  - In a `while` loop, it can serve as an exit point that’s more complex than the loop’s condition itself.
- **Example**:
  ```python
  for number in range(1, 10):
      if number == 5:
          break
      print(number)
  # This will print numbers 1 through 4 and then exit the loop.
  ```

### The `continue` Statement
- **Purpose**: The `continue` statement skips the current iteration and moves onto the next iteration of the loop.
- **Usage in `for` and `while` Loops**:
  - In both types of loops, `continue` can be used to bypass parts of the loop body for certain iterations.
- **Example**:
  ```python
  for number in range(1, 10):
      if number % 2 == 0:
          continue
      print(number)
  # This will print only odd numbers between 1 and 9.
  ```

### The `else` Clause of a `for` Loop
- **Purpose**: The `else` clause in a `for` loop executes after the loop finishes its iterations, but only if the loop was not terminated by a `break`.
- **Usage**:
  - This is particularly useful when you are searching for an item in a collection and want to execute some code if the item was not found.
- **Example**:
  ```python
  for number in range(1, 5):
      if number == 6:
          break
  else:
      print("Number 6 not found in range.")
  # The else clause will execute here, as the break was not triggered.
  ```

## Project: Combining Arithmetic, Boolean Logic, and Loops

Work through these programs. Convince yourself you understand them and how they work. Think about the role that the loops play and how you might employ a different type of loop to accomplish the same task.

### Example Program 1: Prime Number Checker

**Problem Statement**:
Create a program that checks whether a number is a prime number. A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.

**Solution**:
```python
def is_prime(number):
    if number <= 1:
        return False
    for i in range(2, number):
        if number % i == 0:
            return False
    return True

# Test the function
num = 29
if is_prime(num):
    print(f"{num} is a prime number.")
else:
    print(f"{num} is not a prime number.")
```

### Example Program 2: Basic Arithmetic Quiz

**Problem Statement**:
Create a simple arithmetic quiz program that asks the user to solve a series of basic math problems (addition, subtraction, multiplication). The program should:
- Present a new problem in each iteration of the loop.
- Ask the user for their answer.
- Use boolean logic to check if the answer is correct.
- Give immediate feedback (correct/incorrect) and provide the correct answer if incorrect.
- End the quiz after a certain number of questions.

#### Solution:

```python
import random

# Set the number of questions
num_questions = 5

for i in range(num_questions):
    # Generate two random numbers and a random operation
    num1 = random.randint(1, 10)
    num2 = random.randint(1, 10)
    operation = random.choice(['+', '-', '*'])
    correct_answer = 0

    # Calculate the correct answer
    if operation == '+':
        correct_answer = num1 + num2
    elif operation == '-':
        correct_answer = num1 - num2
    elif operation == '*':
        correct_answer = num1 * num2

    # Ask the user for their answer
    user_answer = int(input(f"Question {i+1}: What is {num1} {operation} {num2}? "))

    # Check the answer and give feedback
    if user_answer == correct_answer:
        print("Correct!")
    else:
        print(f"Incorrect. The correct answer was {correct_answer}.")
```

### Your Task:
Your goal is to create a similar program that quizzes the user on simple arithmetic problems. Feel free to modify the range of numbers, types of operations, or the number of questions. This exercise will help reinforce your understanding of loops, basic arithmetic operations, and boolean logic in Python. Remember, the key is to write code that repeatedly presents questions and gives feedback based on the user's input.

## Further Resources


