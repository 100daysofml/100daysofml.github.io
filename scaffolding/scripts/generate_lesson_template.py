#!/usr/bin/env python3
"""
Generate a basic Jupyter notebook template for a lesson.

This script creates a standardized lesson template with common sections
including introduction, theory, code examples, hands-on activities,
and further resources.

Usage:
    python generate_lesson_template.py <lesson_number> <lesson_topic> <week_number>

Example:
    python generate_lesson_template.py 5 "Data Structures" 2
"""

import json
import argparse
import sys
from pathlib import Path


def create_notebook_template(lesson_number, lesson_topic, week_number):
    """
    Create a Jupyter notebook template with standard sections.

    Args:
        lesson_number (int): The lesson number (e.g., 1, 5, 12)
        lesson_topic (str): The main topic of the lesson
        week_number (int): The week number (e.g., 1, 2, 7)

    Returns:
        dict: A dictionary representing the .ipynb JSON structure
    """

    # Create cells for the notebook
    cells = [
        # Introduction cell
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# Day {lesson_number}: {lesson_topic}\n",
                "\n",
                "## Introduction\n",
                "\n",
                f"<!-- TODO: Provide an engaging introduction to {lesson_topic} -->\n",
                "<!-- TODO: Explain why this topic is important -->\n",
                "<!-- TODO: Provide learning objectives -->\n",
            ],
            "id": "cell-introduction"
        },
        # Theory section
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Theory\n",
                "\n",
                f"<!-- TODO: Explain the core concepts of {lesson_topic} -->\n",
                "<!-- TODO: Add diagrams or mathematical formulas if needed -->\n",
                "<!-- TODO: Break down complex ideas into smaller parts -->\n",
            ],
            "id": "cell-theory"
        },
        # Imports cell
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import necessary libraries\n",
                "# TODO: Add relevant imports for this lesson\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
            ],
            "id": "cell-imports"
        },
        # Visualization example
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# TODO: Create visualizations related to the lesson\n",
                "# Example: matplotlib, seaborn, or plotly visualization\n",
            ],
            "id": "cell-visualization"
        },
        # Code examples
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# TODO: Add code examples demonstrating key concepts\n",
            ],
            "id": "cell-examples"
        },
        # Hands-on activity
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Hands-On Activity\n",
                "\n",
                f"<!-- TODO: Create a practical exercise to practice {lesson_topic} -->\n",
                "<!-- TODO: Provide step-by-step instructions -->\n",
                "<!-- TODO: Include starter code if needed -->\n",
            ],
            "id": "cell-activity"
        },
        # Hands-on code cell
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# TODO: Complete the hands-on activity here\n",
            ],
            "id": "cell-activity-code"
        },
        # Key takeaways
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Key Takeaways\n",
                "\n",
                "- TODO: Summarize main point 1\n",
                "- TODO: Summarize main point 2\n",
                "- TODO: Summarize main point 3\n",
            ],
            "id": "cell-takeaways"
        },
        # Further resources
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Further Resources\n",
                "\n",
                "- [TODO: Add resource title](https://example.com)\n",
                "- [TODO: Add another resource](https://example.com)\n",
                "- [TODO: Add documentation link](https://example.com)\n",
            ],
            "id": "cell-resources"
        }
    ]

    # Create the notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return notebook


def main():
    """Parse arguments and generate the notebook template."""
    parser = argparse.ArgumentParser(
        description="Generate a Jupyter notebook template for a lesson"
    )
    parser.add_argument(
        "lesson_number",
        type=int,
        help="The lesson number (e.g., 1, 5, 12)"
    )
    parser.add_argument(
        "lesson_topic",
        type=str,
        help="The main topic of the lesson (e.g., 'Python Basics')"
    )
    parser.add_argument(
        "week_number",
        type=int,
        help="The week number (e.g., 1, 2, 7)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.lesson_number < 1:
        print("Error: lesson_number must be >= 1", file=sys.stderr)
        sys.exit(1)

    if args.week_number < 0:
        print("Error: week_number must be >= 0", file=sys.stderr)
        sys.exit(1)

    if not args.lesson_topic.strip():
        print("Error: lesson_topic cannot be empty", file=sys.stderr)
        sys.exit(1)

    # Determine the content directory (parent of this script's parent's parent)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    content_dir = project_root / "content"

    # Create week directory
    week_dir = content_dir / f"Week_{args.week_number:02d}"
    week_dir.mkdir(parents=True, exist_ok=True)

    # Create lesson file path
    lesson_file = week_dir / f"Lesson_{args.lesson_number:02d}.ipynb"

    # Check if file already exists
    if lesson_file.exists():
        response = input(
            f"File {lesson_file} already exists. Overwrite? (y/n): "
        )
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    # Generate the notebook
    notebook = create_notebook_template(
        args.lesson_number,
        args.lesson_topic,
        args.week_number
    )

    # Write to file
    try:
        with open(lesson_file, 'w') as f:
            json.dump(notebook, f, indent=2)
        print(f"Successfully created: {lesson_file}")
        return 0
    except IOError as e:
        print(f"Error writing to file: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
