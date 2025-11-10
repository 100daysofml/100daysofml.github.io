#!/usr/bin/env python3
"""
Incremental Lesson Builder - Build Jupyter notebooks in chunks to avoid context overflow.
Accepts a lesson number and provides an interactive menu to add sections incrementally.
"""

import json
import sys
import os
from pathlib import Path
from typing import Optional


class IncrementalLessonBuilder:
    """Build lessons by appending cells to notebooks without loading entire file into memory."""

    def __init__(self, lesson_number: int):
        """Initialize builder with lesson number."""
        self.lesson_number = lesson_number
        self.notebook_path = self._resolve_lesson_path()
        self.cells = []
        self.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}

    def _resolve_lesson_path(self) -> Path:
        """Find the lesson notebook path."""
        base_dir = Path(__file__).parent.parent.parent / "content"

        # Try to find Week/Lesson pattern
        for week_dir in sorted(base_dir.glob("Week_*")):
            if not week_dir.is_dir():
                continue
            lesson_file = week_dir / f"Lesson_{self.lesson_number}.ipynb"
            if lesson_file.exists():
                return lesson_file

        # If not found, suggest creating a new one
        week_num = (self.lesson_number - 1) // 7 + 1
        new_path = base_dir / f"Week_{week_num:02d}" / f"Lesson_{self.lesson_number}.ipynb"
        new_path.parent.mkdir(parents=True, exist_ok=True)
        return new_path

    def _load_existing_cells(self) -> None:
        """Load existing cells from notebook if it exists."""
        if not self.notebook_path.exists():
            return

        try:
            with open(self.notebook_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            self.cells = nb.get('cells', [])
            self.metadata = nb.get('metadata', self.metadata)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading notebook: {e}")

    def _save_notebook(self) -> None:
        """Save notebook to disk efficiently."""
        notebook = {
            "cells": self.cells,
            "metadata": self.metadata,
            "nbformat": 4,
            "nbformat_minor": 5
        }

        try:
            with open(self.notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=2)
            print(f"Notebook saved: {self.notebook_path}")
        except IOError as e:
            print(f"Error saving notebook: {e}")

    def _create_cell(self, cell_type: str, source: str) -> dict:
        """Create a notebook cell."""
        if cell_type == "markdown":
            return {
                "cell_type": "markdown",
                "metadata": {},
                "source": source.split('\n') if '\n' in source else [source]
            }
        else:  # code
            return {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.split('\n') if '\n' in source else [source]
            }

    def add_introduction(self) -> None:
        """Add introduction section."""
        title = input("Enter lesson title: ").strip()
        if not title:
            print("Title cannot be empty")
            return

        content = input("Enter introduction content (multiline, end with blank line):\n")
        lines = [content] if content else []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)

        full_content = '\n'.join(lines)
        self.cells.append(self._create_cell("markdown", f"# {title}"))
        self.cells.append(self._create_cell("markdown", full_content))
        print("Introduction section added.")

    def add_theory(self) -> None:
        """Add theory/concept section."""
        section_name = input("Enter section name: ").strip()
        if not section_name:
            print("Section name cannot be empty")
            return

        content = input("Enter theory content (multiline, end with blank line):\n")
        lines = [content] if content else []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)

        full_content = '\n'.join(lines)
        self.cells.append(self._create_cell("markdown", f"## {section_name}"))
        self.cells.append(self._create_cell("markdown", full_content))
        print("Theory section added.")

    def add_imports(self) -> None:
        """Add code imports section."""
        print("Enter import statements (one per line, empty line to finish):")
        imports = []
        while True:
            line = input().strip()
            if not line:
                break
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
            else:
                print("Warning: Not a valid import statement. Skipping.")

        if imports:
            code_content = '\n'.join(imports)
            self.cells.append(self._create_cell("code", code_content))
            print(f"Added {len(imports)} import statement(s).")

    def add_visualization(self) -> None:
        """Add visualization section."""
        title = input("Enter visualization title: ").strip()
        if not title:
            print("Title cannot be empty")
            return

        self.cells.append(self._create_cell("markdown", f"## {title}"))

        code = input("Enter visualization code (multiline, end with blank line):\n")
        lines = [code] if code else []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)

        full_code = '\n'.join(lines)
        if full_code.strip():
            self.cells.append(self._create_cell("code", full_code))
            print("Visualization section added.")

    def add_activity(self) -> None:
        """Add hands-on activity section."""
        title = input("Enter activity title: ").strip()
        if not title:
            print("Title cannot be empty")
            return

        description = input("Enter activity description (multiline, end with blank line):\n")
        lines = [description] if description else []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)

        full_description = '\n'.join(lines)
        self.cells.append(self._create_cell("markdown", f"## {title}"))
        self.cells.append(self._create_cell("markdown", full_description))

        code = input("Enter starter code (optional, multiline, end with blank line):\n")
        code_lines = [code] if code else []
        while True:
            line = input()
            if not line:
                break
            code_lines.append(line)

        full_code = '\n'.join(code_lines)
        if full_code.strip():
            self.cells.append(self._create_cell("code", full_code))

        print("Activity section added.")

    def add_takeaways(self) -> None:
        """Add key takeaways section."""
        print("Enter key takeaways (one per line, empty line to finish):")
        takeaways = []
        while True:
            line = input("- ").strip()
            if not line:
                break
            takeaways.append(f"- {line}")

        if takeaways:
            content = '\n'.join(takeaways)
            self.cells.append(self._create_cell("markdown", "## Key Takeaways"))
            self.cells.append(self._create_cell("markdown", content))
            print(f"Added {len(takeaways)} takeaway(s).")

    def add_resources(self) -> None:
        """Add resources section."""
        print("Enter resources (one per line, empty line to finish):")
        resources = []
        while True:
            line = input().strip()
            if not line:
                break
            resources.append(f"- {line}")

        if resources:
            content = '\n'.join(resources)
            self.cells.append(self._create_cell("markdown", "## Resources"))
            self.cells.append(self._create_cell("markdown", content))
            print(f"Added {len(resources)} resource(s).")

    def execute_all_cells(self) -> None:
        """Execute all code cells using nbconvert (requires jupyter/nbconvert)."""
        try:
            import subprocess
            result = subprocess.run(
                ['jupyter', 'nbconvert', '--to', 'notebook', '--execute', str(self.notebook_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                print("All cells executed successfully.")
            else:
                print(f"Execution failed: {result.stderr}")
        except (FileNotFoundError, ImportError):
            print("Jupyter/nbconvert not available. Install with: pip install jupyter nbconvert")
        except subprocess.TimeoutExpired:
            print("Execution timed out.")
        except Exception as e:
            print(f"Error executing cells: {e}")

    def interactive_menu(self) -> None:
        """Display interactive menu for lesson building."""
        self._load_existing_cells()

        menu_options = {
            "1": ("Add Introduction", self.add_introduction),
            "2": ("Add Theory Section", self.add_theory),
            "3": ("Add Code Imports", self.add_imports),
            "4": ("Add Visualization", self.add_visualization),
            "5": ("Add Hands-on Activity", self.add_activity),
            "6": ("Add Key Takeaways", self.add_takeaways),
            "7": ("Add Resources", self.add_resources),
            "8": ("Execute All Cells", self.execute_all_cells),
            "9": ("Save and Exit", lambda: None)
        }

        print(f"\n{'='*50}")
        print(f"Building Lesson {self.lesson_number}")
        print(f"Path: {self.notebook_path}")
        print(f"Current cells: {len(self.cells)}")
        print(f"{'='*50}\n")

        while True:
            print("\nOptions:")
            for key, (name, _) in menu_options.items():
                print(f"  {key}. {name}")

            choice = input("\nSelect option (1-9): ").strip()

            if choice not in menu_options:
                print("Invalid option. Please try again.")
                continue

            name, func = menu_options[choice]

            if choice == "9":
                self._save_notebook()
                print("Lesson building complete. Exiting.")
                break

            try:
                func()
                self._save_notebook()
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python build_lesson_incrementally.py <lesson_number>")
        print("Example: python build_lesson_incrementally.py 8")
        sys.exit(1)

    try:
        lesson_num = int(sys.argv[1])
        if lesson_num < 1:
            print("Lesson number must be positive")
            sys.exit(1)
    except ValueError:
        print("Lesson number must be an integer")
        sys.exit(1)

    builder = IncrementalLessonBuilder(lesson_num)
    builder.interactive_menu()


if __name__ == "__main__":
    main()
