#!/usr/bin/env python3
"""
Quick Start Script for Lesson Setup - Cross-platform version
Automates the entire lesson creation workflow in a platform-independent way.

Usage:
    python quick_start.py <lesson_number>

Example:
    python quick_start.py 42
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import Tuple, Optional

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from suggest_topics import get_topic_for_lesson
from build_lesson_incrementally import IncrementalLessonBuilder


class Colors:
    """Terminal color support with fallback for systems without color support."""

    def __init__(self):
        self.enabled = self._should_use_colors()

    @staticmethod
    def _should_use_colors() -> bool:
        """Check if terminal supports colors."""
        return (
            sys.stdout.isatty() and
            os.environ.get('NO_COLOR') is None
        )

    @property
    def RED(self) -> str:
        return '\033[0;31m' if self.enabled else ''

    @property
    def GREEN(self) -> str:
        return '\033[0;32m' if self.enabled else ''

    @property
    def YELLOW(self) -> str:
        return '\033[1;33m' if self.enabled else ''

    @property
    def BLUE(self) -> str:
        return '\033[0;34m' if self.enabled else ''

    @property
    def CYAN(self) -> str:
        return '\033[0;36m' if self.enabled else ''

    @property
    def BOLD(self) -> str:
        return '\033[1m' if self.enabled else ''

    @property
    def RESET(self) -> str:
        return '\033[0m' if self.enabled else ''


class QuickStartLessonSetup:
    """Main class for lesson quick start workflow."""

    def __init__(self, lesson_number: int):
        """Initialize the quick start setup."""
        self.lesson_number = lesson_number
        self.colors = Colors()
        self.project_root = SCRIPT_DIR.parent.parent
        self.content_dir = self.project_root / "content"
        self.topic = ""
        self.week_number = 0
        self.lesson_file = None

    def print_header(self, text: str) -> None:
        """Print a formatted header."""
        print(f"{self.colors.BOLD}{self.colors.CYAN}{text}{self.colors.RESET}")

    def print_success(self, text: str) -> None:
        """Print a success message."""
        print(f"{self.colors.GREEN}✓ {text}{self.colors.RESET}")

    def print_info(self, text: str) -> None:
        """Print an info message."""
        print(f"{self.colors.BLUE}ℹ {text}{self.colors.RESET}")

    def print_warning(self, text: str) -> None:
        """Print a warning message."""
        print(f"{self.colors.YELLOW}⚠ {text}{self.colors.RESET}")

    def print_error(self, text: str) -> None:
        """Print an error message."""
        print(f"{self.colors.RED}✗ {text}{self.colors.RESET}", file=sys.stderr)

    def display_banner(self) -> None:
        """Display the welcome banner."""
        print()
        self.print_header("╔════════════════════════════════════════════════════════╗")
        self.print_header("║         100 Days of ML - Lesson Quick Start            ║")
        self.print_header("╚════════════════════════════════════════════════════════╝")
        print()

    def get_topic_suggestion(self) -> bool:
        """Get and display topic suggestion."""
        self.print_info(f"Fetching topic suggestion for lesson {self.lesson_number}...")

        result = get_topic_for_lesson(self.lesson_number)

        if "error" in result:
            self.print_error(f"Failed to get topic: {result['error']}")
            return False

        # Pretty print the suggestion
        print()
        print(f"{'='*60}")
        print(f"LESSON {result['lesson']} - Week {result['week']}, Day {result['day_in_week']}")
        print(f"{'='*60}")
        print(f"Theme: {result['theme']}")
        print(f"Difficulty: {result['difficulty']}")
        print(f"\nToday's Topic: {result['topic']}")
        print(f"\nThis Week's Topics:")
        for i, t in enumerate(result['all_week_topics'], 1):
            marker = "→" if t == result['topic'] else " "
            print(f"  {marker} Day {i}: {t}")
        print(f"\nPrerequisites: {', '.join(result['prerequisite_weeks'])}")
        print(f"{'='*60}")
        print()

        self.week_number = result['week']
        self.topic = result['topic']
        return True

    def confirm_or_customize_topic(self) -> bool:
        """Allow user to confirm or customize the topic."""
        self.print_info("Topic confirmation:")

        response = input(
            f"{self.colors.CYAN}Use suggested topic? [y/n]: {self.colors.RESET}"
        ).strip().lower()

        if response in ('y', 'yes'):
            self.print_success(f"Topic confirmed: {self.topic}")
            return True
        elif response in ('n', 'no'):
            custom_topic = input(
                f"{self.colors.CYAN}Enter custom topic: {self.colors.RESET}"
            ).strip()

            if not custom_topic:
                self.print_error("Topic cannot be empty")
                return False

            self.topic = custom_topic
            self.print_success(f"Topic set to: {self.topic}")
            return True
        else:
            self.print_warning("Invalid input. Using suggested topic.")
            return True

    def generate_template(self) -> bool:
        """Generate the lesson template."""
        self.print_info("Generating lesson template...")

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "generate_lesson_template.py"),
                    str(self.lesson_number),
                    self.topic,
                    str(self.week_number)
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                self.print_error("Failed to generate lesson template")
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
                return False

            # Extract path from output
            for line in result.stdout.split('\n'):
                if 'Successfully created:' in line:
                    path_str = line.split(':', 1)[1].strip()
                    self.lesson_file = Path(path_str)
                    break

            if not self.lesson_file or not self.lesson_file.exists():
                self.print_error("Lesson file was not created")
                return False

            self.print_success("Template generated successfully")
            return True

        except subprocess.TimeoutExpired:
            self.print_error("Template generation timed out")
            return False
        except Exception as e:
            self.print_error(f"Error generating template: {e}")
            return False

    def display_lesson_path(self) -> None:
        """Display the path to the lesson guide."""
        print()
        self.print_info("Lesson guide location:")
        print(f"{self.colors.BOLD}{self.lesson_file}{self.colors.RESET}")
        print()

    def launch_incremental_builder(self) -> bool:
        """Ask user if they want to launch the incremental builder."""
        response = input(
            f"{self.colors.CYAN}Open incremental builder? [y/n]: {self.colors.RESET}"
        ).strip().lower()

        if response in ('y', 'yes'):
            print()
            self.print_header("Opening Incremental Builder...")
            print()

            builder = IncrementalLessonBuilder(self.lesson_number)
            builder.interactive_menu()
            return True
        else:
            self.print_info("Skipping incremental builder")
            return False

    def run(self) -> int:
        """Execute the complete quick start workflow."""
        try:
            # Step 1: Display banner
            self.display_banner()

            # Step 2: Get topic suggestion
            if not self.get_topic_suggestion():
                return 1

            # Step 3: Confirm or customize topic
            if not self.confirm_or_customize_topic():
                return 1

            print()

            # Step 4: Generate template
            if not self.generate_template():
                return 1

            # Step 5: Display lesson path
            self.display_lesson_path()

            # Step 6: Launch incremental builder
            self.launch_incremental_builder()

            print()
            self.print_success("Lesson setup complete!")
            return 0

        except KeyboardInterrupt:
            print()
            self.print_warning("Setup cancelled by user")
            return 1
        except Exception as e:
            self.print_error(f"Unexpected error: {e}")
            return 1


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python quick_start.py <lesson_number>")
        print("Example: python quick_start.py 42")
        print()
        print("This script automates the lesson setup process:")
        print("  1. Suggests a topic based on the lesson number")
        print("  2. Allows you to confirm or customize the topic")
        print("  3. Generates a lesson template")
        print("  4. Opens the incremental builder for interactive editing")
        sys.exit(1)

    try:
        lesson_number = int(sys.argv[1])
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid lesson number", file=sys.stderr)
        sys.exit(1)

    if lesson_number < 1:
        print("Error: Lesson number must be positive", file=sys.stderr)
        sys.exit(1)

    setup = QuickStartLessonSetup(lesson_number)
    sys.exit(setup.run())


if __name__ == "__main__":
    main()
