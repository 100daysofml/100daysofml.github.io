#!/bin/bash
#
# Quick Start Script for Lesson Setup
# Automates the entire lesson creation workflow
#
# Usage: ./quick_start.sh <lesson_number>
# Example: ./quick_start.sh 42

set -e

# Color definitions (detect terminal support)
if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    RESET='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    BOLD=''
    RESET=''
fi

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
SUGGEST_SCRIPT="$SCRIPT_DIR/suggest_topics.py"
GENERATE_SCRIPT="$SCRIPT_DIR/generate_lesson_template.py"
BUILD_SCRIPT="$SCRIPT_DIR/build_lesson_incrementally.py"

# Functions
print_header() {
    echo -e "${BOLD}${CYAN}$1${RESET}"
}

print_success() {
    echo -e "${GREEN}✓ $1${RESET}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${RESET}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${RESET}"
}

print_error() {
    echo -e "${RED}✗ $1${RESET}" >&2
}

show_usage() {
    echo "Usage: $0 <lesson_number>"
    echo "Example: $0 42"
    echo ""
    echo "This script automates the lesson setup process:"
    echo "  1. Suggests a topic based on the lesson number"
    echo "  2. Allows you to confirm or customize the topic"
    echo "  3. Generates a lesson template"
    echo "  4. Opens the incremental builder for interactive editing"
}

# Validate inputs
if [ $# -eq 0 ]; then
    print_error "Missing lesson number argument"
    show_usage
    exit 1
fi

if [[ ! "$1" =~ ^[0-9]+$ ]]; then
    print_error "Lesson number must be a positive integer"
    exit 1
fi

LESSON_NUMBER=$1

# Verify scripts exist
if [ ! -f "$SUGGEST_SCRIPT" ]; then
    print_error "suggest_topics.py not found at $SUGGEST_SCRIPT"
    exit 1
fi

if [ ! -f "$GENERATE_SCRIPT" ]; then
    print_error "generate_lesson_template.py not found at $GENERATE_SCRIPT"
    exit 1
fi

if [ ! -f "$BUILD_SCRIPT" ]; then
    print_error "build_lesson_incrementally.py not found at $BUILD_SCRIPT"
    exit 1
fi

# Step 1: Display header
echo ""
print_header "╔════════════════════════════════════════════════════════╗"
print_header "║         100 Days of ML - Lesson Quick Start            ║"
print_header "╚════════════════════════════════════════════════════════╝"
echo ""

# Step 2: Get topic suggestion
print_info "Fetching topic suggestion for lesson $LESSON_NUMBER..."
SUGGESTION=$(python3 "$SUGGEST_SCRIPT" "$LESSON_NUMBER" 2>/dev/null)

if [ $? -ne 0 ]; then
    print_error "Failed to get topic suggestion. Make sure lesson_number is between 36-100"
    exit 1
fi

echo "$SUGGESTION"
echo ""

# Extract week number from suggestion
WEEK_NUMBER=$(python3 -c "
import json
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from suggest_topics import get_topic_for_lesson
result = get_topic_for_lesson($LESSON_NUMBER)
if 'week' in result:
    print(result['week'])
else:
    sys.exit(1)
" 2>/dev/null)

if [ -z "$WEEK_NUMBER" ]; then
    print_error "Failed to calculate week number"
    exit 1
fi

# Step 3: Ask for topic confirmation or custom input
echo ""
print_info "Topic confirmation:"
read -p "$(echo -e ${CYAN}Use suggested topic? [y/n]:${RESET} )" -n 1 -r
echo ""

TOPIC=""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    TOPIC=$(python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from suggest_topics import get_topic_for_lesson
result = get_topic_for_lesson($LESSON_NUMBER)
if 'topic' in result:
    print(result['topic'])
" 2>/dev/null)
else
    read -p "$(echo -e ${CYAN}Enter custom topic:${RESET} ) " TOPIC
fi

if [ -z "$TOPIC" ]; then
    print_error "Topic cannot be empty"
    exit 1
fi

print_success "Topic set to: $TOPIC"
echo ""

# Step 4: Generate template
print_info "Generating lesson template..."
python3 "$GENERATE_SCRIPT" "$LESSON_NUMBER" "$TOPIC" "$WEEK_NUMBER" 2>&1 | grep -E "Successfully|Error" || true

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    print_error "Failed to generate lesson template"
    exit 1
fi

# Get the lesson file path
LESSON_FILE="$PROJECT_ROOT/content/Week_$(printf '%02d' $WEEK_NUMBER)/Lesson_$(printf '%02d' $LESSON_NUMBER).ipynb"

if [ ! -f "$LESSON_FILE" ]; then
    print_error "Lesson file was not created at expected location"
    exit 1
fi

print_success "Template generated successfully"
echo ""

# Step 5: Display lesson guide path
print_info "Lesson guide location:"
echo -e "${BOLD}$LESSON_FILE${RESET}"
echo ""

# Step 6: Ask if user wants to continue with incremental builder
read -p "$(echo -e ${CYAN}Open incremental builder? [y/n]:${RESET} ) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    print_header "Opening Incremental Builder..."
    echo ""
    python3 "$BUILD_SCRIPT" "$LESSON_NUMBER"
else
    print_info "Skipping incremental builder"
    echo ""
    print_success "Lesson setup complete!"
    echo "  Next: python3 $BUILD_SCRIPT $LESSON_NUMBER"
    echo "  To edit: Edit the notebook at: $LESSON_FILE"
fi

echo ""
