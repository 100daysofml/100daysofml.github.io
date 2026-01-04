# Scaffolding: Context-Efficient Lesson Creation

## Purpose

This scaffolding framework exists to enable creation of complex ML lessons (37, 38, 60, 70, 71, 82, 87, 90, 91, 93, 95, 100) **without freezing Claude Code**. These lessons are context-intensive and would cause token budget overflows if created in a single session. This directory provides tools and guides for incremental, manageable lesson development.

## Problem Statement

Creating comprehensive ML lessons for advanced topics causes Claude Code to:
- Exceed context window limits (200k token budget)
- Freeze or become unresponsive
- Require session restarts
- Force developers to context-switch between incomplete sections

**Solution:** Break lesson creation into atomic, scripted steps that can be executed independently.

## Solution: Incremental Building Approach

Rather than creating entire lessons at once, we use:
1. **Lesson Guides** - Pre-written topic outlines and recommended structures
2. **Template Generation** - Scripts that create skeleton notebooks
3. **Incremental Building** - Interactive tools for adding content piece-by-piece
4. **Topic Suggestion** - Curriculum mapping to determine what to teach

This approach keeps any single task below 20-30k tokens, preventing context overflow.

---

## Directory Structure

```
scaffolding/
├── README.md                          # This file
├── lesson_guides/                     # Pre-written guides for each lesson
│   ├── lesson_37_guide.md
│   ├── lesson_38_guide.md
│   ├── lesson_60_guide.md
│   ├── lesson_70_guide.md
│   ├── lesson_71_guide.md
│   ├── lesson_82_guide.md
│   ├── lesson_87_guide.md
│   ├── lesson_90_guide.md
│   ├── lesson_91_guide.md
│   ├── lesson_93_guide.md
│   ├── lesson_95_guide.md
│   └── lesson_100_guide.md
└── scripts/                           # Python helper scripts
    ├── suggest_topics.py              # Curriculum topic lookup
    ├── generate_lesson_template.py    # Create skeleton notebook
    └── build_lesson_incrementally.py  # Interactive incremental builder
```

### Subdirectory Details

**`lesson_guides/`** - Contains detailed lesson outlines
- Each guide covers lesson context, prerequisites, and recommended structure
- Includes timing estimates for each section
- Lists prerequisite knowledge
- Suggests advanced topics and extensions
- **Use this:** Read before starting any lesson creation

**`scripts/`** - Executable Python utilities
- All scripts can run from anywhere with `python` or `python3`
- Support lesson numbers 1-100
- Include error handling and validation
- Provide helpful usage messages

---

## Quick Start Guide

### For Creating Any Lesson (37, 38, 60, 70, 71, 82, 87, 90, 91, 93, 95, 100)

#### Step 1: Determine Your Lesson Topic
```bash
cd /home/user/100daysofml.github.io/scaffolding
python scripts/suggest_topics.py 37
```
Output shows the week, theme, difficulty, and prerequisites.

#### Step 2: Read the Lesson Guide
```bash
# Read the guide that corresponds to your lesson number
cat lesson_guides/lesson_37_guide.md
```
This gives you structure, timing, and specific content recommendations.

#### Step 3: Generate Template (Optional)
If you prefer starting from scratch with imports/structure:
```bash
python scripts/generate_lesson_template.py 37 "Fine-tuning pre-trained models" 8
```
Creates a skeleton notebook with standard sections and TODO markers.

#### Step 4: Build Content Incrementally
```bash
python scripts/build_lesson_incrementally.py 37
```
Interactive menu for adding sections one at a time:
- Add Introduction
- Add Theory Section
- Add Code Imports
- Add Visualization
- Add Hands-on Activity
- Add Key Takeaways
- Add Resources
- Execute All Cells
- Save and Exit

#### Step 5: Execute and Validate
Within the interactive builder (option 8):
```
Select option (1-9): 8
```
Runs all code cells to check for errors before committing.

#### Step 6: Commit Your Work
```bash
cd /home/user/100daysofml.github.io
git add content/Week_08/Lesson_37.ipynb
git commit -m "Add Lesson 37: Fine-tuning pre-trained models"
```

---

## Script Documentation

### 1. `suggest_topics.py` - Curriculum Topic Lookup

**Purpose:** Map lesson numbers to recommended topics, difficulty levels, and prerequisites.

**Usage:**
```bash
python suggest_topics.py <lesson_number>
python suggest_topics.py --list
```

**Examples:**
```bash
# Get topic for lesson 37
python suggest_topics.py 37

# Output shows:
# LESSON 37 - Week 8, Day 2
# Theme: Transfer Learning & Domain Adaptation
# Difficulty: Intermediate
# Today's Topic: Fine-tuning pre-trained models
# Prerequisites: Deep Learning fundamentals, CNN and RNN basics

# List all lessons and topics
python suggest_topics.py --list
```

**Return Value:** JSON-like structure with:
- `lesson` - Lesson number
- `week` - Curriculum week (8-20)
- `day_in_week` - Position within week (1-5)
- `theme` - Weekly theme
- `topic` - Specific suggested topic
- `difficulty` - Level (Intermediate/Advanced/Expert)
- `prerequisite_weeks` - Required prior knowledge

---

### 2. `generate_lesson_template.py` - Skeleton Notebook Creator

**Purpose:** Create a Jupyter notebook scaffold with standard sections and TODO markers.

**Usage:**
```bash
python generate_lesson_template.py <lesson_number> "<lesson_topic>" <week_number>
```

**Example:**
```bash
python generate_lesson_template.py 37 "Fine-tuning pre-trained models" 8

# Creates: content/Week_08/Lesson_37.ipynb
# With sections: Introduction, Theory, Imports, Visualization, Code Examples,
#                Hands-on Activity, Key Takeaways, Further Resources
```

**What It Creates:**
- Markdown cells with section headers and TODO comments
- Code cells with placeholder imports (numpy, pandas, matplotlib)
- Proper Jupyter notebook JSON structure
- Ready-to-edit template

**When to Use:**
- Starting a lesson from scratch
- Need standardized structure
- Prefer template-guided approach

---

### 3. `build_lesson_incrementally.py` - Interactive Builder

**Purpose:** Add content to notebooks section-by-section without loading entire file.

**Usage:**
```bash
python build_lesson_incrementally.py <lesson_number>
```

**Example:**
```bash
python build_lesson_incrementally.py 37

# Interactive Menu:
# Building Lesson 37
# Path: /path/to/content/Week_08/Lesson_37.ipynb
# Current cells: 8
#
# Options:
#   1. Add Introduction
#   2. Add Theory Section
#   3. Add Code Imports
#   4. Add Visualization
#   5. Add Hands-on Activity
#   6. Add Key Takeaways
#   7. Add Resources
#   8. Execute All Cells
#   9. Save and Exit
#
# Select option (1-9): 1
```

**Key Features:**
- **Memory Efficient:** Loads and saves incrementally
- **Safe:** Saves after each operation
- **Guided:** Text prompts for each section
- **Executable:** Can run code cells to verify syntax

**Typical Workflow:**
```bash
# Start with template
python scripts/generate_lesson_template.py 37 "Fine-tuning Models" 8

# Then enhance with interactive builder
python scripts/build_lesson_incrementally.py 37

# Add Introduction
1
# Then Theory
2
# Then Imports
3
# etc...
```

---

## Workflow: Recommended Approach

### Complete Lesson Creation (25-40 minutes total)

**Session 1 (5-10 minutes): Planning**
```bash
# Step 1: Get topic suggestion
cd /home/user/100daysofml.github.io/scaffolding
python scripts/suggest_topics.py 37

# Step 2: Read the comprehensive guide
cat lesson_guides/lesson_37_guide.md

# Note down:
# - Key concepts to cover
# - Time allocation per section
# - Prerequisites to mention
# - Recommended libraries/datasets
```

**Session 2 (15-20 minutes): Content Creation**
```bash
# Step 3: Generate template
python scripts/generate_lesson_template.py 37 "Fine-tuning pre-trained models" 8

# Step 4: Build incrementally
python scripts/build_lesson_incrementally.py 37

# Add each section following the guide structure:
# - Introduction (motivation, learning objectives)
# - Theory (core concepts, formulas, diagrams)
# - Imports (relevant libraries)
# - Visualization (matplotlib/seaborn examples)
# - Activity (practical exercise)
# - Takeaways (summary)
# - Resources (links, references)
```

**Session 3 (5-10 minutes): Validation & Commit**
```bash
# Step 5: Execute code
python scripts/build_lesson_incrementally.py 37
# Select option 8 to execute all cells

# Step 6: Verify in Jupyter
jupyter notebook content/Week_08/Lesson_37.ipynb

# Step 7: Commit
git add content/Week_08/Lesson_37.ipynb
git commit -m "Add Lesson 37: Fine-tuning pre-trained models

- Covers transfer learning concept
- Includes practical PyTorch/TensorFlow examples
- Hands-on activity with real dataset
- References and further resources included"

git push
```

### Token Budget Breakdown

Each step stays within context limits:
- suggest_topics.py - ~5k tokens
- Reading guide - ~8k tokens
- generate_template - ~10k tokens
- Building incrementally - ~15k per section
- Execution/Validation - ~10k tokens

**Total:** ~50-60k tokens spread across 3 sessions = sustainable

---

## Lesson Guide Index

Complete list of all 12 lessons with topics:

| Lesson | Week | Theme | Topic | Difficulty |
|--------|------|-------|-------|------------|
| 37 | 8 | Transfer Learning & Domain Adaptation | Fine-tuning pre-trained models | Intermediate |
| 38 | 8 | Transfer Learning & Domain Adaptation | Feature extraction vs end-to-end learning | Intermediate |
| 60 | 12 | Reinforcement Learning | Multi-agent reinforcement learning | Advanced |
| 70 | 14 | Time Series & Sequence Modeling | Attention-based sequence models and Temporal Fusion | Advanced |
| 71 | 15 | AutoML & Hyperparameter Tuning | Grid search, random search, and Bayesian optimization | Advanced |
| 82 | 17 | MLOps & Production ML | Data versioning and DVC | Expert |
| 87 | 18 | Graph Neural Networks | Message passing and graph embedding | Expert |
| 90 | 18 | Graph Neural Networks | Knowledge graphs and link prediction | Expert |
| 91 | 19 | Federated & Distributed Learning | Federated learning fundamentals | Expert |
| 93 | 19 | Federated & Distributed Learning | Privacy-preserving machine learning | Expert |
| 95 | 19 | Federated & Distributed Learning | Distributed training and gradient synchronization | Expert |
| 100 | 20 | Advanced Optimization & Research | Neural ODE and continuous learning | Expert |

**Guide Files Available:**
- `/scaffolding/lesson_guides/lesson_37_guide.md`
- `/scaffolding/lesson_guides/lesson_38_guide.md`
- `/scaffolding/lesson_guides/lesson_60_guide.md`
- `/scaffolding/lesson_guides/lesson_70_guide.md`
- `/scaffolding/lesson_guides/lesson_71_guide.md`
- `/scaffolding/lesson_guides/lesson_82_guide.md`
- `/scaffolding/lesson_guides/lesson_87_guide.md`
- `/scaffolding/lesson_guides/lesson_90_guide.md`
- `/scaffolding/lesson_guides/lesson_91_guide.md`
- `/scaffolding/lesson_guides/lesson_93_guide.md`
- `/scaffolding/lesson_guides/lesson_95_guide.md`
- `/scaffolding/lesson_guides/lesson_100_guide.md`

---

## Tips & Best Practices

### Avoiding Context Overflow

1. **Use Multiple Sessions**
   - Planning session: read guide + suggest topics (~10k tokens)
   - Content session: build incrementally (~20-30k tokens per section)
   - Validation session: test + commit (~10k tokens)
   - Each session stays well under 50k tokens

2. **Follow the Guide Structure**
   - Each guide has estimated timing per section
   - Sections are designed to be self-contained
   - Prerequisite knowledge already mapped

3. **Test Early, Test Often**
   - Use option 8 (Execute All Cells) frequently
   - Fix errors immediately before moving to next section
   - Verify outputs match expected behavior

4. **Keep Code Sections Focused**
   - One concept per code cell
   - Include explanatory comments
   - Use meaningful variable names

5. **Reference the Curriculum Map**
   - Run `python scripts/suggest_topics.py --list` to see all 65 lessons (36-100)
   - Week themes and difficulty inform content depth
   - Prerequisites help identify what to assume students know

### Content Quality Checklist

- [ ] Read lesson guide before starting
- [ ] Follow recommended section structure and timing
- [ ] Include prerequisite review in introduction
- [ ] Provide concrete, runnable code examples
- [ ] Include visualizations for complex concepts
- [ ] Hands-on activity uses realistic dataset
- [ ] Key takeaways summarize main learning objectives
- [ ] Resources section has 3-5 relevant links
- [ ] All code cells execute without errors
- [ ] Narrative flows logically from theory to practice

### Preferred Tools/Libraries by Topic

| Topic | Recommended Libraries |
|-------|----------------------|
| Transfer Learning | PyTorch, TensorFlow, torchvision |
| Reinforcement Learning | Gym, Stable-Baselines3, Ray RLlib |
| Time Series | statsmodels, Prophet, PyTorch |
| AutoML/Hyperparameter | Optuna, Ray Tune, scikit-optimize |
| MLOps/DVC | DVC, MLflow, Airflow |
| Graph Neural Networks | PyGeometric, DGL, PyTorch |
| Federated Learning | Flower, TensorFlow Federated, PySyft |
| Advanced Optimization | JAX, PyTorch, TensorFlow |

---

## Troubleshooting

### Common Issues & Solutions

**Issue: "Lesson file not found when running builder"**
```bash
# Problem: Scripts can't locate the content directory
# Solution: Run scripts from the scaffolding directory
cd /home/user/100daysofml.github.io/scaffolding
python scripts/build_lesson_incrementally.py 37
```

**Issue: "JSON error when saving notebook"**
```bash
# Problem: Malformed JSON in notebook
# Solution: Check for unescaped quotes or special characters
# Re-run builder and avoid pasting raw JSON
python scripts/build_lesson_incrementally.py 37
```

**Issue: "Invalid import statement warning"**
```bash
# Problem: Non-import code submitted to imports section
# Solution: Use Code section (option 4) for setup code
# Import section is strict: only "import x" or "from x import y"
```

**Issue: "Execution timeout (Lesson took >60 seconds to run)"**
```bash
# Problem: Code cells have infinite loops or download large datasets
# Solution: Remove blocking operations
# For large downloads, note in comments but comment out code
# Example:
# # Download dataset (uncomment to run)
# # data = pd.read_csv('http://...')
# data = pd.DataFrame({'col': [1, 2, 3]})  # sample data
```

**Issue: "Running out of token budget mid-lesson"**
```bash
# Problem: Building entire lesson in one session
# Solution: Save progress and continue in new session
# Option 9 in builder: "Save and Exit"
# Scripts preserve progress - you can resume anytime
# Run builder again to continue adding sections
```

### Debug Commands

```bash
# Verify all lesson guides exist
ls -la /home/user/100daysofml.github.io/scaffolding/lesson_guides/

# Check script functionality
python scripts/suggest_topics.py --list | head -20

# Validate notebook JSON
python -m json.tool content/Week_08/Lesson_37.ipynb > /dev/null

# Check git status before committing
git status
git diff content/Week_08/Lesson_37.ipynb
```

### Getting Help

1. **For topic/curriculum questions:** Check `suggest_topics.py --list`
2. **For structure/timing questions:** Read the specific lesson guide
3. **For script issues:** Run with no args to see usage:
   ```bash
   python scripts/suggest_topics.py
   python scripts/generate_lesson_template.py
   python scripts/build_lesson_incrementally.py
   ```
4. **For notebook errors:** Check Jupyter directly:
   ```bash
   jupyter notebook content/Week_08/Lesson_37.ipynb
   ```

---

## Example: Creating Lesson 37 From Start to Finish

### Session 1: Planning (8 minutes)
```bash
cd /home/user/100daysofml.github.io/scaffolding

# Get topic info
python scripts/suggest_topics.py 37
# Output: Lesson 37, Week 8, Fine-tuning pre-trained models, Intermediate

# Read guide
cat lesson_guides/lesson_37_guide.md
# Review recommended structure, timing, prerequisites

# Plan sections based on guide:
# 1. Introduction (10 min) - importance of transfer learning
# 2. Theory (25 min) - fine-tuning concepts, architectures
# 3. Imports (5 min) - PyTorch/TensorFlow libraries
# 4. Visualization (10 min) - compare frozen vs. fine-tuned layers
# 5. Activity (20 min) - fine-tune ResNet on custom dataset
# 6. Takeaways (5 min) - key learnings
# 7. Resources (2 min) - links to papers and tutorials
```

### Session 2: Content Creation (25 minutes)
```bash
# Generate template
python scripts/generate_lesson_template.py 37 "Fine-tuning pre-trained models" 8

# Build incrementally
python scripts/build_lesson_incrementally.py 37

# Menu-driven creation:
# 1. Introduction: "Explain why fine-tuning matters, learning objectives..."
# 2. Theory: "Fine-tuning process, layer freezing strategies, learning rates..."
# 3. Imports: "import torch; from torchvision import models; import matplotlib..."
# 4. Visualization: "Show architecture differences, plot loss curves..."
# 5. Activity: "Load ResNet50, freeze backbone, add custom head, fine-tune..."
# 6. Takeaways: "Fine-tuning is faster and more data-efficient than training from scratch..."
# 7. Resources: "[Transfer Learning Papers](url), [PyTorch Fine-tuning Guide](url)..."

# Save and exit (option 9)
```

### Session 3: Validation (10 minutes)
```bash
# Execute code
python scripts/build_lesson_incrementally.py 37
# Option 8: Execute All Cells
# Verify all cells run without errors

# Check output in Jupyter
jupyter notebook content/Week_08/Lesson_37.ipynb

# Git operations
git add content/Week_08/Lesson_37.ipynb
git commit -m "Add Lesson 37: Fine-tuning pre-trained models

Content covers:
- Transfer learning fundamentals and best practices
- Fine-tuning strategies for different layer types
- Practical implementation with PyTorch
- Hands-on activity with ResNet and custom dataset
- References to state-of-the-art techniques"

git push origin main
```

**Total Time:** ~43 minutes spread across 3 sessions, staying within token budgets.

---

## Additional Resources

### Related Documentation
- Main README: `/100daysofml.github.io/README.md`
- Curriculum Map: `scripts/suggest_topics.py --list`
- Existing Lessons: `/content/Week_*/`
- Git History: `git log --oneline | head -20`

### Python Environment
```bash
# Recommended packages
pip install jupyter notebook ipython pandas numpy matplotlib seaborn scikit-learn

# For PyTorch lessons
pip install torch torchvision torchaudio

# For TensorFlow lessons
pip install tensorflow

# For advanced topics
pip install networkx dvc optuna ray
```

### Contributing New Lesson Scaffolds
If adding new lessons beyond 37, 38, 60, 70, 71, 82, 87, 90, 91, 93, 95, 100:
1. Create new guide in `lesson_guides/lesson_XX_guide.md`
2. Run `scripts/suggest_topics.py <XX>` to verify curriculum mapping
3. Follow same structure as existing guides
4. Test with `generate_lesson_template.py` and `build_lesson_incrementally.py`

---

## Summary

This scaffolding framework solves context overflow by:
- **Breaking lessons into manageable parts** - Use incremental builder for section-by-section creation
- **Providing structured guides** - Read lesson guides before starting any content
- **Mapping curriculum** - Know topics and prerequisites upfront
- **Automating templates** - Generate skeleton notebooks automatically
- **Spreading work across sessions** - Each session stays within token budgets

**Follow the 3-session workflow** for sustainable lesson creation: Planning → Content → Validation.

**Happy lesson building!**
