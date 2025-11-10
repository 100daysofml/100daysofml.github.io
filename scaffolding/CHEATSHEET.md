# 100 Days of ML - Quick Reference Cheat Sheet

## 1. Quick Commands (One-Liners)

```bash
# Get topic for any lesson (36-100)
python scripts/suggest_topics.py <lesson_num>

# Generate notebook template
python scripts/generate_lesson_template.py <lesson_num> "<topic>" <week_num>

# Interactive content builder (section-by-section)
python scripts/build_lesson_incrementally.py <lesson_num>

# List all 65 lessons with topics
python scripts/suggest_topics.py --list

# Quick start wizard (interactive, all-in-one)
bash scripts/quick_start.sh <lesson_num>
```

---

## 2. Lesson Guide Index (12 Advanced Lessons)

| # | Lesson | Week | Theme | Topic | Difficulty |
|---|--------|------|-------|-------|------------|
| 1 | 37 | 8 | Transfer Learning | Fine-tuning pre-trained models | Intermediate |
| 2 | 38 | 8 | Transfer Learning | Feature extraction | Intermediate |
| 3 | 60 | 12 | Reinforcement Learning | Multi-agent RL | Advanced |
| 4 | 70 | 14 | Time Series | Attention-based sequences | Advanced |
| 5 | 71 | 15 | AutoML | Hyperparameter optimization | Advanced |
| 6 | 82 | 17 | MLOps | Data versioning (DVC) | Expert |
| 7 | 87 | 18 | Graph Neural Networks | Message passing | Expert |
| 8 | 90 | 18 | Graph Neural Networks | Knowledge graphs | Expert |
| 9 | 91 | 19 | Federated Learning | FL fundamentals | Expert |
| 10 | 93 | 19 | Federated Learning | Privacy-preserving ML | Expert |
| 11 | 95 | 19 | Distributed Learning | Gradient synchronization | Expert |
| 12 | 100 | 20 | Advanced Optimization | Neural ODE | Expert |

---

## 3. Common Workflow (3 Steps)

**Step 1: Plan & Research**
```bash
cd /home/user/100daysofml.github.io/scaffolding
python scripts/suggest_topics.py <lesson_num>     # Get topic/difficulty
cat lesson_guides/lesson_<num>_guide.md           # Read full guide with structure
```

**Step 2: Create & Build**
```bash
# Option A: Start from scratch with template
python scripts/generate_lesson_template.py <lesson> "<topic>" <week>

# Option B: Interactive builder (recommended for complex lessons)
python scripts/build_lesson_incrementally.py <lesson>
# Menu: Add Introduction → Theory → Imports → Visualization → Activity → Takeaways → Resources

# Option C: One-command setup
bash scripts/quick_start.sh <lesson>
```

**Step 3: Validate & Commit**
```bash
# Test all code cells
python scripts/build_lesson_incrementally.py <lesson>  # Select option 8: Execute All Cells

# Commit with detailed message
git add content/Week_<WW>/Lesson_<LL>.ipynb
git commit -m "Add Lesson <LL>: <Topic>

- Brief description of content
- Key concepts covered
- Technologies used"

git push origin main
```

---

## 4. Jupyter Notebook Cell Types

| Cell Type | Usage | Example |
|-----------|-------|---------|
| **Markdown** | Titles, explanations, math | `# Topic`, `## Subtopic`, `Learning objectives:` |
| **Code: Imports** | Libraries at top | `import numpy as np`, `from sklearn import datasets` |
| **Code: Setup** | Data/config | `np.random.seed(42)`, `df = pd.read_csv('data.csv')` |
| **Code: Main Logic** | Core algorithm | Model definition, training loops, inference |
| **Code: Visualization** | Plots & results | `plt.plot()`, `sns.heatmap()`, display outputs |
| **Code: Testing** | Validation | Assert statements, unit tests, error checking |

**Pro Tips:**
- Keep imports at top, one logical group per cell
- One concept per code cell (easier to debug)
- Use markdown cells liberally for explanation
- Include expected output comments: `# Output: [shape, value, graph]`

---

## 5. Key LaTeX Formulas (Common ML)

```latex
% Gradient Descent
\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t)

% Linear Regression (Normal Equation)
\theta = (X^T X)^{-1} X^T y

% Cross-Entropy Loss
L = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]

% Neural Network Forward Pass
h^{(l)} = \sigma(W^{(l)} h^{(l-1)} + b^{(l)})

% Softmax (Multi-class)
P(y=k|x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}

% Attention Mechanism
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

% KL Divergence
D_{KL}(P||Q) = \sum_{x} P(x) \log\frac{P(x)}{Q(x)}

% Accuracy/Precision/Recall/F1
\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}
\text{Precision} = \frac{TP}{TP+FP}, \quad \text{Recall} = \frac{TP}{TP+FN}
\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
```

---

## 6. Code Snippets (Copy-Paste Ready)

### Essential Imports
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

### Basic Data Loading & Exploration
```python
# Load data
df = pd.read_csv('data.csv')
print(df.shape, df.head(), df.info(), df.describe())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### PyTorch Model Template
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Training loop
model = SimpleNet(784, 128, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
```

### TensorFlow/Keras Template
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(X_train, y_train, epochs=10, batch_size=32,
                   validation_split=0.2)
```

### Standard Visualization
```python
# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.legend(), plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch'), plt.ylabel('Accuracy'), plt.legend(), plt.grid()
plt.tight_layout()
plt.show()
```

---

## 7. Validation Checklist (Before Committing)

- [ ] **Notebook Runs:** All code cells execute without errors
- [ ] **Imports:** All required packages listed at top
- [ ] **Data Available:** Dataset loading works (no broken paths)
- [ ] **Output Clear:** Each code cell has expected output/comments
- [ ] **Structure Complete:** Introduction → Theory → Code → Activity → Takeaways → Resources
- [ ] **Visualizations:** Plots render correctly with labels/legends
- [ ] **Markdown:** Explanations are clear and concise
- [ ] **LaTeX:** Math formulas render in Markdown cells
- [ ] **No Secrets:** No API keys, passwords, or sensitive data
- [ ] **Cell Count Reasonable:** <100 cells total (performance)
- [ ] **Execution Time:** <60 seconds to run all cells
- [ ] **File Path Correct:** `content/Week_<WW>/Lesson_<LL>.ipynb`

---

## 8. Git Commands (Common Workflows)

### Create & Commit Lesson
```bash
# 1. Checkout/create branch (optional but recommended)
git checkout -b lesson/<lesson_num>/<topic>

# 2. Stage the lesson
git add content/Week_<WW>/Lesson_<LL>.ipynb

# 3. Check what you're committing
git diff --staged content/Week_<WW>/Lesson_<LL>.ipynb | head -50

# 4. Commit with message
git commit -m "Add Lesson <LL>: <Topic>

Comprehensive description of:
- Core concepts covered
- Key takeaways
- Technologies/libraries used
- Hands-on activity included"

# 5. Push to remote
git push -u origin lesson/<lesson_num>/<topic>
```

### Create Pull Request
```bash
# Push branch first (see above)
git push -u origin lesson/<lesson_num>/<topic>

# Then create PR
gh pr create --title "Lesson <LL>: <Topic>" \
             --body "Adds comprehensive lesson on <topic>. See CHEATSHEET.md for validation."
```

### Merge & Clean Up
```bash
# Switch to main
git checkout main
git pull origin main

# Merge your branch (after PR review)
git merge lesson/<lesson_num>/<topic>
git push origin main

# Delete feature branch
git branch -d lesson/<lesson_num>/<topic>
git push origin --delete lesson/<lesson_num>/<topic>
```

### Useful Status Commands
```bash
git status                          # See what changed
git log --oneline -10               # View recent commits
git diff content/Week_08/           # See changes in folder
git show <commit-hash>              # View specific commit
```

---

## Directory Structure Quick Reference

```
/home/user/100daysofml.github.io/
├── content/
│   ├── Week_01/, Week_02/, ..., Week_20/
│   │   └── Lesson_<LL>.ipynb              # Your notebook files
├── scaffolding/
│   ├── CHEATSHEET.md                      # This file
│   ├── README.md                          # Full documentation
│   ├── lesson_guides/                     # Pre-written guides
│   │   ├── lesson_37_guide.md
│   │   ├── lesson_60_guide.md, ...
│   │   └── lesson_100_guide.md
│   ├── scripts/                           # Automation tools
│   │   ├── suggest_topics.py
│   │   ├── generate_lesson_template.py
│   │   └── build_lesson_incrementally.py
│   └── templates/                         # Notebook templates
└── .git/                                   # Git repository
```

---

## Pro Tips

1. **Token Budget:** Each lesson creation session ~50-60k tokens. Spread across 3 sessions: Plan → Build → Validate
2. **Avoid Timeouts:** Run `python scripts/build_lesson_incrementally.py` to test code before committing
3. **Use Guides:** Always read `/scaffolding/lesson_guides/lesson_<num>_guide.md` for structure
4. **Copy Snippets:** Keep commonly used imports/templates in a separate file
5. **Branch Naming:** Use `lesson/<num>/<slug>` format for clarity
6. **Commit Messages:** Include what was learned + technologies used
7. **Execute First:** Always option 8 (Execute All) before committing
8. **Document Early:** Add markdown explanations while coding, not after

---

**Last Updated:** Nov 10, 2025 | Full docs: `/scaffolding/README.md` | For help: `python scripts/<script>.py` (no args)
