---
name: "visualization"
description: "Visualizes experiment results by generating plots and creating markdown reports. Invoke after experiments complete to create visual summaries."
---

# Visualization Skill

This skill generates comprehensive visualizations and reports for machine learning experiment results.

## Workflow Overview

**Trigger**: Invoke after experiment completion when you need to visualize results

**Two-Step Workflow**:
1. **Plot Generation** → Create visualizations and save to workspace
2. **Report Writing** → Generate markdown report with embedded images

## Critical Requirements

### Workspace Management

**ALL outputs MUST be saved to the workspace directory**:

```python
from pathlib import Path

# Step 1: ALWAYS create output directories FIRST
workspace_dir = Path("visualization_output")
images_dir = workspace_dir / "images"

# CRITICAL: Create directories before saving
images_dir.mkdir(parents=True, exist_ok=True)
print(f"✓ Created: {images_dir.absolute()}")
```

### File Saving Rules

1. **Images** → Save to `visualization_output/images/`
2. **Report** → Save to `visualization_output/experiment_report.md`
3. **Use relative paths** (e.g., `images/plot.png` in markdown)
4. **Create directories** before saving files
5. **Print confirmation** after each save

## Step 1: Generate Plots

### Training Curves

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Your training data
train_loss = [2.5, 1.8, 1.2, 0.9, 0.7]
val_loss = [2.6, 2.0, 1.5, 1.1, 0.9]

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curves")
plt.legend()
plt.grid(True, alpha=0.3)

# Save to workspace (CRITICAL: use images_dir from setup)
save_path = images_dir / "training_curves.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {save_path}")
```

### Model Comparison

```python
import numpy as np

model_scores = {
    'Random Forest': 0.92,
    'Gradient Boosting': 0.94,
    'XGBoost': 0.95
}

plt.figure(figsize=(10, 6))
models = list(model_scores.keys())
scores = list(model_scores.values())
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))

plt.bar(models, scores, color=colors, edgecolor='navy')
plt.xlabel("Model")
plt.ylabel("R² Score")
plt.title("Model Performance Comparison")
plt.xticks(rotation=45)
plt.ylim(0, max(scores) * 1.2)

# Add value labels
for i, (model, score) in enumerate(zip(models, scores)):
    plt.text(i, score + 0.01, f'{score:.3f}', ha='center', fontweight='bold')

save_path = images_dir / "model_comparison.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {save_path}")
```

### Confusion Matrix

```python
import numpy as np
import seaborn as sns

confusion_matrix = np.array([
    [45, 2, 1],
    [3, 42, 5],
    [1, 4, 47]
])
class_names = ['Class A', 'Class B', 'Class C']

plt.figure(figsize=(10, 8))
sns.heatmap(
    confusion_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={'label': 'Count'}
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

save_path = images_dir / "confusion_matrix.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {save_path}")
```

### Feature Importance

```python
import numpy as np

features = [f'Feature_{i}' for i in range(1, 16)]
importances = np.random.rand(15) * 0.5 + 0.5
importances = importances / importances.sum()

# Get top N
top_n = 10
indices = np.argsort(importances)[::-1][:top_n]

plt.figure(figsize=(12, 8))
sorted_features = [features[i] for i in indices]
sorted_importances = [importances[i] for i in indices]

colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(sorted_features)))
plt.barh(range(len(sorted_features)), sorted_importances, color=colors)

plt.yticks(range(len(sorted_features)), sorted_features)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.gca().invert_yaxis()

save_path = images_dir / "feature_importance.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {save_path}")
```

## Step 2: Write Markdown Report

```python
from datetime import datetime
from pathlib import Path

report = []

# Title
report.append("# Experiment Report\n")
report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Overview
report.append("\n## 1. Experiment Overview\n")
report.append("**Purpose**: [Describe experiment purpose]\n")
report.append("**Dataset**: [Dataset name and size]\n")
report.append("**Models**: [Models tested]\n")

# Results
report.append("\n## 2. Results\n")
report.append("\n### 2.1 Performance Metrics\n")
report.append("| Metric | Value |\n")
report.append("|--------|-------|\n")
report.append("| Best Model | [Model name] |\n")
report.append("| Best Score | [Score] |\n")

# Visualizations - CRITICAL: Use relative paths
report.append("\n### 2.2 Visualizations\n")
image_files = [
    "images/training_curves.png",
    "images/model_comparison.png",
    "images/confusion_matrix.png",
    "images/feature_importance.png"
]

for img_path in image_files:
    img_name = Path(img_path).stem.replace('_', ' ').title()
    report.append(f"\n#### {img_name}\n")
    report.append(f"![{img_name}]({img_path})\n")

# Analysis
report.append("\n## 3. Analysis\n")
report.append("""
Key findings:

1. **[Finding 1]**: [Description]
2. **[Finding 2]**: [Description]
3. **[Finding 3]**: [Description]
""")

# Conclusion
report.append("\n## 4. Conclusion\n")
report.append("""
**Best Model**: [Model name] ([Score])

**Next Steps**:
1. [Next step 1]
2. [Next step 2]
3. [Next step 3]
""")

# Save report to workspace (CRITICAL: use workspace_dir from setup)
report_path = workspace_dir / "experiment_report.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.writelines(report)

print(f"✓ Saved report to: {report_path}")
print(f"✓ Report location: {report_path.absolute()}")
```

## Expected Output Structure

After execution, your workspace MUST contain:

```
<workspace_root>/
└── visualization_output/
    ├── images/
    │   ├── training_curves.png
    │   ├── model_comparison.png
    │   ├── confusion_matrix.png
    │   └── feature_importance.png
    └── experiment_report.md
```

## Verification Checklist

- [ ] `visualization_output/` directory exists
- [ ] `visualization_output/images/` directory exists
- [ ] All required images saved (>0 bytes)
- [ ] `experiment_report.md` saved with proper formatting
- [ ] Report uses relative image paths (e.g., `images/plot.png`)
- [ ] Printed confirmation for each saved file

## Dependencies

```bash
pip install matplotlib seaborn pandas numpy
```

## Common Mistakes to Avoid

❌ **Don't** use absolute paths in markdown report  
✅ **Do** use relative paths: `images/plot.png`

❌ **Don't** save images without creating directories first  
✅ **Do** call `mkdir(parents=True, exist_ok=True)` before saving

❌ **Don't** call `plt.close()` before `plt.savefig()`  
✅ **Do** save first, then close

❌ **Don't** use wandb, tensorboard, or mlflow unless explicitly requested  
✅ **Do** use matplotlib/seaborn and save to files directly
