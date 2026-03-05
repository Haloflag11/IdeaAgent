---
name: deep-learning
description: Deep learning model training with PyTorch. Covers neural network design, custom training loops, GPU acceleration, convolutional and recurrent networks, and model checkpointing.
license: MIT
compatibility: Requires Python 3.11+, torch, numpy, pandas
metadata:
  author: IdeaAgent Team
  version: "1.1"
  category: modeling
---

# Deep Learning Skill

Neural network training and evaluation using **PyTorch**.

## ⚠️ Critical: CUDA/GPU Detection Before Installing PyTorch

**IMPORTANT**: Before installing PyTorch or any deep learning packages, you MUST first check the system's CUDA version to ensure you install the correct PyTorch build with GPU support.

### Step 1: Check CUDA Version
Run the following command to detect the installed CUDA toolkit version:
```bash
nvcc -V
```

Expected output example:
```
nvcc: NVIDIA (R) Cuda compiler driver
...
Cuda compilation tools, release 11.8, V11.8.89
```

The key information is the **release number** (e.g., `11.8`, `12.1`, `12.4`).

### Step 2: Install Correct PyTorch Version

Based on the CUDA version detected:

| CUDA Version | PyTorch Install Command |
|--------------|------------------------|
| CUDA 11.x    | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` |
| CUDA 12.x    | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` (or cu124 for newer) |
| No CUDA      | `pip install torch torchvision torchaudio` (CPU-only version) |

Even if a CPU version of PyTorch is installed, you should uninstall and reinstall the correct version with GPU support.!!!!!

### Step 3: Verify GPU Support
After installation, verify that PyTorch can see the GPU:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
```

**If CUDA is not detected after installation:**
1. You may have installed the CPU-only version by mistake
2. Uninstall and reinstall with the correct `--index-url` flag
3. Common mistake: `pip install torch` installs CPU-only by default!

## When to Use

Use this skill when:
- Building and training neural networks (MLP, CNN, RNN, Transformer)
- Running custom training loops with gradient-based optimisers
- Leveraging GPU acceleration
- Saving and loading model checkpoints

## File Organisation

Always save outputs to organised subdirectories:
```
models/      ← .pth checkpoint files
results/     ← JSON training history, metrics
results/plots/       ← loss/accuracy curves (.png)
data/        ← processed tensors or cached datasets
logs/        ← per-epoch training logs (.txt or .csv)
```

```python
from pathlib import Path
for d in ["models", "results", "plots", "data", "logs"]:
    Path(d).mkdir(exist_ok=True)
```

## Common Patterns

### Basic MLP
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, out_dim),
        )
    def forward(self, x):
        return self.net(x)

model = MLP(in_dim=128, hidden=256, out_dim=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

### Training Loop
```python
import json
from datetime import datetime

history = {"train_loss": [], "val_loss": [], "val_acc": []}

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch)
            val_loss += criterion(out, y_batch).item()
            correct += (out.argmax(1) == y_batch).sum().item()

    scheduler.step()
    val_acc = correct / len(val_loader.dataset)
    history["train_loss"].append(train_loss / len(train_loader))
    history["val_loss"].append(val_loss / len(val_loader))
    history["val_acc"].append(val_acc)
    print(f"Epoch {epoch:3d} | train_loss={history['train_loss'][-1]:.4f} | val_acc={val_acc:.4f}")

# Save checkpoint and history
torch.save(model.state_dict(), "models/model_final.pth")
with open("results/history.json", "w") as f:
    json.dump(history, f, indent=2)
print("Saved models/model_final.pth and results/history.json")
```

### CNN Example
```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))
```

### Plot Training Curves
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="val")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
axes[1].plot(history["val_acc"], label="val_acc", color="green")
axes[1].set_title("Validation Accuracy"); axes[1].legend(); axes[1].grid(True)
plt.tight_layout()
plt.savefig("plots/training_curves.png", dpi=150)
plt.close()
print("Saved plots/training_curves.png")
```

## Best Practices
1. Always move tensors to `device` before computation
2. Use `model.eval()` and `torch.no_grad()` during validation
3. Save a checkpoint at the best validation metric
4. Use `DataLoader` with `num_workers` for faster data loading
5. Normalise inputs before training
6. Always check if cuda is available, if yes, choose it as the device.
