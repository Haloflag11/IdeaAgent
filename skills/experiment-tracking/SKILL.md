---
name: experiment-tracking
description: Experiment tracking and result visualization for ML research. Includes metrics logging, result comparison, and experiment management.
license: MIT
compatibility: Requires Python 3.11+, MLflow or Weights & Biases (optional)
metadata:
  author: IdeaAgent Team
  version: "1.0"
  category: evaluation
---

# Experiment Tracking Skill

This skill provides comprehensive experiment tracking capabilities for machine learning research.

## When to Use

Use this skill when:
- Running multiple experiments with different hyperparameters
- Need to compare model performance across runs
- Tracking training metrics and validation results
- Managing experiment artifacts and checkpoints
- Collaborating with team members on experiments

## MLflow Tracking

### 1. Basic Setup
```python
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Set tracking URI
mlflow.set_tracking_uri("mlruns")

# Set experiment
mlflow.set_experiment("my-experiment")

# Start run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("n_estimators", 100)
    
    # Train model
    model = train_model()
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("f1_score", 0.93)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### 2. Advanced Logging
```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

with mlflow.start_run(run_name="experiment-1"):
    # Log multiple metrics
    for epoch in range(epochs):
        train_loss, val_loss = train_epoch(epoch)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
    
    # Log artifacts
    mlflow.log_text("Model configuration", "config.txt")
    
    # Log confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    # Log directory of artifacts
    mlflow.log_artifacts("results/")
```

### 3. Compare Experiments
```python
import pandas as pd
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get all runs
experiment_id = "1"
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.accuracy DESC"]
)

# Create comparison dataframe
data = []
for run in runs:
    data.append({
        'run_id': run.info.run_id,
        'run_name': run.info.run_name,
        'accuracy': run.data.metrics.get('accuracy'),
        'f1_score': run.data.metrics.get('f1_score'),
        'learning_rate': run.data.params.get('learning_rate'),
        'n_estimators': run.data.params.get('n_estimators'),
    })

df = pd.DataFrame(data)
print(df.sort_values('accuracy', ascending=False))
```

## Weights & Biases (W&B)

### 1. Basic Setup
```python
import wandb

# Initialize
wandb.init(
    project="my-project",
    entity="my-team",
    config={
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 100
    }
)

# Training loop
for epoch in range(epochs):
    train_loss, val_loss = train_epoch()
    
    # Log metrics
    wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epoch": epoch
    })

# Finish run
wandb.finish()
```

### 2. Advanced Features
```python
import wandb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

wandb.init(project="ml-experiments")

# Log configuration
wandb.config.update({
    "model_type": "RandomForest",
    "n_estimators": 100,
    "max_depth": 10
})

# Log table
wandb.log({
    "predictions": wandb.Table(
        columns=["input", "prediction", "actual", "correct"],
        data=[(x, pred, actual, pred == actual) 
              for x, pred, actual in zip(X_test, y_pred, y_test)]
    )
})

# Log confusion matrix
cm = confusion_matrix(y_test, y_pred)
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
    probs=None,
    y_true=y_test,
    preds=y_pred,
    class_labels=["Class 0", "Class 1"]
)})

# Log model
wandb.save("model.pkl")

# Finish
wandb.finish()
```

### 3. Hyperparameter Sweep
```python
import wandb

# Define sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-1
        },
        'n_estimators': {
            'values': [100, 200, 300, 500]
        },
        'max_depth': {
            'min': 5,
            'max': 50
        }
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="my-project")

# Training function
def train():
    run = wandb.init()
    config = wandb.config
    
    # Use config for training
    model = train_model(
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        max_depth=config.max_depth
    )
    
    # Log results
    accuracy = evaluate_model(model)
    wandb.log({"accuracy": accuracy})

# Run sweep
wandb.agent(sweep_id, train, count=20)
```

## Custom Experiment Tracker

### Simple CSV-based Tracker
```python
import csv
import json
from datetime import datetime
from pathlib import Path

class ExperimentTracker:
    def __init__(self, experiment_name, base_dir="experiments"):
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.experiment_dir / "metrics.csv"
        self.config_file = self.experiment_dir / "config.json"
        
        self._init_metrics_file()
    
    def _init_metrics_file(self):
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'epoch', 'metric_name', 'value'])
    
    def log_config(self, config: dict):
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_metric(self, name: str, value: float, epoch: int = None):
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                epoch,
                name,
                value
            ])
    
    def log_metrics(self, metrics: dict, epoch: int = None):
        for name, value in metrics.items():
            self.log_metric(name, value, epoch)
    
    def save_artifact(self, artifact, name: str):
        artifact_path = self.experiment_dir / name
        if hasattr(artifact, 'save'):
            artifact.save(artifact_path)
        else:
            import joblib
            joblib.dump(artifact, artifact_path)
    
    def get_metrics(self) -> dict:
        import pandas as pd
        df = pd.read_csv(self.metrics_file)
        return {
            'config': json.load(open(self.config_file)),
            'metrics': df
        }
```

## Visualization

### Compare Multiple Runs
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_runs(trackers: list, metric_name: str = 'accuracy'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for tracker in trackers:
        data = tracker.get_metrics()
        metrics_df = data['metrics']
        config = data['config']
        
        # Filter by metric
        metric_data = metrics_df[metrics_df['metric_name'] == metric_name]
        
        # Plot
        axes[0, 0].plot(
            metric_data['epoch'],
            metric_data['value'],
            label=f"{config.get('model_type', 'Unknown')}"
        )
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel(metric_name)
    axes[0, 0].legend()
    axes[0, 0].set_title(f'{metric_name} Comparison')
    
    # Heatmap of final metrics
    final_metrics = []
    for tracker in trackers:
        data = tracker.get_metrics()
        final_value = data['metrics'].iloc[-1]['value']
        final_metrics.append(final_value)
    
    sns.heatmap(
        [final_metrics],
        annot=True,
        fmt='.4f',
        ax=axes[0, 1],
        cmap='YlGnBu'
    )
    
    plt.tight_layout()
    plt.savefig('experiment_comparison.png')
    plt.show()
```

## Best Practices

1. **Log everything**: parameters, metrics, artifacts, environment
2. **Use meaningful run names**: Include key hyperparameters or experiment type
3. **Tag related runs**: Use tags to group experiments by category
4. **Version control code**: Always track which code version produced results
5. **Save model checkpoints**: Especially for long training runs
6. **Document experiments**: Add notes about observations and insights
7. **Clean up old runs**: Archive or delete unsuccessful experiments

## References

- [MLflow Documentation](https://mlflow.org/docs/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)
