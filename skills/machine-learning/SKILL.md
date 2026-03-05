---
name: machine-learning
description: Classical machine learning with scikit-learn. Covers linear/tree/ensemble models, regression, classification, clustering, cross-validation, hyperparameter tuning, and model evaluation.
license: MIT
compatibility: Requires Python 3.11+, scikit-learn, pandas, numpy, scipy
metadata:
  author: IdeaAgent Team
  version: "1.0"
  category: modeling
---

# Machine Learning Skill

Classical supervised and unsupervised learning using **scikit-learn**.

## When to Use

Use this skill when:
- Training linear, tree-based, or ensemble models
- Performing classification or regression on tabular data
- Running cross-validation and hyperparameter search
- Clustering or dimensionality reduction

## File Organisation

Always save outputs to organised subdirectories:
```
models/      ← joblib/pickle files (.pkl)
results/     ← JSON/CSV metrics and evaluation outputs
plots/       ← matplotlib/seaborn figures (.png)
data/        ← processed datasets (.csv)
```

```python
from pathlib import Path
Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)
```

## Common Patterns

### Classification
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
import joblib

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(f"CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")

model.fit(X_train, y_train)
joblib.dump(model, "models/rf_classifier.pkl")
print(classification_report(y_test, model.predict(X_test)))
```

### Regression
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")
joblib.dump(model, "models/ridge_regressor.pkl")
```

### Hyperparameter Search
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    "n_estimators": randint(100, 500),
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": randint(2, 20),
}
search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=30, cv=5, scoring="roc_auc",
    n_jobs=-1, random_state=42, verbose=1
)
search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
print(f"Best score:  {search.best_score_:.4f}")
joblib.dump(search.best_estimator_, "models/best_model.pkl")
```

### Save Results
```python
import json, csv
from datetime import datetime

results = {
    "timestamp": datetime.now().isoformat(),
    "model": "RandomForestClassifier",
    "cv_auc_mean": float(scores.mean()),
    "cv_auc_std": float(scores.std()),
    "best_params": search.best_params_,
}
with open("results/metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved results/metrics.json")
```

## Best Practices
1. Split data **before** any preprocessing (avoid leakage)
2. Fit transformers on **train only**, transform both sets
3. Always set `random_state` for reproducibility
4. Use `cross_val_score` for unbiased performance estimates
5. Save models with `joblib`, not pickle
