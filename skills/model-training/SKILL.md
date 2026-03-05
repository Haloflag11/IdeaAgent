---
name: model-training
description: Machine learning and deep learning model training skill. Supports various algorithms including classification, regression, clustering, and neural networks.
license: MIT
compatibility: Requires Python 3.11+, scikit-learn, PyTorch/TensorFlow (optional)
metadata:
  author: IdeaAgent Team
  version: "1.0"
  category: modeling
---

# Model Training Skill

This skill provides comprehensive model training capabilities for machine learning and deep learning experiments.

## When to Use

Use this skill when:
- Training classification or regression models
- Building and training neural networks
- Performing hyperparameter tuning
- Evaluating model performance
- Saving and loading trained models

## Machine Learning Models

### 1. Classification Models
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Logistic Regression
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# XGBoost
model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# LightGBM
model = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
```

### 2. Regression Models
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Ridge Regression
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# XGBoost Regressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
```

### 3. Clustering Models
```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# K-Means
model = KMeans(n_clusters=5, random_state=42)
model.fit(X)

# DBSCAN
model = DBSCAN(eps=0.5, min_samples=5)
model.fit(X)

# Hierarchical Clustering
model = AgglomerativeClustering(n_clusters=5)
model.fit(X)

# Gaussian Mixture Model
model = GaussianMixture(n_components=5, random_state=42)
model.fit(X)
```

## Deep Learning Models

### 1. PyTorch Neural Network
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

# Initialize model
model = NeuralNetwork(input_size=100, hidden_size=128, output_size=10)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Train the model
train(model, train_loader, criterion, optimizer, epochs=10)
```

### 2. TensorFlow/Keras Model
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Build model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(input_size,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(output_size, activation='softmax')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
)
```

## Hyperparameter Tuning

### 1. Grid Search
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

### 2. Random Search
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [10, 20, 30, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)
}

# Create random search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Fit random search
random_search.fit(X_train, y_train)
```

### 3. Optuna Hyperparameter Optimization
```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    
    model = XGBClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    return model.score(X_val, y_val)

# Create study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Best parameters
print(f"Best parameters: {study.best_params}")
print(f"Best score: {study.best_value:.4f}")
```

## Model Evaluation

### Classification Metrics
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# Detailed report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
```

### Regression Metrics
```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, median_absolute_error
)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
median_ae = median_absolute_error(y_test, y_pred)
```

## Model Saving and Loading

### Save/Load with Joblib
```python
import joblib

# Save model
joblib.dump(model, 'model.pkl')

# Load model
model = joblib.load('model.pkl')

# Save preprocessing pipeline
joblib.dump(preprocessor, 'preprocessor.pkl')
```

### Save/Load PyTorch Model
```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model = NeuralNetwork(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### Save/Load Keras Model
```python
# Save model
model.save('model.h5')

# Load model
model = keras.models.load_model('model.h5')
```

## Best Practices

1. **Use cross-validation** for reliable performance estimates
2. **Set random_state** for reproducibility
3. **Use early stopping** to prevent overfitting
4. **Monitor validation metrics** during training
5. **Save model artifacts** (weights, parameters, preprocessing)
6. **Document experiments** with version control
7. **Use appropriate metrics** for your problem type

## References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)
- [Optuna Documentation](https://optuna.readthedocs.io/)
