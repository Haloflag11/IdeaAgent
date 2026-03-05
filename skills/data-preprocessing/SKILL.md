---
name: data-preprocessing
description: Data preprocessing and cleaning for machine learning experiments. Includes handling missing values, normalization, feature scaling, and data transformation.
license: MIT
compatibility: Requires Python 3.11+, numpy, pandas, scikit-learn
metadata:
  author: IdeaAgent Team
  version: "1.0"
  category: data-processing
---

# Data Preprocessing Skill

This skill provides comprehensive data preprocessing capabilities for machine learning experiments.

## When to Use

Use this skill when:
- You need to clean and prepare raw data for ML models
- Handling missing values in datasets
- Normalizing or standardizing features
- Encoding categorical variables
- Splitting data into train/test sets
- **NO Visualization at this step!**
## Steps

### 1. Load Data
```python
import pandas as pd

# Load from various formats
df = pd.read_csv('data.csv')
# or
df = pd.read_excel('data.xlsx')
# or
df = pd.read_json('data.json')
```

### 2. Handle Missing Values
```python
from sklearn.impute import SimpleImputer
import numpy as np

# Check missing values
print(df.isnull().sum())

# Option 1: Drop rows with missing values
df_clean = df.dropna()

# Option 2: Impute with mean/median/most_frequent
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
df_imputed = imputer.fit_transform(df)

# Option 3: Forward/backward fill (for time series)
df_filled = df.fillna(method='ffill')  # or 'bfill'
```

### 3. Feature Scaling
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler (Z-score normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MinMaxScaler (scale to [0, 1])
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# RobustScaler (robust to outliers)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

### 4. Encode Categorical Variables
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label Encoding (for ordinal or target variables)
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# One-Hot Encoding (for nominal variables)
df_encoded = pd.get_dummies(df, columns=['category'], prefix='cat')

# Or using sklearn
ohe = OneHotEncoder(sparse=False)
X_encoded = ohe.fit_transform(X_categorical)
```

### 5. Feature Engineering
```python
# Create new features
df['feature_ratio'] = df['feature1'] / df['feature2']
df['log_feature'] = np.log1p(df['feature'])

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Feature interaction
df['interaction'] = df['feature1'] * df['feature2']
```

### 6. Split Data
```python
from sklearn.model_selection import train_test_split

# Basic train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Stratified split (for classification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# With validation set
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
```

### 7. Handle Outliers
```python
# IQR method
Q1 = df['feature'].quantile(0.25)
Q3 = df['feature'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_no_outliers = df[(df['feature'] >= lower_bound) & 
                    (df['feature'] <= upper_bound)]

# Z-score method
from scipy import stats
z_scores = np.abs(stats.zscore(df['feature']))
df_no_outliers = df[z_scores < 3]
```

## Common Pipelines

### Complete Preprocessing Pipeline
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define preprocessing for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)
```

## Best Practices

1. **Always split before preprocessing** to avoid data leakage
2. **Fit transformers on training data only**, then transform both train and test
3. **Document all preprocessing steps** for reproducibility
4. **Save preprocessing objects** (scalers, encoders) for inference
5. **Validate preprocessing** by checking distributions before/after
6. **NO Visualization at this step!**

## Validation

After preprocessing, verify:
- No missing values remain
- Features are on appropriate scales
- Categorical variables are properly encoded
- No data leakage occurred
- Train/test distributions are similar

## References

- [Scikit-learn Preprocessing Guide](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
