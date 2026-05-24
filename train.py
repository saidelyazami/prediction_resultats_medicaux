import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
import os

# 1. Load Data
df = pd.read_csv('pr4/healthcare_dataset.csv')

# 2. Preprocessing & Feature Engineering
# Calculate length of stay
df['Admission Date'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
df['Stay_Duration'] = (df['Discharge Date'] - df['Admission Date']).dt.days

# Drop unnecessary columns
cols_to_drop = ['Name', 'Doctor', 'Hospital', 'Date of Admission', 'Discharge Date']
df = df.drop(columns=cols_to_drop)

# Target encoding
le = LabelEncoder()
df['Test Results'] = le.fit_transform(df['Test Results'])
target_names = le.classes_

# Define features
X = df.drop('Test Results', axis=1)
y = df['Test Results']

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 3. EDA (Visualizations will be in the notebook, here we just do basic stats)
print("Dataset Shape:", df.shape)
print("Target Distribution:\n", y.value_counts(normalize=True))

# 4. Pipeline Setup
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

# 5. Model Selection (10 algorithms)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    'LightGBM': LGBMClassifier(verbose=-1),
    'CatBoost': CatBoostClassifier(verbose=0),
    'SVM': SVC(probability=True)
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

results = {}
for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# 6. Tuning Top 3 (Simplified for speed in this script, but demonstrating the approach)
# Let's pick Random Forest, XGBoost, and LightGBM as examples
param_grid_rf = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10]
}

best_model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', RandomForestClassifier(random_state=42))])

grid_search = GridSearchCV(best_model_pipeline, param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Save the best model
joblib.dump(grid_search.best_estimator_, 'pr4/model.joblib')
joblib.dump(le, 'pr4/label_encoder.joblib')

print("Model and Label Encoder saved.")
