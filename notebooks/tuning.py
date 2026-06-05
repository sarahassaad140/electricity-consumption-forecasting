# -*- coding: utf-8 -*-
"""
Created on Thu May  1 17:00:42 2025

@author: User
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
import pickle

warnings.filterwarnings("ignore")

# ------------------------------
# Step 1: Load and sample data
# ------------------------------
print("Loading and preparing data...")
df = pd.read_csv("df_fixed.csv")
df['target'] = df['total load actual'].shift(-1)
df.dropna(subset=['target'], inplace=True)
df.drop(columns=['total load actual', 'date'], inplace=True)

#  Use last 5000 rows
df = df.tail(5000)

# ------------------------------
# Step 2: Define X, y, and split
# ------------------------------
X = df.drop(columns=['target'])
y = df['target']

split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Scale only for tree-based models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

tscv = TimeSeriesSplit(n_splits=3)

# ------------------------------
# Step 3: Initialize and Tune Models
# ------------------------------
print("Tuning models...")



#  Decision Tree
grid_dt = RandomizedSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_distributions={
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    n_iter=10, cv=tscv, scoring='r2', n_jobs=-1
)
grid_dt.fit(X_train_scaled, y_train)

#  Random Forest
grid_rf = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions={
        'n_estimators': [100, 150],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    n_iter=10, cv=tscv, scoring='r2', n_jobs=-1
)
grid_rf.fit(X_train_scaled, y_train)

#  XGBoost
grid_xgb = RandomizedSearchCV(
    XGBRegressor(random_state=42, n_jobs=-1),
    param_distributions={
        'n_estimators': [100, 150],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    },
    n_iter=10, cv=tscv, scoring='r2'
)
grid_xgb.fit(X_train_scaled, y_train)

#  LightGBM
grid_lgb = RandomizedSearchCV(
    LGBMRegressor(random_state=42),
    param_distributions={
        'n_estimators': [100, 150],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    },
    n_iter=10, cv=tscv, scoring='r2'
)
grid_lgb.fit(X_train_scaled, y_train)

# ------------------------------
# Step 4: Evaluate all models
# ------------------------------
print(" Evaluating model performance...")

models = {
    
    "Decision Tree": grid_dt.best_estimator_,
    "Random Forest": grid_rf.best_estimator_,
    "XGBoost": grid_xgb.best_estimator_,
    "LightGBM": grid_lgb.best_estimator_,
}

results = []
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)  # scaled
    results.append({
        "Model": name,
        "R²": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred, squared=False)
    })

results_df = pd.DataFrame(results).sort_values(by='R²', ascending=False)

# ------------------------------
# Step 5: Display Clean Table
# ------------------------------
print("\n Model Performance:\n")
print(results_df.style.format({
    "R²": "{:.4f}",
    "MAE": "{:,.2f}",
    "RMSE": "{:,.2f}"
}).to_string())

# ------------------------------
# Step 6: Save Best Model
# ------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(f"\n Best model saved: {best_model_name} → best_model.pkl")
