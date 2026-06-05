# -*- coding: utf-8 -*-
"""
Created on Fri May  2 16:19:32 2025

@author: User
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import matplotlib.pyplot as plt
import pickle

# --------------------------
# 1. Load the enhanced dataset
# --------------------------
df = pd.read_csv("df_fixed.csv")

# --------------------------
# 2. Create the shifted target (next-hour forecast)
# --------------------------
df['target'] = df['total load actual'].shift(-1)
df.dropna(subset=['target'], inplace=True)

# Drop current total load to prevent leakage
df.drop(columns=['total load actual'], inplace=True)

# --------------------------
# 3. Split features and target
# --------------------------
X = df.drop(columns=['target', 'date'])  # drop 'date' since it's not numerical
y = df['target']

# --------------------------
# 4. Time-aware train/test split (80/20)
# --------------------------
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# --------------------------
# 5. Time Series Cross-Validation on Train Set
# --------------------------
tscv = TimeSeriesSplit(n_splits=5)
cv_model = XGBRegressor(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

cv_scores = cross_val_score(cv_model, X_train, y_train, cv=tscv, scoring='r2')
print(" Cross-Validated R² Scores:", cv_scores)
print(f" Average R² Score (Train CV): {np.mean(cv_scores):.4f}")

# --------------------------
# 6. Train Final Model on Full Train Set
# --------------------------
model = XGBRegressor(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=1
)
model.fit(X_train, y_train)

# --------------------------
# 7. Predict & Evaluate on Test Set
# --------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("\n XGBoost Test Forecasting Performance:")
print(f"R² Score: {r2:.4f}")
print(f"MAE     : {mae:.2f}")
print(f"RMSE    : {rmse:.2f}")

# --------------------------
# 8. Plot Actual vs Predicted (Sampled for clarity)
# --------------------------
sample_fraction = 0.05
sample_indices = np.random.choice(len(y_test), size=int(len(y_test) * sample_fraction), replace=False)
sample_indices = np.sort(sample_indices)

y_test_sampled = y_test.iloc[sample_indices].values
y_pred_sampled = y_pred[sample_indices]

plt.figure(figsize=(12, 5))
plt.plot(y_test_sampled, label='Actual Load', linewidth=2, alpha=0.8)
plt.plot(y_pred_sampled, label='Predicted Load', linewidth=2, alpha=0.8)
plt.title("XGBoost - Forecasting Next-Hour Electricity Load ")
plt.xlabel("Sampled Time Index")
plt.ylabel("Load (kW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------
# 9. Save Model for Deployment
# --------------------------
with open("xgboost_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n Model saved as xgboost_model.pkl")

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle

# --------------------------
# 1. Load the enhanced dataset
# --------------------------
df = pd.read_csv("df_fixed.csv")

# --------------------------
# 2. Create the shifted target (next-hour forecast)
# --------------------------
df['target'] = df['total load actual'].shift(-1)
df.dropna(subset=['target'], inplace=True)

# Drop current total load to prevent leakage
df.drop(columns=['total load actual'], inplace=True)

# --------------------------
# 3. Split features and target
# --------------------------
X = df.drop(columns=['target', 'date'])  # drop 'date' since it's not numerical
y = df['target']

# --------------------------
# 4. Time-aware train/test split (80/20)
# --------------------------
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# --------------------------
# 5. Time Series Cross-Validation on Train Set
# --------------------------
tscv = TimeSeriesSplit(n_splits=5)
cv_model = LGBMRegressor(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

cv_scores = cross_val_score(cv_model, X_train, y_train, cv=tscv, scoring='r2')
print("Cross-Validated R² Scores:", cv_scores)
print(f" Average R² Score (Train CV): {np.mean(cv_scores):.4f}")

# --------------------------
# 6. Train Final Model on Full Train Set
# --------------------------
model = LGBMRegressor(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train)

# --------------------------
# 7. Predict & Evaluate on Test Set
# --------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("\n LightGBM Test Forecasting Performance:")
print(f"R² Score: {r2:.4f}")
print(f"MAE     : {mae:.2f}")
print(f"RMSE    : {rmse:.2f}")

# --------------------------
# 8. Plot Actual vs Predicted (Sampled for clarity)
# --------------------------
sample_fraction = 0.05
sample_indices = np.random.choice(len(y_test), size=int(len(y_test) * sample_fraction), replace=False)
sample_indices = np.sort(sample_indices)

y_test_sampled = y_test.iloc[sample_indices].values
y_pred_sampled = y_pred[sample_indices]

plt.figure(figsize=(12, 5))
plt.plot(y_test_sampled, label='Actual Load', linewidth=2, alpha=0.8)
plt.plot(y_pred_sampled, label='Predicted Load', linewidth=2, alpha=0.8)
plt.title("LightGBM - Forecasting Next-Hour Electricity Load (Sampled)")
plt.xlabel("Sampled Time Index")
plt.ylabel("Load (kW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------
# 9. Save Model for Deployment
# --------------------------
with open("lightgbm_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n Model saved as lightgbm_model.pkl")

