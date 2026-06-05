# -*- coding: utf-8 -*-
"""
Created on Thu May  1 15:55:06 2025

@author: User
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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
# 5. Train Random Forest Regressor
# --------------------------
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# --------------------------
# 6. Predict & Evaluate
# --------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("\n Random Forest Forecasting Performance:")
print(f"R² Score: {r2:.4f}")
print(f"MAE     : {mae:.2f}")
print(f"RMSE    : {rmse:.2f}")

# --------------------------
# 7. Plot Actual vs Predicted (Sampled for clarity)
# --------------------------

# Sample ~5% of test set
sample_fraction = 0.05
sample_indices = np.random.choice(len(y_test), size=int(len(y_test) * sample_fraction), replace=False)
sample_indices = np.sort(sample_indices)

# Extract sampled values
y_test_sampled = y_test.iloc[sample_indices].values
y_pred_sampled = y_pred[sample_indices]

# Plot the sampled actual vs predicted
plt.figure(figsize=(12, 5))
plt.plot(y_test_sampled, label='Actual Load', linewidth=2, alpha=0.8)
plt.plot(y_pred_sampled, label='Predicted Load', linewidth=2, alpha=0.8)
plt.title("Random Forest - Forecasting Next-Hour Electricity Load")
plt.xlabel("Time Index")
plt.ylabel("Load (kW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# --------------------------
# 7.5 Residual/Error Plot (for checklist)
# --------------------------
residuals = y_test - y_pred
plt.figure(figsize=(10, 4))
plt.hist(residuals, bins=50, color='skyblue', edgecolor='black')
plt.title("Random Forest - Residual Distribution (y_test - y_pred)")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------
# 8. Save Model for Deployment
# --------------------------
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n Model saved as random_forest_model.pkl")
