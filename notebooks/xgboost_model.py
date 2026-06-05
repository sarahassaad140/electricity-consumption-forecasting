# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
import shap
from sklearn.preprocessing import StandardScaler

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
X = df.drop(columns=['target', 'date'])  # drop 'date'
y = df['target']

# --------------------------
# 4. Time-aware train/test split (80/20)
# --------------------------
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# --------------------------
# 5. Clean & Scale Data
# --------------------------
def clean_numeric(df):
    df = df.select_dtypes(include=[np.number])
    df = df.dropna()
    return df.astype(np.float64)

X_train = clean_numeric(X_train)
X_test = clean_numeric(X_test)
y_train = y_train.loc[X_train.index]
y_test = y_test.loc[X_test.index]

# Fit StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# 6. Train XGBoost Regressor
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
model.fit(X_train_scaled, y_train)

# --------------------------
# 7. Predict & Evaluate
# --------------------------
y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("\n XGBoost Forecasting Performance:")
print(f"R² Score: {r2:.4f}")
print(f"MAE     : {mae:.2f}")
print(f"RMSE    : {rmse:.2f}")

# --------------------------
# 8. Plot Actual vs Predicted (Sampled)
# --------------------------
sample_fraction = 0.05
sample_indices = np.random.choice(len(y_test), size=int(len(y_test) * sample_fraction), replace=False)
sample_indices = np.sort(sample_indices)

y_test_sampled = y_test.iloc[sample_indices].values
y_pred_sampled = y_pred[sample_indices]

plt.figure(figsize=(12, 5))
plt.plot(y_test_sampled, label='Actual Load', linewidth=2, alpha=0.8)
plt.plot(y_pred_sampled, label='Predicted Load', linewidth=2, alpha=0.8)
plt.title("XGBoost - Forecasting Next-Hour Electricity Load")
plt.xlabel("Time Index")
plt.ylabel("Load (kW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------
# 9. Residual Plot
# --------------------------
residuals = y_test - y_pred
plt.figure(figsize=(10, 4))
plt.hist(residuals, bins=50, color='skyblue', edgecolor='black')
plt.title("Residual Distribution (y_test - y_pred)")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------
# 10. Feature Importance Plot
# --------------------------
importances = model.feature_importances_
feature_names = X_train.columns
top_idx = np.argsort(importances)[-10:]

plt.figure(figsize=(8, 5))
plt.barh(range(len(top_idx)), importances[top_idx], align='center')
plt.yticks(range(len(top_idx)), [feature_names[i] for i in top_idx])
plt.xlabel("Importance Score")
plt.title("Top 10 Feature Importances (XGBoost)")
plt.tight_layout()
plt.show()

# --------------------------
# 11. SHAP Explainability
# --------------------------
print("\n Generating SHAP summary and waterfall plots...")

X_test_clean = X_test.copy()
X_test_clean = X_test_clean.select_dtypes(include=[np.number])
X_test_clean = X_test_clean.dropna().astype(np.float64)
X_test_clean = X_test_clean[X_train.columns.intersection(X_test_clean.columns)]
X_test_clean = X_test_clean.iloc[:1000]

explainer = shap.Explainer(model, X_train_scaled, feature_names=X_train.columns)
shap_values = explainer(scaler.transform(X_test_clean))

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test_clean, max_display=15)

# SHAP Waterfall for first test sample
print("\n SHAP Waterfall plot for first test prediction:")
shap.plots.waterfall(shap_values[0], max_display=15)

# --------------------------
# 12. Save Model and Scaler
# --------------------------
with open("xgboost_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n Model saved as xgboost_model.pkl")
print("Scaler saved as scaler.pkl")


