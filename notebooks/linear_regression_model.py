import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
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
df.drop(columns=['total load actual'], inplace=True)

# --------------------------
# 3. Split features and target
# --------------------------
X = df.drop(columns=['target', 'date'])  # drop date
y = df['target']

# --------------------------
# 4. Time-aware train/test split (80/20)
# --------------------------
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# --------------------------
# 5. Clean and scale numeric features
# --------------------------
def clean_numeric(df):
    df = df.select_dtypes(include=[np.number])
    df = df.dropna()
    return df.astype(np.float64)

X_train = clean_numeric(X_train)
X_test = clean_numeric(X_test)

# Align target
y_train = y_train.loc[X_train.index]
y_test = y_test.loc[X_test.index]

# Fit scaler and transform
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# 6. Train Linear Regression Model
# --------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --------------------------
# 7. Predict & Evaluate
# --------------------------
y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("\n Linear Regression Forecasting Performance:")
print(f"R² Score: {r2:.4f}")
print(f"MAE     : {mae:.2f}")
print(f"RMSE    : {rmse:.2f}")

# --------------------------
# 8. Prediction vs Actual Plot (Sampled)
# --------------------------
sample_fraction = 0.05
sample_indices = np.random.choice(len(y_test), size=int(len(y_test) * sample_fraction), replace=False)
sample_indices = np.sort(sample_indices)

y_test_sampled = y_test.iloc[sample_indices].values
y_pred_sampled = y_pred[sample_indices]

plt.figure(figsize=(12, 5))
plt.plot(y_test_sampled, label='Actual Load', linewidth=2, alpha=0.8)
plt.plot(y_pred_sampled, label='Predicted Load', linewidth=2, alpha=0.8)
plt.title("Linear Regression - Forecasting Next-Hour Electricity Load")
plt.xlabel("Time Index")
plt.ylabel("Load (kW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------
# 9. Residual/Error Analysis
# --------------------------
residuals = y_test - y_pred
plt.figure(figsize=(10, 4))
plt.hist(residuals, bins=50, color='skyblue', edgecolor='black')
plt.title("Linear Regression - Residual Distribution (y_test - y_pred)")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------
# 10. Save Model and Scaler
# --------------------------
with open("linear_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n Model saved as linear_regression_model.pkl")
print("Scaler saved as scaler.pkl")


