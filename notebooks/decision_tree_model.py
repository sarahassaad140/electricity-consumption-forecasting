import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle

# --------------------------
# 1. Load the preprocessed dataset
# --------------------------
df = pd.read_csv("df_fixed.csv")

# Create the shifted target (next-hour prediction)
df['target'] = df['total load actual'].shift(-1)
df.dropna(subset=['target'], inplace=True)
df.drop(columns=['total load actual', 'date'], inplace=True)

# --------------------------
# 2. Split features and target
# --------------------------
X = df.drop(columns=['target'])
y = df['target']

# Time-aware 80/20 train-test split
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# --------------------------
# 3. Train Decision Tree Regressor
# --------------------------
model = DecisionTreeRegressor(max_depth=10, random_state=42)
model.fit(X_train, y_train)

# --------------------------
# 4. Predict and Evaluate
# --------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("\nDecision Tree Forecasting Performance:")
print(f"R² Score: {r2:.4f}")
print(f"MAE     : {mae:.2f}")
print(f"RMSE    : {rmse:.2f}")

# --------------------------
# 5. Plot Actual vs Predicted (Sampled)
# --------------------------
sample_indices = np.random.choice(len(y_test), size=int(len(y_test) * 0.05), replace=False)
sample_indices = np.sort(sample_indices)

plt.figure(figsize=(12, 5))
plt.plot(y_test.iloc[sample_indices].values, label='Actual Load', linewidth=2)
plt.plot(y_pred[sample_indices], label='Predicted Load', linewidth=2)
plt.title("Decision Tree - Forecasting Next-Hour Electricity Load")
plt.xlabel("Time Index")
plt.ylabel("Load (kW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------
# 6. Residual Plot
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
# 7. Save the model
# --------------------------
with open("decision_tree_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as decision_tree_model.pkl")

