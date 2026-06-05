import requests 
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler

# Set page layout
st.set_page_config(page_title="Electricity Load Forecast", layout="wide")

# --------------------------- #
# Load Model and Scaler
# --------------------------- #
@st.cache_resource
def load_model_scaler():
    with open("xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_scaler()

# --------------------------- #
# Load Dataset
# --------------------------- #
df = pd.read_csv("df_fixed.csv")
df['target'] = df['total load actual'].shift(-1)
df.dropna(subset=['target'], inplace=True)
df.drop(columns=['total load actual'], inplace=True)

X = df.drop(columns=['target', 'date'])
y = df['target']
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

def clean_numeric(df):
    df = df.select_dtypes(include=[np.number]).dropna()
    return df.astype(np.float64)

X_train = clean_numeric(X_train)
X_test  = clean_numeric(X_test)
y_train = y_train.loc[X_train.index]
y_test  = y_test.loc[X_test.index]

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# --------------------------- #
# Main Dashboard
# --------------------------- #
st.title("Electricity Load Forecasting Dashboard")

# 1. Forecasting Metrics
st.header("Forecasting Performance")
y_pred = model.predict(X_test_scaled)
r2    = round(model.score(X_test_scaled, y_test), 4)
mae   = round(np.mean(np.abs(y_pred - y_test)), 2)
rmse  = round(np.sqrt(np.mean((y_pred - y_test)**2)), 2)
st.markdown(f"- **R² Score:** {r2}")
st.markdown(f"- **MAE:** {mae}")
st.markdown(f"- **RMSE:** {rmse}")

# 2. Forecast vs Actual
st.subheader("Forecast vs Actual (Sampled)")
sample_size    = int(len(y_test) * 0.05)
sample_idx_arr = np.sort(np.random.choice(len(y_test), sample_size, replace=False))
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(y_test.iloc[sample_idx_arr].values, label='Actual')
ax1.plot(y_pred[sample_idx_arr], label='Predicted')
ax1.set_title("Next‑Hour Forecast (Sampled)")
ax1.set_xlabel("Sampled Time Index")
ax1.set_ylabel("Load (kW)")
ax1.legend(); ax1.grid(True)
st.pyplot(fig1)

# 3. Residual Distribution
st.subheader("Residual Distribution")
residuals = y_test.values - y_pred
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.hist(residuals, bins=50, color='skyblue', edgecolor='black')
ax2.set_title("Prediction Errors")
ax2.set_xlabel("Residuals")
ax2.set_ylabel("Frequency")
st.pyplot(fig2)

# 4. Feature Importance
st.subheader("Top 10 Feature Importances")
importances   = model.feature_importances_
feature_names = X_train.columns
top_idx       = np.argsort(importances)[-10:]
fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.barh(range(len(top_idx)), importances[top_idx])
ax3.set_yticks(range(len(top_idx)))
ax3.set_yticklabels([feature_names[i] for i in top_idx])
ax3.set_xlabel("Importance Score")
ax3.set_title("XGBoost Feature Importances")
st.pyplot(fig3)

# 5. SHAP Explainability
st.subheader("SHAP Explainability")
X_test_clean = clean_numeric(X_test).iloc[:1000]
explainer    = shap.Explainer(model, X_train_scaled, feature_names=feature_names)
shap_values  = explainer(scaler.transform(X_test_clean))

# now that shap_values exists, build slider with correct bounds
max_idx = len(shap_values) - 1
st.sidebar.header("Simulation Settings")
sample_idx = st.sidebar.slider(
    "Sample Index for SHAP Waterfall",
    min_value=0,
    max_value=max_idx,
    value=0
)

st.markdown("SHAP Summary Plot:")
fig4 = plt.figure()
shap.summary_plot(shap_values, X_test_clean, max_display=15, show=False)
st.pyplot(fig4)

st.markdown("SHAP Waterfall Plot:")
fig5 = plt.figure()
shap.plots.waterfall(shap_values[sample_idx], max_display=15, show=False)
st.pyplot(fig5)

# 6. Download Predictions
st.subheader("Download Forecast Results")
output_df = pd.DataFrame({
    "Actual Load":    y_test.values,
    "Predicted Load": y_pred
})
csv = output_df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "load_predictions.csv", "text/csv")

# 7. What‑if Simulation
st.subheader("What‑if Scenarios")
st.info("Simulate how temperature, weather or hour changes affect load.")

# 8. Real‑Time Simulation (Real‑World Inputs)
st.header(" Real‑Time Load Simulation")

temperature  = st.slider("Temperature (°C)", min_value=-10, max_value=40, value=22)
humidity     = st.slider("Humidity (%)", min_value=0,  max_value=100, value=50)
wind_speed   = st.slider("Wind Speed (m/s)",  min_value=0,  max_value=30,  value=10)
clouds       = st.slider("Cloud Coverage (%)",min_value=0,  max_value=100, value=30)
hour         = st.slider("Hour of Day",       min_value=0,  max_value=23,  value=14)
day_of_week  = st.selectbox("Day of Week (0=Mon)", list(range(7)), index=2)
is_weekend   = 1 if day_of_week in [5,6] else 0

sim_input = {
  'temp_max':    temperature,
  'humidity':    humidity,
  'wind_speed':  wind_speed,
  'clouds_all':  clouds,
  'hour':        hour,
  'day_of_week': day_of_week,
  'is_weekend':  is_weekend
}
# fill remaining features with zero
for c in feature_names:
    sim_input.setdefault(c, 0)
sim_df    = pd.DataFrame([sim_input])[feature_names]
sim_scaled= scaler.transform(sim_df)
sim_pred  = model.predict(sim_scaled)[0]

st.subheader(f"Predicted Electricity Load: {sim_pred:.2f} kW")

# 9. Automated EMS Control (Point 3)
threshold_high     = y_test.quantile(0.75)
threshold_critical = y_test.quantile(0.95)
st.subheader("Automated Grid Control")
if sim_pred >= threshold_critical:
    st.error("CRITICAL load spike! Sending shed command…")
    resp = requests.post(
      "http://localhost:8000/control",
      json={"action":"shed_load","value": float(sim_pred - threshold_high)}
    )
    st.write("EMS API replied:", resp.json())
elif sim_pred >= threshold_high:
    st.warning(" High load—ready to shed if needed.")
else:
    st.success(" Load normal—no action needed.")

# 10. Sustainability Insights (Point 4)
st.subheader("Sustainability Impact")
solar_now = st.sidebar.slider("Solar Gen (MW)", 0.0,5000.0,1500.0,step=100.0)
wind_now  = st.sidebar.slider("Wind Gen (MW)",  0.0,5000.0,2000.0,step=100.0)
em_factor = 0.5  # kg CO₂ per kWh
co2_est   = sim_pred * em_factor
st.write(f"**Estimated CO₂ next hour:** {co2_est:,.0f} kg")
share     = (solar_now+wind_now)/sim_pred*100 if sim_pred>0 else 0
st.write(f"**Current renewable share:** {share:.1f}% of demand")
    
