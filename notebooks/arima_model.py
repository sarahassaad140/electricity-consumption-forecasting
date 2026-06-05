# -*- coding: utf-8 -*-
"""
Created on Fri May  2 20:14:09 2025

@author: User
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

def main():
    # --------------------------
    # 1. Load & prepare data
    # --------------------------
    df = pd.read_csv("df_fixed.csv")
    df['ds'] = pd.to_datetime(df['date'], errors='coerce')
    df['y'] = df['total load actual']
    df = df[['ds', 'y']].dropna()

    # Downsample if needed
    if len(df) > 5000:
        print(" Downsampling hourly for ARIMA...")
        df = df.set_index('ds').resample('H').mean().dropna().reset_index()

    df.set_index('ds', inplace=True)

    # --------------------------
    # 2. Train-test split (80/20)
    # --------------------------
    split_idx = int(len(df) * 0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    # --------------------------
    # 3. Fit ARIMA model
    # --------------------------
    print(" Fitting ARIMA(5,1,2)...")
    model = ARIMA(train['y'], order=(5, 1, 2))
    model_fit = model.fit()

    # --------------------------
    # 4. Forecast
    # --------------------------
    forecast = model_fit.forecast(steps=len(test))
    y_true = test['y'].values
    y_pred = forecast.values

    # --------------------------
    # 5. Evaluation
    # --------------------------
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    print("\n ARIMA Forecasting Performance:")
    print(f" MAE : {mae:.2f}")
    print(f" RMSE: {rmse:.2f}")
    print(f" R²  : {r2:.4f}")

    # --------------------------
    # 6. Plot: Actual vs Predicted
    # --------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:200], label='Actual', linewidth=2)
    plt.plot(y_pred[:200], label='Predicted', linewidth=2)
    plt.title("ARIMA - Actual vs Predicted")
    plt.xlabel("Time Index")
    plt.ylabel("Electricity Load")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------
    # 7. Residual Plot
    # --------------------------
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 4))
    plt.hist(residuals, bins=50, color='skyblue', edgecolor='black')
    plt.title("ARIMA - Residual Distribution (y_true - y_pred)")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------
    # 8. Save model
    # --------------------------
    joblib.dump(model_fit, "arima_model.pkl")
    print("\n ARIMA model saved as arima_model.pkl")

if __name__ == '__main__':
    main()
