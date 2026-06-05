import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def main():
    # --------------------------
    # 1) Load and prepare data
    # --------------------------
    df = pd.read_csv("df_fixed.csv")
    df['ds'] = pd.to_datetime(df['date'], errors='coerce')
    df['y'] = df['total load actual']
    df = df[['ds', 'y']].dropna()

    # --------------------------
    # 2) Downsample if dataset is too large
    # --------------------------
    if len(df) > 5000:
        print("Downsampling hourly for performance...")
        df = df.set_index('ds').resample('H').mean().dropna().reset_index()

    # --------------------------
    # 3) Train-test split (80/20)
    # --------------------------
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    # --------------------------
    # 4) Initialize and train Prophet
    # --------------------------
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(train_df)

    # --------------------------
    # 5) Forecast
    # --------------------------
    freq = pd.infer_freq(df['ds'].sort_values()) or 'H'
    future = model.make_future_dataframe(periods=len(test_df), freq=freq)
    forecast = model.predict(future)

    # --------------------------
    # 6) Merge forecast with test set
    # --------------------------
    merged = pd.merge(test_df, forecast[['ds', 'yhat']], on='ds', how='inner')
    y_true = merged['y'].values
    y_pred = merged['yhat'].values

    # --------------------------
    # 7) Evaluation metrics
    # --------------------------
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    print("\n Prophet Forecasting Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE     : {mae:.2f}")
    print(f"RMSE    : {rmse:.2f}")

    # --------------------------
    # 8) Forecast vs Actual Plot
    # --------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label='Actual Load', linewidth=2)
    plt.plot(y_pred, label='Forecasted Load (Prophet)', linewidth=2)
    plt.title("Prophet - Forecasting Next-Hour Electricity Load")
    plt.xlabel("Time Index")
    plt.ylabel("Load (kW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------
    # 9) Residual Plot
    # --------------------------
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 4))
    plt.hist(residuals, bins=50, color='skyblue', edgecolor='black')
    plt.title("Prophet - Residual Distribution (y_true - y_pred)")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------
    #  10) Full Forecast Plot
    # --------------------------
    fig = model.plot(forecast)
    plt.title("Full Forecast with Prophet")
    plt.xlabel("Date")
    plt.ylabel("Predicted Load (kW)")
    plt.tight_layout()
    plt.show()
    
    
    # --------------------------
    # 11) Save model
    # --------------------------
    joblib.dump(model, "prophet_model.pkl")
    print("\n Prophet model saved as prophet_model.pkl")

if __name__ == '__main__':
    main()
