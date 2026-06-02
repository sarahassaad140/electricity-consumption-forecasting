#  Electricity Consumption Forecasting for Smart Cities

## Overview

This project focuses on forecasting electricity consumption in smart cities using Machine Learning and Time Series Analysis. Accurate energy demand prediction helps utility providers optimize energy distribution, improve grid stability, reduce waste, and support sustainable urban development.

The project uses four years of hourly electricity and weather data to build predictive models capable of forecasting future energy demand.

---

## Objectives

* Forecast electricity consumption accurately.
* Compare traditional, ensemble, and time-series forecasting models.
* Identify the most influential factors affecting energy demand.
* Provide real-time forecasting through an interactive dashboard.
* Support smart-grid decision making and energy optimization.

---

## Dataset

The project combines:

### Energy Dataset

* Electricity demand
* Renewable energy generation
* Non-renewable energy generation
* Market prices
* Storage information

### Weather Dataset

* Temperature
* Humidity
* Wind speed
* Pressure
* Rain and cloud coverage

Data covers approximately four years of hourly observations.

---

## Data Preprocessing

* Data cleaning and merging
* Missing value handling
* Feature selection
* Time-based feature engineering
* One-hot encoding
* Outlier treatment
* Feature scaling
* Cross-validation preparation

---

## Models Implemented

### Baseline Models

* Linear Regression
* Decision Tree Regressor

### Advanced Models

* Random Forest
* XGBoost
* LightGBM

### Time-Series Models

* ARIMA
* Prophet

---

## Evaluation Metrics

Models were evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## Results

### Best Performing Model: XGBoost

| Metric   | Value   |
| -------- | ------- |
| R² Score | 0.9013  |
| MAE      | 1096.59 |
| RMSE     | 1419.33 |

XGBoost outperformed all other models and demonstrated strong predictive performance on unseen data.

---

## Explainable AI

SHAP (SHapley Additive exPlanations) was used to interpret model predictions and identify the most important features affecting electricity demand.

Key influential factors included:

* Hour of day
* Wind generation
* Hydro generation
* Fossil gas generation

---

## Deployment

A Streamlit web application was developed to provide:

* Real-time load forecasting
* Interactive simulations
* SHAP visual explanations
* Grid management recommendations
* Sustainability indicators

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* LightGBM
* ARIMA
* Prophet
* SHAP
* Streamlit
* Matplotlib

---

## Project Structure

```text
├── data/
├── notebooks/
├── models/
├── app/
├── reports/
├── README.md
└── requirements.txt
```

---

## Future Improvements

* Deep Learning models (LSTM, GRU)
* Real-time IoT integration
* Multi-city forecasting
* Cloud deployment
* Automated model retraining

---

## Author

**Sarah Assaad**

Master's Student in Data Science & Artificial Intelligence

Université Saint-Joseph (USJ) & Université Paris-Saclay
