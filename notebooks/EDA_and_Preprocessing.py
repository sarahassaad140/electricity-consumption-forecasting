import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# -----------------------------
# Step 1: Load raw data
# -----------------------------
dfe = pd.read_csv("energy_dataset.csv", parse_dates=['time'])
dfw = pd.read_csv("weather_features.csv", parse_dates=['dt_iso'])

# -----------------------------
# Step 2: Convert datetime columns to timezone-aware
# -----------------------------
dfe['time'] = pd.to_datetime(dfe['time'], errors='coerce', utc=True)
dfw['dt_iso'] = pd.to_datetime(dfw['dt_iso'], errors='coerce', utc=True)

# -----------------------------
# Step 3: Merge datasets on datetime
# -----------------------------
merged_df = pd.merge(dfe, dfw, left_on='time', right_on='dt_iso', how='inner')
merged_df = merged_df.drop_duplicates(subset=['time'])

# -----------------------------
# Step 4: Select and rename relevant columns
# -----------------------------
columns_to_keep = [
    'time', 'generation biomass', 'generation fossil brown coal/lignite', 'generation fossil gas',
    'generation fossil hard coal', 'generation fossil oil', 'generation hydro pumped storage consumption',
    'generation hydro run-of-river and poundage', 'generation hydro water reservoir', 'generation nuclear',
    'generation other', 'generation other renewable', 'generation solar', 'generation waste',
    'generation wind onshore', 'total load actual', 'price actual', 'temp_min', 'temp_max',
    'pressure', 'humidity', 'wind_speed', 'clouds_all', 'weather_main'
]

merged_new = merged_df[columns_to_keep].copy()
merged_new.rename(columns={"generation wind onshore": "generation wind"}, inplace=True)

# -----------------------------
# Step 5: Handle missing values
# -----------------------------
merged_new.dropna(inplace=True)

# -----------------------------
# Step 6: Extract time-based features
# -----------------------------
merged_new['date'] = merged_df['time'].dt.date
merged_new['hour'] = merged_df['time'].dt.hour
merged_new['month'] = merged_df['time'].dt.month
merged_new['week'] = merged_df['time'].dt.isocalendar().week
merged_new['day'] = merged_df['time'].dt.day
merged_new['year'] = merged_df['time'].dt.year
merged_new['day_of_week'] = merged_df['time'].dt.dayofweek
merged_new['is_weekend'] = merged_new['day_of_week'].isin([5, 6]).astype(int)

# -----------------------------
# Step 7: Add new categorical time-based feature (hour_category)
# ----------------------------

def categorize_hour(h):
    if h < 6:
        return 'night'
    elif h < 12:
        return 'morning'
    elif h < 18:
        return 'afternoon'
    else:
        return 'evening'

merged_new['hour_category'] = merged_new['hour'].apply(categorize_hour)
merged_new = pd.get_dummies(merged_new, columns=['hour_category'], prefix='hour_cat', drop_first=True)

# -----------------------------
# Step 8: Encode categorical weather
# -----------------------------
merged_new = pd.get_dummies(merged_new, columns=['weather_main'], drop_first=True)

# -----------------------------
# Step 9: Drop time column after extraction
# -----------------------------
merged_new.drop(columns=['time'], inplace=True)

# -----------------------------
# Step 10: Reorder columns
# -----------------------------
meta_cols = ['date', 'year', 'month', 'week', 'day', 'hour', 'day_of_week', 'is_weekend']
merged_new = merged_new[meta_cols + [col for col in merged_new.columns if col not in meta_cols]]

# -----------------------------
# Step 11: Outlier capping (IQR)
# -----------------------------
numeric_cols = merged_new.select_dtypes(include='number').columns
for col in numeric_cols:
    Q1 = merged_new[col].quantile(0.25)
    Q3 = merged_new[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    merged_new[col] = np.clip(merged_new[col], lower, upper)

# -----------------------------
# Step 12: Scale numeric features (excluding time and categorical flags)
# -----------------------------

do_not_scale = [
    # 1) The target
    'total load actual',

    # 2) Hour‐category dummies
    'hour_cat_evening',
    'hour_cat_morning',
    'hour_cat_night',   # (afternoon is the “dropped” category)

    # 3) Weather‐condition dummies
    'weather_main_clouds',
    'weather_main_drizzle',
    'weather_main_fog',
    'weather_main_haze',
    'weather_main_mist',
    'weather_main_rain',
    'weather_main_smoke',
    'weather_main_thunderstorm',

    # 4) Time metadata 
    'date',
    'year',
    'month',
    'week',
    'day',
    'hour',
    'day_of_week',
    'is_weekend'
]
features_to_scale = [
    col for col in merged_new.select_dtypes(include='number').columns
    if col not in do_not_scale
]

scaler = StandardScaler()
merged_new[features_to_scale] = scaler.fit_transform(merged_new[features_to_scale])

# -----------------------------
# Step 13: Save final output
# -----------------------------
merged_new.to_csv("df_fixed.csv", index=False)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(" Preprocessing complete. Cleaned dataset and scaler saved.")
