import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1) Load and parse dates
    df = pd.read_csv('df_eda.csv', parse_dates=['date'])
    
    # 2) Make sure total load column exists
    if 'total load actual' not in df.columns:
        raise KeyError("Column 'total load actual' not found in df_eda.csv")
    
    # --- Average by Day of Week ---
    # 3a) Extract day name
    df['day_of_week'] = df['date'].dt.day_name()
    # 3b) Define order Mon–Sun
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # 3c) Compute and reindex
    avg_dow = df.groupby('day_of_week')['total load actual'].mean().reindex(dow_order)
    
    # 4a) Plot Day‐of‐Week
    plt.figure(figsize=(8, 6))
    plt.bar(avg_dow.index, avg_dow.values, color='lightblue', edgecolor='black')
    plt.title('Average Total Load by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Total Load Actual (MW)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    # --- Average by Month ---
    # 3d) Extract month number
    df['month'] = df['date'].dt.month
    # 3e) Compute and reindex 1–12
    avg_month = df.groupby('month')['total load actual'].mean().reindex(range(1, 13))
    # 3f) Short month labels
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
    # 4b) Plot Month
    plt.figure(figsize=(8, 6))
    plt.bar(month_labels, avg_month.values, color='lightblue', edgecolor='black')
    plt.title('Average Total Load by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Total Load Actual (MW)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
