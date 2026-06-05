import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1) Load the unscaled dataset
    df = pd.read_csv('df_eda.csv', parse_dates=['date'])
    
    # 2) Make sure there’s an integer 'hour' column
    if 'hour' not in df.columns:
        df['hour'] = df['date'].dt.hour.astype(int)
    
    # 3) Compute average total load by hour
    avg_load = df.groupby('hour')['total load actual'].mean()
    
    # 4) Prepare x (0–23) and y values (fill any missing hours with NaN)
    hours = list(range(24))
    values = avg_load.reindex(hours).values
    
    # 5) Plot with light-blue bars
    plt.figure(figsize=(10, 6))
    plt.bar(hours, values, color='lightblue', edgecolor='black')
    
    # 6) Set x-ticks to hours 0–23
    plt.xticks(hours, [str(h) for h in hours], rotation=0)
    
    # 7) Labels, title, grid
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Total Load Actual (MW)')
    plt.title('Average Total Load by Hour of Day (Unscaled Data)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 8) Layout and show
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()


