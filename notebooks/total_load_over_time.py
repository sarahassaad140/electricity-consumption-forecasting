import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1) Load the dataset with date parsing
    df = pd.read_csv('df_eda.csv', parse_dates=['date'])

    # 2) Set datetime as index for resampling
    df.set_index('date', inplace=True)

    # 3) Resample to daily average
    daily_avg = df['total load actual'].resample('D').mean()

    # 4) Plot daily average total load (blue)
    plt.figure(figsize=(14, 6))
    plt.plot(daily_avg.index, daily_avg.values, color='lightblue', linewidth=1.5)

    # 5) Add labels and title (no emoji)
    plt.title('Daily Average Total Load Over Time', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Daily Avg Total Load (MW)')
    plt.grid(True)

    # 6) Improve layout and show
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
