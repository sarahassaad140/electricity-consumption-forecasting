import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1) Load the dataset
    df = pd.read_csv('df_eda.csv', parse_dates=['date'])
    
    # 2) Verify necessary columns exist
    if 'weather_main' not in df.columns or 'total load actual' not in df.columns:
        raise KeyError("Columns 'weather_main' and 'total load actual' must be in df_eda.csv")
    
    # 3) Drop rows missing these values
    df = df.dropna(subset=['weather_main', 'total load actual'])
    
    # 4) Plot boxplot grouped by weather condition
    plt.figure(figsize=(12, 6))
    df.boxplot(
        column='total load actual',
        by='weather_main',
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', edgecolor='blue'),
        whiskerprops=dict(color='blue'),
        capprops=dict(color='blue'),
        medianprops=dict(color='red', linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='blue', markeredgecolor='blue', markersize=4)
    )
    
    # 5) Refine plot appearance
    plt.title('Total Load Actual by Weather Condition')
    plt.suptitle('')  # Remove default "Boxplot grouped by ..." subtitle
    plt.xlabel('Weather Condition')
    plt.ylabel('Total Load Actual (MW)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # 6) Display the plot
    plt.show()

if __name__ == '__main__':
    main()
