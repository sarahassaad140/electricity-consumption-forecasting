import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1) Load the dataset
    df = pd.read_csv('df_eda.csv', parse_dates=['date'])
    
    # 2) Ensure the 'weather_main' column exists
    if 'weather_main' not in df.columns:
        raise KeyError("Column 'weather_main' not found in df_eda.csv")
    
    # 3) Compute the top 3 most frequent weather conditions
    top3 = df['weather_main'].value_counts().nlargest(3)
    
    # 4) Plot as a bar chart
    plt.figure(figsize=(6, 4))
    top3.plot(kind='bar', color='lightblue', edgecolor='black')
    
    # 5) Labels and title
    plt.title('Top 3 Weather Condition Frequencies')
    plt.xlabel('Weather Condition')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 6) Layout and show
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
