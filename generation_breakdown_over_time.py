import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1) Load & parse dates
    df = pd.read_csv('df_eda.csv', parse_dates=['date'])
    
    # 2) Sort and index
    df = df.sort_values('date').set_index('date')
    
    # 3) Identify generation columns and compute their monthly average
    gen_cols = [
        'generation biomass',
        'generation fossil gas',
        'generation fossil brown coal/lignite',
        'generation fossil hard coal',
        'generation fossil oil',
        'generation hydro pumped storage consumption',
        'generation hydro run-of-river and poundage',
        'generation hydro water reservoir',
        'generation nuclear',
        'generation solar',
        'generation wind'
    ]
    gen_cols = [c for c in gen_cols if c in df.columns]
    monthly_avg = df[gen_cols].resample('M').mean().clip(lower=0)
    
    # 4) Find top‑3 mean contributors over entire period
    long_term_means = monthly_avg.mean().sort_values(ascending=False)
    top3 = long_term_means.index[:3].tolist()
    
    # 5) Build a reduced DataFrame: top 3 + “Other”
    reduced = monthly_avg[top3].copy()
    reduced['Other'] = monthly_avg.drop(columns=top3).sum(axis=1)
    
    # 6) Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    reduced.plot.area(ax=ax, linewidth=0, alpha=0.7)
    
    # 7) Polish
    ax.set_title("Monthly Average Generation: Top 3 + Other", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Generation (MW)")
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
