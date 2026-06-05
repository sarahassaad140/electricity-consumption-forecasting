import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1) Load the unscaled data
    df = pd.read_csv('df_eda.csv', parse_dates=['date'])
    
    # 2) Compute correlation of numeric features with the target
    target = 'total load actual'
    df_num = df.select_dtypes(include=[np.number])
    corrs = df_num.corr()[target].drop(target)
    
    # 3) Pick the top N features by absolute correlation
    N = 5
    top_feats = corrs.abs().nlargest(N).index.tolist()
    
    # 4) Build sub-matrix (target + those features)
    ordered = [target] + top_feats
    sub_corr = df_num[ordered].corr().loc[ordered, ordered]
    
    # 5) Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolor(
        sub_corr,
        cmap='coolwarm',
        vmin=-1, vmax=1,
        edgecolors='w', linewidths=0.5
    )
    
    # 6) Colorbar
    fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04, label='Pearson r')
    
    # 7) Ticks & labels
    ticks = np.arange(0.5, len(ordered), 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ordered, rotation=90, fontsize=10)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ordered, fontsize=10)
    
    # 8) Make sure x-axis labels are at the bottom
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.tick_bottom()
    
    # (Optional) invert y-axis if you like the target at the top
    ax.invert_yaxis()
    
    # 9) Annotate every cell
    for i in range(len(ordered)):
        for j in range(len(ordered)):
            val = sub_corr.iat[i, j]
            ax.text(
                j + 0.5, i + 0.5,
                f"{val:.2f}",
                ha='center', va='center',
                fontsize=10,
                color='white' if abs(val) > 0.5 else 'black'
            )
    
    # 10) Title & layout
    ax.set_title(f"Top {N} Features Correlated with '{target}'", pad=20)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()




    
