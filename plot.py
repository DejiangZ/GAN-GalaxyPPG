import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv(
    '/Users/dejiang.zheng/Library/CloudStorage/GoogleDrive-dejiang.jeong@gmail.com/其他计算机/Home/Dataset/ppg/WindowData/analysis_results/activity_summary.csv')

plt.style.use('seaborn-whitegrid')
sns.set_style("whitegrid", {'grid.linestyle': '--'})

fig, ax = plt.subplots(figsize=(15, 8), dpi=300)

x = range(len(df['Activity']))
bars = ax.bar(x, df['Mean_MAE'], yerr=df['Std_MAE'],
              capsize=5, alpha=0.8, color='#2878B5',
              error_kw={'ecolor': '#C4412E', 'capthick': 2})

ax.set_ylabel('Mean Absolute Error (BPM)', fontsize=12, fontweight='bold')
ax.set_xlabel('Activity Type', fontsize=12, fontweight='bold')

plt.xticks(x, df['Activity'], rotation=45, ha='right', fontsize=10)

plt.title('Heart Rate Estimation Performance Across Different Activities',
          fontsize=14, fontweight='bold', pad=20)

for idx, bar in enumerate(bars):
    height = bar.get_height()
    error = df['Std_MAE'].iloc[idx]
    # Calculate the position for the label (above the error bar)
    label_height = height + error + 2  # Add some padding

    # Create the label with smaller font and background
    label = f'{height:.1f}±{error:.1f}'

    # Add white background to text for better visibility
    text = ax.text(bar.get_x() + bar.get_width() / 2., label_height, label,
                   ha='center', va='bottom', fontsize=8,
                   bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ymax = max(df['Mean_MAE'] + df['Std_MAE']) * 1.2  # Add 20% padding
ax.set_ylim(0, ymax)

plt.tight_layout()

plt.savefig('mae_across_activities.png', dpi=300, bbox_inches='tight')
plt.close()