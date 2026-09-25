import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding='utf-8')
os.makedirs("eda/plots", exist_ok=True)

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# 1. Dataset Scale Comparison
fig, ax = plt.subplots(figsize=(10, 5))
categories = ['Source 1 (Reference)', 'Source 2', 'Source 3', 'Total Records']
train_counts = [2206821, 5034616, 5285603, 12527040]
test_counts  = [1732544, 4887273, 5082316, 11702133]

x = np.arange(len(categories))
width = 0.35

rects1 = ax.bar(x - width/2, [c/1e6 for c in train_counts], width, label='Train Set (12.5M)', color='#2563eb', edgecolor='black', alpha=0.9)
rects2 = ax.bar(x + width/2, [c/1e6 for c in test_counts], width, label='Test Set (11.7M)', color='#10b981', edgecolor='black', alpha=0.9)

ax.set_ylabel('Number of Records (Millions)', fontsize=12, fontweight='bold')
ax.set_title('Amazon ML Challenge 2026: Dataset Scale by Source (Train vs Test)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.legend(frameon=True, facecolor='white', framealpha=1)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.2f}M', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.2f}M', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("eda/plots/01_dataset_scale_comparison.png", dpi=300)
plt.close()
print("Saved 01_dataset_scale_comparison.png")

# 2. Country Distribution (Train vs Test) - Shift to France!
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Train S1 Country
train_countries = {'US': 1323633, 'India': 883188}
colors_train = ['#3b82f6', '#f59e0b']
ax1.pie(train_countries.values(), labels=[f"{k}\n({v/2206821*100:.1f}%)" for k, v in train_countries.items()],
        autopct='%1.1f%%', startangle=140, colors=colors_train, explode=(0.04, 0),
        textprops={'fontsize': 11, 'fontweight': 'bold'})
ax1.set_title("Training Set Country Distribution\n(S1: 2,206,821 records)", fontsize=13, fontweight='bold')

# Test S1 Country
test_countries = {'India': 809986, 'US': 663106, 'France (New!)': 259452}
colors_test = ['#f59e0b', '#3b82f6', '#ec4899']
ax2.pie(test_countries.values(), labels=[f"{k}\n({v/1732544*100:.1f}%)" for k, v in test_countries.items()],
        autopct='%1.1f%%', startangle=140, colors=colors_test, explode=(0.03, 0.03, 0.08),
        textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title("Test Set Country Distribution (Domain Shift)\n(S1: 1,732,544 records)", fontsize=13, fontweight='bold')

plt.suptitle("Country Proportions & Zero-Shot Country Expansion (France)", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("eda/plots/02_country_distribution_shift.png", dpi=300)
plt.close()
print("Saved 02_country_distribution_shift.png")

# 3. Ground Truth Match Count Distribution
match_dist = {
    0: 123247,
    1: 119157,
    2: 375212,
    3: 530841,
    4: 484115,
    5: 321957,
    6: 164868,
    7: 63968,
    8: 18680,
    9: 4205,
    10: 534
}
total_s1 = sum(match_dist.values())
fig, ax = plt.subplots(figsize=(10, 5))
keys = list(match_dist.keys())
pcts = [match_dist[k] / total_s1 * 100 for k in keys]
bars = ax.bar(keys, pcts, color='#6366f1', edgecolor='black', alpha=0.85)
bars[0].set_color('#ef4444')  # Highlight singletons

ax.set_xlabel('Number of Matched Records (from S2 and S3)', fontsize=12, fontweight='bold')
ax.set_ylabel('Percentage of S1 Entities (%)', fontsize=12, fontweight='bold')
ax.set_title('Ground Truth Match Count Distribution per Source 1 Entity\n(Singletons = 5.58%, Modal Matches = 3)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(keys)

for bar, k in zip(bars, keys):
    h = bar.get_height()
    count = match_dist[k]
    ax.annotate(f'{h:.1f}%\n({count:,d})', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_ylim(0, 30)
plt.tight_layout()
plt.savefig("eda/plots/03_ground_truth_match_counts.png", dpi=300)
plt.close()
print("Saved 03_ground_truth_match_counts.png")

# 4. Similarity Distributions on True Matches
if os.path.exists("eda/sim_sample_results.csv"):
    df_sim = pd.read_csv("eda/sim_sample_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Name similarities
    sns.kdeplot(df_sim['name_ratio'], ax=axes[0], label='Levenshtein Ratio', color='#ef4444', lw=2)
    sns.kdeplot(df_sim['name_sort'], ax=axes[0], label='Token Sort Ratio', color='#3b82f6', lw=2)
    sns.kdeplot(df_sim['name_set'], ax=axes[0], label='Token Set Ratio', color='#10b981', lw=2)
    axes[0].set_title('Business Name String Similarity on True Matches', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Similarity Score (0-100)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Density', fontsize=11, fontweight='bold')
    axes[0].legend(frameon=True)
    
    # Address similarities
    sns.kdeplot(df_sim['addr_ratio'], ax=axes[1], label='Levenshtein Ratio', color='#ef4444', lw=2)
    sns.kdeplot(df_sim['addr_sort'], ax=axes[1], label='Token Sort Ratio', color='#3b82f6', lw=2)
    sns.kdeplot(df_sim['addr_set'], ax=axes[1], label='Token Set Ratio', color='#10b981', lw=2)
    axes[1].set_title('Business Address String Similarity on True Matches', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Similarity Score (0-100)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Density', fontsize=11, fontweight='bold')
    axes[1].legend(frameon=True)
    
    plt.suptitle("True Match Pairwise Similarity Distributions (N=60,004 sampled pairs)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("eda/plots/04_similarity_distributions.png", dpi=300)
    plt.close()
    print("Saved 04_similarity_distributions.png")

# 5. F_0.5 Metric Precision vs Recall Sensitivity Curve
precisions = np.linspace(0.1, 1.0, 100)
recalls = [0.5, 0.7, 0.85, 0.95, 1.0]

fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#94a3b8', '#38bdf8', '#3b82f6', '#1d4ed8', '#10b981']

for r, c in zip(recalls, colors):
    f05 = (1.25 * precisions * r) / (0.25 * precisions + r)
    ax.plot(precisions, f05, label=f'Recall = {r:.2f}', color=c, lw=2.5)

ax.set_title(r'Behavior of $F_{0.5}$ Metric: Heavy Precision Weighting ($\beta = 0.5$)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Precision', fontsize=12, fontweight='bold')
ax.set_ylabel(r'$F_{0.5}$ Score', fontsize=12, fontweight='bold')
ax.axvline(0.8, color='grey', linestyle='--', alpha=0.5, label='High Precision Regime (>0.8)')
ax.legend(frameon=True, facecolor='white', framealpha=1, loc='upper left')

plt.tight_layout()
plt.savefig("eda/plots/05_f05_metric_tradeoff.png", dpi=300)
plt.close()
print("Saved 05_f05_metric_tradeoff.png")

print("All visualizations successfully created in eda/plots/")
