import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('steva_ecommerce.csv')

print("=== DISTRIBUSI CHURN ===")
print(df['Churned'].value_counts())
print(df['Churned'].value_counts(normalize=True) * 100)

FEATURES = [
    'Age', 'Membership_Years', 'Lifetime_Value', 'Total_Purchases',
    'Days_Since_Last_Purchase', 'Average_Order_Value', 
    'Returns_Rate', 'Cart_Abandonment_Rate'
]

# ==================== 1. HISTOGRAM ====================
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, feature in enumerate(FEATURES):
    df[df['Churned']==0][feature].hist(ax=axes[i], alpha=0.7, color='blue', 
                                bins=30, label='No Churn')
    df[df['Churned']==1][feature].hist(ax=axes[i], alpha=0.7, color='red', 
                                bins=30, label='Churn')
    axes[i].set_title(feature)
    axes[i].legend()

plt.suptitle('Distribusi Fitur berdasarkan Churn Status', fontsize=16)
plt.tight_layout()
plt.savefig('histogram_all_features.png')
plt.show()

# ==================== 2. KORELASI & HEATMAP ====================
corr_matrix = df[FEATURES + ['Churned']].corr()

print("\n" + "="*70)
print("KORELASI DENGAN CHURNED")
print("="*70)
print(corr_matrix['Churned'].sort_values(ascending=False))

# Heatmap
plt.figure(figsize=(11, 9))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.3f', linewidths=0.5, annot_kws={"size": 10})
plt.title('Correlation Heatmap - Fitur vs Churned', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

print("\n✅ Heatmap telah disimpan sebagai: correlation_heatmap.png")