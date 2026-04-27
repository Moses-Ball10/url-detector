import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import warnings
import os

warnings.filterwarnings('ignore')

print("=== Phase 2: Data Visualization & Validation ===")

# Get base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "images")

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# 1. Load Data
print("Loading 'Processed_URL_Dataset.csv'...")
df = pd.read_csv(os.path.join(BASE_DIR, "Processed_URL_Dataset.csv"))
X = df.drop(columns=['URL_Type_obf_Type'])
y = df['URL_Type_obf_Type']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 2. Feature Importance
print("Calculating Feature Importances via Random Forest...")
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X, y_encoded)

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(15), palette='viridis')
plt.title('Top 15 Most Important Features for URL Detection', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'Feature_Importance.png'), dpi=300)
plt.close()
print("Saved 'Feature_Importance.png'")

# 3. PCA (Class Separability)
print("Performing PCA for 2D visualization...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 10))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['URL_Type_obf_Type'], palette='Set1', s=15, alpha=0.6)
plt.title('PCA of URL Features (2D Projection)', fontsize=16)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.legend(title='URL Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'PCA_2D_Projection.png'), dpi=300)
plt.close()
print("Saved 'PCA_2D_Projection.png'")

# 4. Boxplots for Top 5 Features
print("Generating Boxplots for Top 5 Features...")
top_5_features = feature_importances.head(5)['Feature'].tolist()
plt.figure(figsize=(15, 10))
for i, feature in enumerate(top_5_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='URL_Type_obf_Type', y=feature, data=df, palette='Set2')
    plt.xticks(rotation=45)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, 'Top5_Boxplots.png'), dpi=300)
plt.close()
print("Saved 'Top5_Boxplots.png'")

print("Phase 2 Visualization Complete! Review the 'images/' folder.")