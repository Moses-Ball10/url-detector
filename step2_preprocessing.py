"""
=============================================================================
 STEP 2: DATA PREPROCESSING
=============================================================================
 Project : Multi-Class Malicious URL Detection System
 Dataset : All.csv (ISCX-URL2016 — 79 pre-extracted features)
 Purpose : Clean the data (NaN, inf, sentinel values), encode labels,
           scale features, handle class imbalance, and split into
           train/test sets. Save all preprocessing artifacts for reuse
           in the FastAPI microservice (Step 4).
=============================================================================
"""

# ──────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle                      # To save preprocessing objects (scaler, encoder)
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split   # Split data
from sklearn.preprocessing import LabelEncoder         # Encode target labels
from sklearn.preprocessing import StandardScaler       # Feature normalization
from collections import Counter                        # For class distribution

# ──────────────────────────────────────────────────────────────────────────
# 2. LOAD THE DATASET
# ──────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "All.csv")

print("=" * 70)
print(" STEP 2: DATA PREPROCESSING")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"\n📂 Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Separate feature columns from the target column
TARGET_COL = "URL_Type_obf_Type"
feature_cols = [col for col in df.columns if col != TARGET_COL]

print(f"   ➜ Feature columns: {len(feature_cols)}")
print(f"   ➜ Target column  : {TARGET_COL}")

# ──────────────────────────────────────────────────────────────────────────
# 3. HANDLE INFINITY VALUES
# ──────────────────────────────────────────────────────────────────────────
# From Step 1, we discovered that 'argPathRatio' contains inf values.
# inf values break model training — we must replace them.
# Strategy: Replace inf with NaN first, then handle all NaN together.
print("\n" + "─" * 70)
print(" 3. HANDLING INFINITY VALUES")
print("─" * 70)

# Count inf values before replacement
inf_count = np.isinf(df[feature_cols].select_dtypes(include=[np.number])).sum()
inf_cols = inf_count[inf_count > 0]

if len(inf_cols) > 0:
    print(f"\n⚠️  Found inf values in {len(inf_cols)} column(s):")
    for col in inf_cols.index:
        print(f"   ➜ {col}: {inf_cols[col]:,} inf values")
    
    # Replace all inf and -inf with NaN so they get handled in the next step
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("\n✅ All inf values replaced with NaN (will be handled next).")
else:
    print("\n✅ No inf values found.")

# ──────────────────────────────────────────────────────────────────────────
# 4. HANDLE SENTINEL VALUES (-1)
# ──────────────────────────────────────────────────────────────────────────
# The ISCX-URL2016 dataset uses -1 as a sentinel meaning "not applicable"
# or "feature could not be computed" (e.g., no query string → query features = -1).
# Strategy: Replace -1 sentinels with NaN, then impute all NaN at once.
print("\n" + "─" * 70)
print(" 4. HANDLING SENTINEL VALUES (-1)")
print("─" * 70)

# Identify columns where -1 is used as a sentinel (not a real value)
# These are columns where -1 appears frequently and represents "N/A"
sentinel_cols_to_fix = []
for col in feature_cols:
    if df[col].dtype in ['float64', 'int64']:
        neg_one_count = (df[col] == -1).sum()
        neg_one_pct = neg_one_count / len(df) * 100
        # If -1 appears in more than 5% of rows, it's likely a sentinel
        if neg_one_pct > 5:
            sentinel_cols_to_fix.append(col)
            print(f"   ➜ {col}: {neg_one_count:,} sentinel values ({neg_one_pct:.1f}%)")

print(f"\n   Total columns with sentinel -1: {len(sentinel_cols_to_fix)}")

# Replace sentinel -1 values with NaN in identified columns
for col in sentinel_cols_to_fix:
    df[col] = df[col].replace(-1, np.nan)

print("✅ Sentinel -1 values replaced with NaN.")

# ──────────────────────────────────────────────────────────────────────────
# 5. HANDLE ALL MISSING VALUES (NaN)
# ──────────────────────────────────────────────────────────────────────────
# Now all problematic values (inf, sentinels) have been converted to NaN.
# Strategy: Impute NaN with the MEDIAN of each column.
# Why median? It's robust to outliers (better than mean for skewed data).
print("\n" + "─" * 70)
print(" 5. IMPUTING MISSING VALUES")
print("─" * 70)

total_nan_before = df[feature_cols].isnull().sum().sum()
print(f"\n   Total NaN values before imputation: {total_nan_before:,}")

# Impute all NaN values with column median
# If a column is entirely NaN (e.g., ISIpAddressInDomainName was 100% sentinel),
# its median is also NaN — we fall back to 0 for such columns.
for col in feature_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        if pd.isna(median_val):
            # Column is entirely NaN — fill with 0 (feature carries no info)
            print(f"   ⚠️  '{col}' is entirely NaN — filling with 0 (constant column)")
            df[col].fillna(0, inplace=True)
        else:
            df[col].fillna(median_val, inplace=True)

total_nan_after = df[feature_cols].isnull().sum().sum()
print(f"   Total NaN values after imputation : {total_nan_after}")
print("✅ All missing values imputed with column medians (fallback: 0).")

# ──────────────────────────────────────────────────────────────────────────
# 6. ENCODE THE TARGET LABELS
# ──────────────────────────────────────────────────────────────────────────
# The target column 'URL_Type_obf_Type' contains string labels:
#   Defacement, benign, phishing, malware, spam
# Deep learning models require numerical labels.
# LabelEncoder maps each class to an integer (0, 1, 2, 3, 4).
print("\n" + "─" * 70)
print(" 6. ENCODING TARGET LABELS")
print("─" * 70)

label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df[TARGET_COL])

print(f"\n   Label mapping:")
for i, cls in enumerate(label_encoder.classes_):
    count = (df['label_encoded'] == i).sum()
    print(f"   ➜ {cls:<15} →  {i}  ({count:,} samples)")

# ──────────────────────────────────────────────────────────────────────────
# 7. SPLIT INTO FEATURES (X) AND TARGET (y)
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(" 7. SEPARATING FEATURES & TARGET")
print("─" * 70)

X = df[feature_cols].values.astype(np.float32)  # 79 features as numpy array
y = df['label_encoded'].values                  # Encoded labels as numpy array

print(f"\n   ➜ X shape: {X.shape}  (samples × features)")
print(f"   ➜ y shape: {y.shape}  (samples,)")
print(f"   ➜ X dtype: {X.dtype}")
print(f"   ➜ y dtype: {y.dtype}")

# ──────────────────────────────────────────────────────────────────────────
# 8. TRAIN / TEST SPLIT (80% Train, 20% Test)
# ──────────────────────────────────────────────────────────────────────────
# stratify=y ensures each class is proportionally represented in both sets.
# random_state=42 for reproducibility.
print("\n" + "─" * 70)
print(" 8. TRAIN / TEST SPLIT")
print("─" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,        # 20% for testing
    random_state=42,       # Reproducible results
    stratify=y             # Maintain class proportions in both splits
)

print(f"\n   ➜ Training set : {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   ➜ Testing set  : {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# Verify stratification preserved class proportions
print(f"\n   Class distribution in Training set:")
train_counts = Counter(y_train)
for label in sorted(train_counts):
    cls_name = label_encoder.inverse_transform([label])[0]
    print(f"   ➜ {cls_name:<15}: {train_counts[label]:,} ({train_counts[label]/len(y_train)*100:.1f}%)")

print(f"\n   Class distribution in Testing set:")
test_counts = Counter(y_test)
for label in sorted(test_counts):
    cls_name = label_encoder.inverse_transform([label])[0]
    print(f"   ➜ {cls_name:<15}: {test_counts[label]:,} ({test_counts[label]/len(y_test)*100:.1f}%)")

# ──────────────────────────────────────────────────────────────────────────
# 9. FEATURE SCALING (StandardScaler: zero mean, unit variance)
# ──────────────────────────────────────────────────────────────────────────
# Neural networks converge faster and perform better when features are on
# a similar scale. StandardScaler transforms each feature to have:
#   mean ≈ 0 and std ≈ 1
#
# CRITICAL: We fit the scaler ONLY on the training data to prevent
# data leakage. The test set is then transformed using the same scaler.
print("\n" + "─" * 70)
print(" 9. FEATURE SCALING (StandardScaler)")
print("─" * 70)

scaler = StandardScaler()

# Fit on training data only, then transform both sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n   Before scaling (Training set):")
print(f"   ➜ Mean range : [{X_train.mean(axis=0).min():.4f}, {X_train.mean(axis=0).max():.4f}]")
print(f"   ➜ Std range  : [{X_train.std(axis=0).min():.4f}, {X_train.std(axis=0).max():.4f}]")

print(f"\n   After scaling (Training set):")
print(f"   ➜ Mean range : [{X_train_scaled.mean(axis=0).min():.6f}, {X_train_scaled.mean(axis=0).max():.6f}]")
print(f"   ➜ Std range  : [{X_train_scaled.std(axis=0).min():.6f}, {X_train_scaled.std(axis=0).max():.6f}]")

print("\n✅ Features scaled to zero mean, unit variance.")

# ──────────────────────────────────────────────────────────────────────────
# 10. SAVE PREPROCESSING ARTIFACTS
# ──────────────────────────────────────────────────────────────────────────
# We must save the scaler and label encoder so the FastAPI microservice
# (Step 4) can apply the EXACT SAME transformations to new incoming data.
print("\n" + "─" * 70)
print(" 10. SAVING PREPROCESSING ARTIFACTS")
print("─" * 70)

# Create an 'artifacts' directory to store all saved objects
artifacts_dir = os.path.join(BASE_DIR, "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)

# Save the fitted StandardScaler
scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"\n   ✅ Scaler saved to     : {scaler_path}")

# Save the fitted LabelEncoder
encoder_path = os.path.join(artifacts_dir, "label_encoder.pkl")
with open(encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"   ✅ Encoder saved to    : {encoder_path}")

# Save the feature column names (order matters for the API)
feature_names_path = os.path.join(artifacts_dir, "feature_names.pkl")
with open(feature_names_path, 'wb') as f:
    pickle.dump(feature_cols, f)
print(f"   ✅ Feature names saved : {feature_names_path}")

# Save the preprocessed train/test data as .npz for Step 3
data_path = os.path.join(artifacts_dir, "preprocessed_data.npz")
np.savez(
    data_path,
    X_train=X_train_scaled,
    X_test=X_test_scaled,
    y_train=y_train,
    y_test=y_test
)
print(f"   ✅ Data splits saved   : {data_path}")

# ──────────────────────────────────────────────────────────────────────────
# 11. CORRELATION HEATMAP (Top 20 Features)
# ──────────────────────────────────────────────────────────────────────────
# Visualize feature correlations to understand redundancy.
# We plot only the top 20 features with highest variance to keep it readable.
print("\n" + "─" * 70)
print(" 11. GENERATING CORRELATION HEATMAP")
print("─" * 70)

# Select top 20 features by variance (most informative)
variances = pd.Series(X_train_scaled.var(axis=0), index=feature_cols)
top20_features = variances.nlargest(20).index.tolist()

# Build correlation matrix for top 20 features
top20_df = pd.DataFrame(X_train_scaled, columns=feature_cols)[top20_features]
corr_matrix = top20_df.corr()

# Plot the heatmap
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',           # Red-Blue diverging colormap
    center=0,                # Center colormap at 0
    square=True,
    linewidths=0.5,
    cbar_kws={'shrink': 0.8},
    ax=ax
)
ax.set_title('Feature Correlation Heatmap (Top 20 by Variance)', fontsize=14, fontweight='bold')
plt.tight_layout()

heatmap_path = os.path.join(BASE_DIR, "step2_correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n   ✅ Heatmap saved to: {heatmap_path}")
plt.close()

# ──────────────────────────────────────────────────────────────────────────
# 12. FEATURE DISTRIBUTION BEFORE vs AFTER SCALING (Sample)
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(" 12. GENERATING SCALING COMPARISON CHART")
print("─" * 70)

# Pick 4 representative features to show the effect of scaling
sample_features = ['urlLen', 'Entropy_URL', 'domain_token_count', 'Querylength']
sample_indices = [feature_cols.index(f) for f in sample_features]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle('Feature Distributions: Before vs After Scaling', fontsize=14, fontweight='bold')

for i, (feat_name, feat_idx) in enumerate(zip(sample_features, sample_indices)):
    # Before scaling
    axes[0, i].hist(X_train[:, feat_idx], bins=50, color='#e74c3c', alpha=0.7, edgecolor='white')
    axes[0, i].set_title(f'{feat_name}\n(Before)', fontsize=10)
    axes[0, i].set_ylabel('Frequency' if i == 0 else '')
    
    # After scaling
    axes[1, i].hist(X_train_scaled[:, feat_idx], bins=50, color='#2ecc71', alpha=0.7, edgecolor='white')
    axes[1, i].set_title(f'{feat_name}\n(After)', fontsize=10)
    axes[1, i].set_ylabel('Frequency' if i == 0 else '')

plt.tight_layout()
scaling_chart_path = os.path.join(BASE_DIR, "step2_scaling_comparison.png")
plt.savefig(scaling_chart_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n   ✅ Scaling chart saved to: {scaling_chart_path}")
plt.close()

# ──────────────────────────────────────────────────────────────────────────
# 13. FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(" ✅ STEP 2 COMPLETE — SUMMARY")
print("=" * 70)
print(f"""
   • Infinity values     : Replaced with NaN, then imputed
   • Sentinel values (-1): Replaced in {len(sentinel_cols_to_fix)} columns, then imputed
   • Missing values (NaN): Imputed with column medians (0 remaining)
   • Label encoding       : {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}
   • Feature scaling      : StandardScaler (zero mean, unit variance)
   • Train/Test split     : {X_train_scaled.shape[0]:,} / {X_test_scaled.shape[0]:,} (80/20, stratified)
   
   Saved Artifacts (in ./artifacts/):
   ├── scaler.pkl            — Fitted StandardScaler
   ├── label_encoder.pkl     — Fitted LabelEncoder  
   ├── feature_names.pkl     — Ordered list of 79 feature names
   └── preprocessed_data.npz — X_train, X_test, y_train, y_test

   ➜ Ready to proceed to Step 3: Deep Learning Model
""")
