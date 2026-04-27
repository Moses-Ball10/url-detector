import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import warnings
import json
import os

warnings.filterwarnings('ignore')

print("=== Phase 1: Data Preparation ===")

# 1. Load Data
print("Loading 'All.csv'...")
# Get base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, "All.csv"), sep=None, engine='python')
df.columns = df.columns.str.strip()

# 2. Data Cleaning: Handle NaNs
print("Cleaning data...")
# Many URL features are NaN because the URL lacks that component (e.g., no path = no path entropy).
# Filling with 0 is logically sound for these specific features.
df = df.fillna(0)

# Replace any Infinity values with 0
df.replace([np.inf, -np.inf], 0, inplace=True)

# Separate features (X) and target (y)
X = df.drop(columns=['URL_Type_obf_Type'])
y = df['URL_Type_obf_Type']

# Convert all features to numeric, just in case
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# 3. Feature Selection: Drop useless columns
# 3. Feature Selection: Drop useless columns
print("Applying feature selection...")

# a) Constant columns (no variance)
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]

# b) Known Dead Weight (From previous visual analysis)
dead_weight_cols = ['isPortEighty', 'dld_domain']

# c) Highly Correlated Features (>0.90 correlation)
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
highly_correlated_cols = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]

total_drop = list(set(constant_cols + dead_weight_cols + highly_correlated_cols))
total_drop = [col for col in total_drop if col in X.columns]

X_cleaned = X.drop(columns=total_drop)
print(f"Dropped {len(total_drop)} columns. Features remaining: {X_cleaned.shape[1]}")
print(f"Dropped {len(total_drop)} columns (Constants, Dead Weight, or Highly Correlated).")
print(f"Features remaining: {X_cleaned.shape[1]}")

# 4. SMOTE Balancing
print("Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_cleaned, y)
print(f"Balanced Class Distribution:\n{y_resampled.value_counts()}")

# 5. Export Processed Data & Feature Schema
print("Exporting datasets...")
df_resampled = pd.concat([
    pd.DataFrame(X_resampled, columns=X_cleaned.columns), 
    pd.Series(y_resampled, name='URL_Type_obf_Type')
], axis=1)

df_resampled.to_csv(os.path.join(BASE_DIR, "Processed_URL_Dataset.csv"), index=False)
print("Data saved as 'Processed_URL_Dataset.csv'")

expected_features = list(X_cleaned.columns)
with open(os.path.join(BASE_DIR, "models", "expected_features.json"), "w") as f:
    json.dump(expected_features, f)
print(f"Saved {len(expected_features)} expected feature names to 'expected_features.json' for the web app.")