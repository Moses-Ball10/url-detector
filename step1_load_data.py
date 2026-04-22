"""
=============================================================================
 STEP 1: DATA LOADING & INITIAL EXPLORATION
=============================================================================
 Project : Multi-Class Malicious URL Detection System
 Dataset : All.csv (ISCX-URL2016 — 79 pre-extracted features)
 Classes : Benign, Phishing, Malware, Spam, Defacement
 Purpose : Load the local dataset, inspect its structure, check for
           missing values, and visualize the target class distribution.
=============================================================================
"""

# ──────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────────────────
import pandas as pd               # Data manipulation and CSV loading
import numpy as np                # Numerical operations
import matplotlib.pyplot as plt   # Static plotting
import seaborn as sns             # Statistical visualization (built on matplotlib)
import os                         # File path operations
import warnings
warnings.filterwarnings('ignore') # Suppress non-critical warnings for cleaner output

# ──────────────────────────────────────────────────────────────────────────
# 2. LOAD THE DATASET
# ──────────────────────────────────────────────────────────────────────────
# Build the path to All.csv relative to this script's location
# This ensures it works regardless of where you run the script from.
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "All.csv")

print("=" * 70)
print(" STEP 1: DATA LOADING & INITIAL EXPLORATION")
print("=" * 70)
print(f"\n📂 Loading dataset from: {DATA_PATH}")

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(DATA_PATH)

print(f"✅ Dataset loaded successfully!")
print(f"   ➜ Shape: {df.shape[0]:,} rows  ×  {df.shape[1]} columns")

# ──────────────────────────────────────────────────────────────────────────
# 3. BASIC DATASET INFORMATION
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(" 3. COLUMN OVERVIEW")
print("─" * 70)

# Show the first 5 rows for a quick visual sanity check
print("\n🔍 First 5 rows (transposed for readability):")
print(df.head().T.to_string())  # Transpose so 79 columns display vertically

# Print column names and data types
print(f"\n📋 Total columns: {df.shape[1]}")
print(f"   ➜ Feature columns: {df.shape[1] - 1}")
print(f"   ➜ Target column  : 'URL_Type_obf_Type'")

# Data types summary
print(f"\n📊 Data types:")
dtype_counts = df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"   ➜ {dtype}: {count} columns")

# ──────────────────────────────────────────────────────────────────────────
# 4. MISSING VALUES ANALYSIS
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(" 4. MISSING VALUES ANALYSIS")
print("─" * 70)

# Count missing values per column
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100

# Filter to only show columns that HAVE missing values
missing_cols = missing[missing > 0].sort_values(ascending=False)

if len(missing_cols) == 0:
    print("\n✅ No missing values found in any column.")
else:
    print(f"\n⚠️  {len(missing_cols)} column(s) with missing values:")
    for col in missing_cols.index:
        print(f"   ➜ {col}: {missing_cols[col]:,} missing ({missing_pct[col]:.2f}%)")

# Also check for special sentinel values that may represent missing data
# The dataset uses -1 as a sentinel for "not applicable" in some features
sentinel_count = (df == -1).sum()
sentinel_cols = sentinel_count[sentinel_count > 0].sort_values(ascending=False)
if len(sentinel_cols) > 0:
    print(f"\n⚠️  {len(sentinel_cols)} column(s) contain sentinel value (-1):")
    for col in sentinel_cols.index[:10]:  # Show top 10
        print(f"   ➜ {col}: {sentinel_cols[col]:,} occurrences ({sentinel_cols[col]/len(df)*100:.1f}%)")
    if len(sentinel_cols) > 10:
        print(f"   ... and {len(sentinel_cols) - 10} more columns")

# ──────────────────────────────────────────────────────────────────────────
# 5. TARGET CLASS DISTRIBUTION
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(" 5. TARGET CLASS DISTRIBUTION")
print("─" * 70)

# Get the value counts for the target column
target_col = "URL_Type_obf_Type"
class_dist = df[target_col].value_counts()
class_pct = df[target_col].value_counts(normalize=True) * 100

print(f"\n📊 Class distribution for '{target_col}':\n")
print(f"   {'Class':<20} {'Count':>8}   {'Percentage':>10}")
print(f"   {'─'*20} {'─'*8}   {'─'*10}")
for cls in class_dist.index:
    print(f"   {cls:<20} {class_dist[cls]:>8,}   {class_pct[cls]:>9.2f}%")

print(f"\n   {'TOTAL':<20} {class_dist.sum():>8,}   {'100.00%':>10}")

# ──────────────────────────────────────────────────────────────────────────
# 6. VISUALIZE CLASS DISTRIBUTION (Saved as PNG)
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(" 6. GENERATING CLASS DISTRIBUTION CHART")
print("─" * 70)

# Set a modern dark style for the chart
plt.style.use('seaborn-v0_8-darkgrid')

# Create a figure with two subplots: bar chart + pie chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('URL Type Class Distribution — All.csv (ISCX-URL2016)',
             fontsize=16, fontweight='bold', y=1.02)

# --- Subplot 1: Horizontal Bar Chart ---
colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#3498db']
bars = axes[0].barh(class_dist.index, class_dist.values, color=colors, edgecolor='white')
axes[0].set_xlabel('Number of Samples', fontsize=12)
axes[0].set_title('Sample Count per Class', fontsize=13, fontweight='bold')
axes[0].invert_yaxis()  # Largest class at the top

# Add count labels on each bar
for bar, count, pct in zip(bars, class_dist.values, class_pct.values):
    axes[0].text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
                 f'{count:,} ({pct:.1f}%)', va='center', fontsize=10)

# --- Subplot 2: Pie Chart ---
wedges, texts, autotexts = axes[1].pie(
    class_dist.values,
    labels=class_dist.index,
    colors=colors,
    autopct='%1.1f%%',
    startangle=140,
    textprops={'fontsize': 10},
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
axes[1].set_title('Class Proportions', fontsize=13, fontweight='bold')

plt.tight_layout()

# Save the chart to disk
chart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "step1_class_distribution.png")
plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n✅ Chart saved to: {chart_path}")
plt.close()

# ──────────────────────────────────────────────────────────────────────────
# 7. QUICK STATISTICAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(" 7. QUICK STATISTICAL SUMMARY (Numerical Features)")
print("─" * 70)

# Select only numerical columns (excludes the target label column)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n   ➜ Number of numerical feature columns: {len(numerical_cols)}")

# Print basic stats: min, max, mean, std for a quick sanity check
desc = df[numerical_cols].describe().T[['mean', 'std', 'min', 'max']]
print(f"\n{desc.to_string()}")

# ──────────────────────────────────────────────────────────────────────────
# 8. FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(" ✅ STEP 1 COMPLETE — SUMMARY")
print("=" * 70)
print(f"""
   • Dataset       : All.csv
   • Total Samples  : {df.shape[0]:,}
   • Total Columns  : {df.shape[1]}
   • Features       : {df.shape[1] - 1} (all pre-extracted, numerical/categorical)
   • Target Column  : {target_col}
   • Classes         : {', '.join(class_dist.index.tolist())}
   • Missing Values  : {df.isnull().sum().sum():,} total NaN cells
   • Chart Saved To  : step1_class_distribution.png

   ➜ Ready to proceed to Step 2: Data Preprocessing
""")
