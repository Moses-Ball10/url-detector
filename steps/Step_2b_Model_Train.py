import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
import os
warnings.filterwarnings('ignore')

# 1. Load Data
print("Loading balanced dataset...")
# Get base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
df = pd.read_csv(os.path.join(BASE_DIR, "Processed_URL_Dataset.csv"))
X = df.drop(columns=['URL_Type_obf_Type'])
y = df['URL_Type_obf_Type']

# 2. Encode Labels & Scale Features (Crucial for Website)
print("Preparing scalers and encoders...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))

# Train/Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Scale the data based on the training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler and encoder for the future web application
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
print("Saved 'scaler.pkl' and 'label_encoder.pkl' for the web backend.")

# Helper function for evaluation
def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate False Positive Rate (FPR)
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    FPR = np.mean(FP / (FP + TN + 1e-9)) 
    
    print(f"\n--- {model_name} ---")
    print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print(f"False Positive Rate (FPR): {FPR:.6f}")


# 3. Train XGBoost Model
print("\nInitializing XGBoost Model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300, 
    random_state=42, 
    n_jobs=-1,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8
)

print("Training XGBoost...")
xgb_model.fit(X_train_scaled, y_train)

print("\nEvaluating XGBoost...")
y_pred_xgb = xgb_model.predict(X_test_scaled)
evaluate_model(y_test, y_pred_xgb, "XGBoost Classifier")

# Save the XGBoost model
joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgboost_model.pkl"))
print("\nSaved 'xgboost_model.pkl' successfully! Ready for web deployment.")

# 4. Train Random Forest Model
print("\nInitializing Random Forest Model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

print("Training Random Forest...")
rf_model.fit(X_train_scaled, y_train)

print("\nEvaluating Random Forest...")
y_pred_rf = rf_model.predict(X_test_scaled)
evaluate_model(y_test, y_pred_rf, "Random Forest")

# Save the final Random Forest
joblib.dump(rf_model, os.path.join(MODEL_DIR, "random_forest.pkl"))
print("\nSaved 'random_forest.pkl' successfully! Ready for dual-model web deployment.")