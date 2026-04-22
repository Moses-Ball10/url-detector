"""
=============================================================================
 STEP 3: DEEP LEARNING MODEL — BUILD, TRAIN & EVALUATE
=============================================================================
 Project : Multi-Class Malicious URL Detection System
 Model   : Deep Neural Network (DNN / Multi-Layer Perceptron)
 Input   : 79 pre-extracted features (scaled, from Step 2)
 Output  : 5-class classification (Defacement, Benign, Malware, Phishing, Spam)
 Goal    : High accuracy + Low False Positive Rate (FPR)
=============================================================================
"""

# ──────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────────────────
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# TensorFlow / Keras imports
import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks, regularizers

# Scikit-learn evaluation metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Reproducibility — fix random seeds for consistent results
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ──────────────────────────────────────────────────────────────────────────
# 2. LOAD PREPROCESSED DATA & ARTIFACTS
# ──────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

print("=" * 70)
print(" STEP 3: DEEP LEARNING MODEL — BUILD, TRAIN & EVALUATE")
print("=" * 70)

# Load the preprocessed train/test arrays saved in Step 2
data = np.load(os.path.join(ARTIFACTS_DIR, "preprocessed_data.npz"))
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# Load the label encoder to map predictions back to class names
with open(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"), 'rb') as f:
    label_encoder = pickle.load(f)

CLASS_NAMES = label_encoder.classes_.tolist()
NUM_CLASSES = len(CLASS_NAMES)
NUM_FEATURES = X_train.shape[1]

print(f"\n📂 Data loaded from: {ARTIFACTS_DIR}")
print(f"   ➜ X_train : {X_train.shape}")
print(f"   ➜ X_test  : {X_test.shape}")
print(f"   ➜ Classes : {CLASS_NAMES}")
print(f"   ➜ Features: {NUM_FEATURES}")

# ──────────────────────────────────────────────────────────────────────────
# 3. COMPUTE CLASS WEIGHTS
# ──────────────────────────────────────────────────────────────────────────
# Even though our classes are relatively balanced (~18-22%), applying class
# weights ensures the model doesn't slightly favor the majority class.
# This helps reduce False Positive Rate (FPR) across all classes.
print("\n" + "─" * 70)
print(" 3. COMPUTING CLASS WEIGHTS")
print("─" * 70)

class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Convert to dictionary format {class_index: weight}
class_weights = dict(enumerate(class_weights_array))

print("\n   Class weights (inversely proportional to frequency):")
for idx, weight in class_weights.items():
    print(f"   ➜ {CLASS_NAMES[idx]:<15}: {weight:.4f}")

# ──────────────────────────────────────────────────────────────────────────
# 4. BUILD THE DNN MODEL
# ──────────────────────────────────────────────────────────────────────────
# Architecture: Input(79) → Dense(256) → Dense(128) → Dense(64) → Dense(32) → Output(5)
#
# Key design decisions for HIGH ACCURACY + LOW FPR:
#   • BatchNormalization: Stabilizes training, allows higher learning rates
#   • Dropout: Prevents overfitting (regularization)
#   • L2 Regularization: Additional overfitting protection on dense layers
#   • LeakyReLU: Avoids "dying neuron" problem common with standard ReLU
#   • Softmax output: Produces probability distribution over 5 classes
print("\n" + "─" * 70)
print(" 4. BUILDING THE DNN MODEL")
print("─" * 70)

def build_model(input_dim, num_classes):
    """
    Build a Deep Neural Network optimized for multi-class URL classification.
    
    Args:
        input_dim  : Number of input features (79)
        num_classes: Number of output classes (5)
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # ── Input Layer ──
        layers.Input(shape=(input_dim,), name='input_features'),
        
        # ── Hidden Layer 1: 256 neurons ──
        layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4), name='dense_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.LeakyReLU(negative_slope=0.1),  # Small slope for negative inputs
        layers.Dropout(0.3, name='dropout_1'),             # 30% dropout
        
        # ── Hidden Layer 2: 128 neurons ──
        layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4), name='dense_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Dropout(0.3, name='dropout_2'),
        
        # ── Hidden Layer 3: 64 neurons ──
        layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4), name='dense_3'),
        layers.BatchNormalization(name='bn_3'),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Dropout(0.2, name='dropout_3'),             # Less dropout near output
        
        # ── Hidden Layer 4: 32 neurons ──
        layers.Dense(32, kernel_regularizer=regularizers.l2(1e-4), name='dense_4'),
        layers.BatchNormalization(name='bn_4'),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Dropout(0.2, name='dropout_4'),
        
        # ── Output Layer: 5 classes with softmax ──
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='URL_Classifier_DNN')
    
    # Compile with Adam optimizer and sparse categorical crossentropy
    # (sparse = labels are integers, not one-hot encoded)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

model = build_model(NUM_FEATURES, NUM_CLASSES)

# Print the model architecture
model.summary()

# ──────────────────────────────────────────────────────────────────────────
# 5. DEFINE TRAINING CALLBACKS
# ──────────────────────────────────────────────────────────────────────────
# Callbacks automate training optimizations:
#   • EarlyStopping   : Stop if validation loss doesn't improve (prevent overfitting)
#   • ReduceLROnPlateau: Reduce learning rate when training stalls
#   • ModelCheckpoint  : Save the best model (by val_accuracy)
print("\n" + "─" * 70)
print(" 5. CONFIGURING TRAINING CALLBACKS")
print("─" * 70)

# Create directory for the saved model
model_dir = os.path.join(ARTIFACTS_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

best_model_path = os.path.join(model_dir, "url_classifier_best.keras")

training_callbacks = [
    # Stop training if val_loss doesn't improve for 7 epochs
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,  # Restore the best weights at the end
        verbose=1
    ),
    
    # Reduce learning rate by 50% if val_loss stalls for 3 epochs
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,           # Multiply LR by 0.5
        patience=3,
        min_lr=1e-6,          # Don't go below this
        verbose=1
    ),
    
    # Save the best model checkpoint
    callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("   ✅ EarlyStopping  (patience=7, monitor=val_loss)")
print("   ✅ ReduceLROnPlateau (factor=0.5, patience=3)")
print(f"   ✅ ModelCheckpoint (save_best → {best_model_path})")

# ──────────────────────────────────────────────────────────────────────────
# 6. TRAIN THE MODEL
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(" 6. TRAINING THE MODEL")
print("─" * 70)

EPOCHS = 30           # Max epochs (EarlyStopping will likely stop earlier)
BATCH_SIZE = 64       # Mini-batch size

print(f"\n   ➜ Max Epochs  : {EPOCHS}")
print(f"   ➜ Batch Size  : {BATCH_SIZE}")
print(f"   ➜ Optimizer   : Adam (lr=0.001)")
print(f"   ➜ Loss        : Sparse Categorical Crossentropy")
print(f"   ➜ Validation  : 15% of training data\n")

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.15,       # 15% of training data for validation
    class_weight=class_weights,  # Apply class balancing weights
    callbacks=training_callbacks,
    verbose=1
)

# ──────────────────────────────────────────────────────────────────────────
# 7. EVALUATE ON TEST SET
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(" 7. EVALUATING ON TEST SET")
print("─" * 70)

# Get predictions
y_pred_probs = model.predict(X_test, verbose=0)       # Probability for each class
y_pred = np.argmax(y_pred_probs, axis=1)               # Predicted class (highest prob)

# Overall accuracy
test_accuracy = accuracy_score(y_test, y_pred)
test_f1_macro = f1_score(y_test, y_pred, average='macro')
test_f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"\n   📊 TEST SET RESULTS:")
print(f"   ➜ Accuracy          : {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   ➜ F1 Score (macro)  : {test_f1_macro:.4f}")
print(f"   ➜ F1 Score (weighted): {test_f1_weighted:.4f}")

# Detailed classification report (precision, recall, F1 per class)
print(f"\n   📋 CLASSIFICATION REPORT:\n")
report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
print(report)

# ──────────────────────────────────────────────────────────────────────────
# 8. FALSE POSITIVE RATE (FPR) ANALYSIS
# ──────────────────────────────────────────────────────────────────────────
# FPR per class = FP / (FP + TN)
# Low FPR is critical: we don't want benign URLs flagged as malicious.
print("─" * 70)
print(" 8. FALSE POSITIVE RATE (FPR) PER CLASS")
print("─" * 70)

cm = confusion_matrix(y_test, y_pred)

print(f"\n   {'Class':<15} {'FPR':>8}   {'FP':>6}  {'TN':>6}   Status")
print(f"   {'─'*15} {'─'*8}   {'─'*6}  {'─'*6}   {'─'*10}")

fpr_values = {}
for i, cls in enumerate(CLASS_NAMES):
    # False Positives: predicted as class i, but actually not class i
    fp = cm[:, i].sum() - cm[i, i]
    # True Negatives: not class i and correctly not predicted as class i
    tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fpr_values[cls] = fpr
    status = "✅ Low" if fpr < 0.05 else "⚠️ Watch" if fpr < 0.10 else "❌ High"
    print(f"   {cls:<15} {fpr:>8.4f}   {fp:>6}  {tn:>6}   {status}")

avg_fpr = np.mean(list(fpr_values.values()))
print(f"\n   ➜ Average FPR: {avg_fpr:.4f}")

# ──────────────────────────────────────────────────────────────────────────
# 9. CONFUSION MATRIX HEATMAP
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(" 9. GENERATING CONFUSION MATRIX HEATMAP")
print("─" * 70)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Model Evaluation — DNN URL Classifier', fontsize=16, fontweight='bold')

# --- Subplot 1: Confusion Matrix (Counts) ---
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    ax=axes[0], linewidths=0.5, cbar_kws={'shrink': 0.8}
)
axes[0].set_xlabel('Predicted Label', fontsize=12)
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_title('Confusion Matrix (Counts)', fontsize=13, fontweight='bold')

# --- Subplot 2: Confusion Matrix (Normalized %) ---
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(
    cm_normalized, annot=True, fmt='.2%', cmap='Greens',
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    ax=axes[1], linewidths=0.5, cbar_kws={'shrink': 0.8}
)
axes[1].set_xlabel('Predicted Label', fontsize=12)
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_title('Confusion Matrix (Normalized %)', fontsize=13, fontweight='bold')

plt.tight_layout()
cm_path = os.path.join(BASE_DIR, "step3_confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n   ✅ Confusion matrix saved to: {cm_path}")
plt.close()

# ──────────────────────────────────────────────────────────────────────────
# 10. TRAINING HISTORY PLOTS
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(" 10. GENERATING TRAINING HISTORY CHARTS")
print("─" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Training History — DNN URL Classifier', fontsize=16, fontweight='bold')

epochs_range = range(1, len(history.history['loss']) + 1)

# --- Subplot 1: Loss ---
axes[0].plot(epochs_range, history.history['loss'], 'r-', label='Training Loss', linewidth=2)
axes[0].plot(epochs_range, history.history['val_loss'], 'b-', label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# --- Subplot 2: Accuracy ---
axes[1].plot(epochs_range, history.history['accuracy'], 'r-', label='Training Accuracy', linewidth=2)
axes[1].plot(epochs_range, history.history['val_accuracy'], 'b-', label='Validation Accuracy', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
hist_path = os.path.join(BASE_DIR, "step3_training_history.png")
plt.savefig(hist_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n   ✅ Training history chart saved to: {hist_path}")
plt.close()

# ──────────────────────────────────────────────────────────────────────────
# 11. SAVE THE FINAL MODEL
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(" 11. SAVING THE FINAL MODEL")
print("─" * 70)

# Save in both .keras (modern) and .h5 (legacy) formats
keras_path = os.path.join(ARTIFACTS_DIR, "url_classifier.keras")
h5_path = os.path.join(ARTIFACTS_DIR, "url_classifier.h5")

model.save(keras_path)
model.save(h5_path)

print(f"\n   ✅ Model saved (.keras): {keras_path}")
print(f"   ✅ Model saved (.h5)   : {h5_path}")

# ──────────────────────────────────────────────────────────────────────────
# 12. QUICK INFERENCE TEST
# ──────────────────────────────────────────────────────────────────────────
# Verify the saved model works by loading it and predicting on a few samples.
print("\n" + "─" * 70)
print(" 12. QUICK INFERENCE TEST (Verifying Saved Model)")
print("─" * 70)

loaded_model = keras.models.load_model(keras_path)
sample_preds = loaded_model.predict(X_test[:5], verbose=0)

print(f"\n   Predictions on first 5 test samples:")
for i in range(5):
    true_label = CLASS_NAMES[y_test[i]]
    pred_label = CLASS_NAMES[np.argmax(sample_preds[i])]
    confidence = np.max(sample_preds[i]) * 100
    match = "✅" if true_label == pred_label else "❌"
    print(f"   {match} Sample {i+1}: True={true_label:<12} → Predicted={pred_label:<12} (Confidence: {confidence:.1f}%)")

# ──────────────────────────────────────────────────────────────────────────
# 13. FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(" ✅ STEP 3 COMPLETE — SUMMARY")
print("=" * 70)
print(f"""
   Model Architecture:
   ├── Input       : {NUM_FEATURES} features
   ├── Dense(256)  → BatchNorm → LeakyReLU → Dropout(0.3)
   ├── Dense(128)  → BatchNorm → LeakyReLU → Dropout(0.3)
   ├── Dense(64)   → BatchNorm → LeakyReLU → Dropout(0.2)
   ├── Dense(32)   → BatchNorm → LeakyReLU → Dropout(0.2)
   └── Dense({NUM_CLASSES})    → Softmax (output)
   
   Results:
   ├── Test Accuracy      : {test_accuracy*100:.2f}%
   ├── F1 Score (macro)   : {test_f1_macro:.4f}
   ├── F1 Score (weighted): {test_f1_weighted:.4f}
   └── Average FPR        : {avg_fpr:.4f}
   
   Saved Files:
   ├── artifacts/url_classifier.keras  — Final model (Keras format)
   ├── artifacts/url_classifier.h5     — Final model (H5 format)
   ├── artifacts/model/url_classifier_best.keras — Best checkpoint
   ├── step3_confusion_matrix.png      — Confusion matrix chart
   └── step3_training_history.png      — Loss/Accuracy curves
   
   ➜ Ready to proceed to Step 4: FastAPI Microservice
""")
