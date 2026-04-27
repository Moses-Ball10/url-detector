# Malicious URL Detector 🛡️

A high-performance machine learning system designed to classify URLs into five categories: **Benign, Phishing, Malware, Spam, and Defacement**. This project features a robust feature extraction engine, a balanced training pipeline, and a modern, glassmorphic web interface.

## 📁 Project Structure

```text
.
├── app.py                  # Flask Web Application (Entry point)
├── extractor.py            # Core URL Feature Extraction Engine
├── models/                 # Serialized AI Models & Scalers
│   ├── xgboost_model.pkl   # Primary XGBoost Classifier
│   ├── random_forest.pkl   # Secondary Random Forest Classifier
│   ├── scaler.pkl          # StandardScaler for feature normalization
│   ├── label_encoder.pkl   # Class name encoder
│   └── expected_features.json # Schema for the 49-feature vector
├── steps/                  # Data Science & Training Pipeline
│   ├── Step_1a_Data_Preparation_Balancing.py
│   ├── Step_1b_Data_Visualization.py
│   └── Step_2b_Model_Train.py
├── images/                 # Generated Data Analysis Plots
├── templates/              # UI Components (HTML/CSS)
├── All.csv                 # Raw Training Dataset (ISCX-2016)
├── requirements.txt        # Python Dependencies
└── .gitignore              # Project exclusions
```

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.9+ installed.

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Application
Start the Flask server to launch the web interface.

```bash
python app.py
```
Visit `http://127.0.0.1:5000` in your browser.

## 🧠 Model & Pipeline

### Feature Extraction
The `extractor.py` module converts raw URLs into a numerical vector of 79 features, which is then filtered down to the **49 most impactful features** used for prediction. It analyzes:
- **Structural Properties**: Lengths, token counts, entropy.
- **Content Analysis**: Vowel counts, special characters, digit ratios.
- **Path/Domain Logic**: LDL (Letter-Digit-Letter) patterns, directory depth.

### Data Science Pipeline
The project follows a 3-step pipeline (located in `steps/`):
1. **Data Cleaning & Balancing**: Handles missing values and uses **SMOTE** to balance the 5 target classes.
2. **Visualization**: Performs **PCA** for class separability and calculates feature importance.
3. **Training**: Trains and evaluates an **XGBoost** model (primary) and a **Random Forest** (secondary).

## 📊 Performance
The system currently achieves ~98% accuracy on balanced test sets, with a focus on minimizing False Positives for Benign URLs.

## 🎨 UI/UX Features
- **Dark Mode**: Sleek dark-blue aesthetic.
- **Glassmorphism**: Premium frosted-glass cards with neon shadows.
- **Dynamic Feedback**: Real-time probability distributions for all 5 classes with animated circular gauges.

## 🏷️ Classification Categories

- **🟢 Benign**: Safe, standard URLs with natural path structures and no suspicious tracking.
- **🔴 Phishing**: Deceptive URLs mimicking brands, often using subdomains like `secure.paypal-verify.com`.
- **🟠 Malware**: URLs designed for automated file delivery, often featuring high digit counts and "LDL" patterns.
- **🟡 Spam**: High-tracking URLs with massive query strings (`?id=...&ref=...`) and high digit-to-letter ratios.
- **⚪ Defacement**: Direct-to-root URLs targeting specific system files, often with many probes in the query parameters.

---
*Developed as part of a Production-Ready Malicious URL Classification Project.*
