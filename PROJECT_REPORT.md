# Multi-Class Malicious URL Detection System
## Executive Project Report

### 1. Project Overview
This project aimed to build a robust, AI-powered multi-class malicious URL detection system. Rather than creating a generic binary classifier (Safe vs. Malicious), this system distinguishes between 5 specific classes of URLs using Deep Learning: **Benign, Phishing, Malware, Spam, and Defacement**. The project was strictly constrained to use 79 pre-extracted lexical and structural features from the ISCX-URL2016 dataset, avoiding NLP-based text embedding techniques to focus entirely on tabular numerical data.

The finalized architecture is a hybrid system utilizing Python (for Deep Learning and Data Science) and PHP/Laravel (for the user-facing web dashboard). 

### 2. Step-by-Step Implementation Summary

#### Step 1: Data Exploration & Loading
* **Dataset Profile**: The `All.csv` dataset consisted of 36,707 samples with 79 features and 1 target label.
* **Class Balance**: The distribution was relatively balanced, ranging from 18.2% to 21.6% per class. No aggressive synthetic oversampling (like SMOTE) was required.
* **Data Quality Issues**: The initial review identified 19,183 `NaN` missing values, hundreds of thousands of sentinel values (`-1` used for N/A), and occurrences of infinite (`inf`) values in the `argPathRatio` column.

#### Step 2: Data Preprocessing & Cleaning 
* **Data Cleansing**: 
  * Replaced all `inf` variations with `NaN`.
  * Translated all `-1` sentinel values across 22 affected columns to `NaN`.
  * Imputed the resulting 305,147 `NaN` cells using the **Column Median** (which is more robust to outliers than the mean). For the `ISIpAddressInDomainName` column which was 100% missing data, a 0 fallback was used.
* **Transformation**: Handled string-based labels via `LabelEncoder` (Defacement: 0, benign: 1, malware: 2, phishing: 3, spam: 4).
* **Feature Scaling**: Implemented a `StandardScaler` to ensure all features had a mean of 0 and unit variance, which is critical for neural network convergence. Scaling was strictly fitted on the training split to prevent data leakage.
* **Data Splitting**: Executed an 80/20 train-test split using `stratify=y` to carefully preserve the class proportions across both sets.
* **Artifact Generation**: Saved the Scaler, Encoder, and Feature Names list to disk for later reuse in the production API.

#### Step 3: Deep Learning Model Architecture (DNN)
* **Architecture Design**: Constructed a 4-hidden-layer Deep Neural Network (DNN) using TensorFlow/Keras.
  * *Input*: 79 numerical features.
  * *Layers*: Dense(256) → Dense(128) → Dense(64) → Dense(32) → Output(5).
* **Advanced Optimizations**:
  * Incorporated `BatchNormalization` for training stability.
  * Used `LeakyReLU` activations to avoid "dying neuron" problems.
  * Added `Dropout` (20-30%) and `L2 Regularization` to prevent overfitting.
* **Training Dynamics**: Utilized the Adam optimizer with `SparseCategoricalCrossentropy` loss. Implemented dynamically weighted classes to combat minor imbalances and lower the False Positive Rate.
* **Callbacks**: Deployed `EarlyStopping` and `ReduceLROnPlateau` to halt training when validation improvements stalled, preventing overtraining.
* **Results**: 
  * **Test Accuracy**: 95.30%
  * **F1 Score**: 0.9533
  * **False Positive Rate (FPR)**: Outstandingly low at 1.18% (vital for cybersecurity tools to avoid user fatigue from false alarms).

#### Step 4: FastAPI Microservice (Python)
* **Design Pattern**: Decoupled the heavy AI processing from the web frontend by creating an isolated HTTP microservice via FastAPI.
* **Feature Engineering Bridge**: Developed extreme edge-case handling in a custom `extract_features` algorithm. Since users input raw url strings (e.g. `http://scam.com`) but the model demands 79 numerical vectors, this bridge manually parses structural token lengths, character distributions, ratios, and Shannon Entropies to dynamically generate a Pandas DataFrame mimicking the ISCX-URL2016 structure exactly.
* **Output**: Exposes the `/predict` POST endpoint returning a rich JSON response detailing Risk Level, Winning Prediction, and a mapping of confidence intervals for all 5 classes. 

#### Step 5: Laravel Integration (PHP)
* Built a dedicated `UrlScannerController` in Laravel.
* **Responsibilities**: Validates user web forms, makes a secure timeout-protected HTTP cURL request to the Python backend on port 8000, and gracefully passes the nested JSON response back to the Blade template.
* **Resilience**: If the AI engine is offline, the controller traps the exception and returns the user to the form with a seamless flash error alert, preventing fatal 500 crashes on the web tier.

#### Step 6: Frontend UI Dashboard (Tailwind CSS)
* Designed a responsive, modern HTML dashboard (`dashboard.blade.php`) using Tailwind CSS.
* **UX Highlights**: 
  * State-aware coloring: The result card turns Green for 'Safe/Benign', Red for 'Malware', Orange for 'Phishing', etc.
  * Deep Learning Transparency: Iterates the probability matrix from the API to visually render confidence progress-bars for all 5 potential threat types.

### 3. Conclusion
The resulting software is highly accurate, extremely modular, and ready for integration. By breaking the pipeline down into Data Cleaning ▸ Deep Learning ▸ API Microservice ▸ Laravel Frontend ▸ UI, the architecture achieves a true production-ready micro-service separation of concerns.
