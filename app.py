from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import traceback
import json
import xgboost as xgb
from extractor import URLExtractor 
from urllib.parse import urlparse

# Static list of known safe domains
SAFE_DOMAINS = {
    'google.com', 'www.google.com',
    'youtube.com', 'www.youtube.com',
    'facebook.com', 'www.facebook.com',
    'github.com', 'www.github.com',
    'wikipedia.org', 'en.wikipedia.org',
    'twitter.com', 'www.twitter.com',
    'linkedin.com', 'www.linkedin.com',
    'microsoft.com', 'www.microsoft.com',
    'apple.com', 'www.apple.com'
}

app = Flask(__name__)

# 1. Load the AI components
print("--- Loading Model and Scalers ---")
expected_features = []
try:
    xgb_model = joblib.load('models/xgboost_model.pkl')
    rf_model = joblib.load('models/random_forest.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    
    with open('models/expected_features.json', 'r') as f:
        expected_features = json.load(f)
        
    print(f"AI Components Loaded! Expecting {len(expected_features)} features.")
except Exception as e:
    print(f"Error loading files: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get URL from the frontend
        url = request.form.get('url')
        model_type = request.form.get('model_type', 'xgb') # Default to xgb
        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        print(f"\n{'='*50}")
        print(f"STEP 1: Received Request")
        print(f"URL   : {url}")
        print(f"Model : {model_type.upper()}")
        print(f"{'='*50}")

        # 1.5 Static Safe List Check
        parsed_url = urlparse(url if url.startswith('http') else f'http://{url}')
        if parsed_url.netloc.lower() in SAFE_DOMAINS:
            print(" -> STATIC ANALYSIS: URL matched safe domains list. Returning Benign.")
            probs_dict = {
                'benign': 100.0,
                'phishing': 0.0,
                'malware': 0.0,
                'spam': 0.0,
                'Defacement': 0.0
            }
            return jsonify({
                'url': url,
                'prediction': 'benign',
                'confidence': 100.0,
                'probabilities': probs_dict
            })

        # 2. Extract Features
        print(f"\nSTEP 2: Extracting raw features...")
        extractor = URLExtractor(url)
        raw_features_df = extractor.extract() # Now returns a DataFrame with all original columns
        print(f" -> Successfully extracted {raw_features_df.shape[1]} raw features from URL structure.")
        
        # 3. Filter exactly what the model expects, in the exact same order
        print(f"\nSTEP 3: Filtering features to match training schema...")
        final_features_df = raw_features_df[expected_features]
        print(f" -> Filtered down to {final_features_df.shape[1]} expected features.")

        # --- DEBUGGING BLOCK: Print final features to Terminal ---
        print("\n--- EXTRACTED FEATURE VALUES FOR PREDICTION ---")
        for col in final_features_df.columns:
            val = final_features_df[col].iloc[0]
            # Print non-zero values in green for easier reading, zeros in gray
            if val != 0:
                print(f"\033[92m{col:35}: {val}\033[0m")
            else:
                print(f"\033[90m{col:35}: {val}\033[0m")
        print(f"{'-'*47}\n")
        # --- END DEBUGGING BLOCK ---

        # 4. Scale the features
        print(f"STEP 4: Scaling features using standard scaler...")
        scaled_features = scaler.transform(final_features_df)
        
        # 5. Perform Prediction based on selected model
        print(f"\nSTEP 5: Running prediction through {model_type.upper()} model...")
        if model_type == 'rf':
            prediction_probs = rf_model.predict_proba(scaled_features)[0]
        else:
            prediction_probs = xgb_model.predict_proba(scaled_features)[0]
            
        class_index = np.argmax(prediction_probs)
        
        # Convert index back to text
        result = label_encoder.inverse_transform([class_index])[0]
        confidence = float(np.max(prediction_probs) * 100)
        
        print(f"\nSTEP 6: Final Decision")
        print(f"Result     : \033[1m{result.upper()}\033[0m")
        print(f"Confidence : {confidence:.2f}%")
        print(f"{'='*50}\n")

        # Build full probability breakdown for frontend
        all_probs = {}
        for i, class_name in enumerate(label_encoder.classes_):
            all_probs[class_name] = round(float(prediction_probs[i] * 100), 2)

        return jsonify({
            'url': url,
            'prediction': result,
            'confidence': confidence,
            'probabilities': all_probs
        })

    except Exception as e:
        print("\n--- CRITICAL ERROR DURING PREDICTION ---")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)