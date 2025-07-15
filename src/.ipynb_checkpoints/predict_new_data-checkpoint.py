# src/predict_new_data.py

import os
import pickle
import pandas as pd
import joblib
import sys
import numpy as np


from utils import extract_features

# --- Paths ---
INTERIM_DATA_PATH = 'data/interim/'
MODELS_PATH = 'models/'
PROCESSED_DATA_PATH = 'data/processed/' 


os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

def predict_and_prepare_for_analysis(input_interim_filename, output_analysis_filename):
    """
    Load a file .pkl of cleaned data from interim, it extracts features,
    makes predictions and saves data for the further distribution analysis.

    Args:
        input_interim_filename (str): name of the .pkl file of cleaned data (from data/interim/).
        output_analysis_filename (str): name of the file .pkl where to save extracted features
                                        and predictions (in data/processed/).
    """
    print(f"--- Starting prediction for data from {input_interim_filename} ---")

    interim_file_path = os.path.join(INTERIM_DATA_PATH, input_interim_filename)

    # 1. Load cleaned data from specific file in interim
    try:
        with open(interim_file_path, 'rb') as f:
            cleaned_data_for_prediction = pickle.load(f)
        print(f"Cleaned data loaded from: {interim_file_path}")
    except FileNotFoundError:
        print(f"Error: Cleaned data file '{interim_file_path}' not found. Please run process_new_raw_data.py first.")
        return
    except Exception as e:
        print(f"Error loading cleaned data: {e}")
        return

    # 2. Extracts features
    print("Extracting features from cleaned data...")
    features_list = []
    
    for fill_id, fill_data in cleaned_data_for_prediction.items():
        luminosity_values = fill_data['Luminosity']
        time_values = fill_data['Relative_Time']
        
        features = extract_features(luminosity_values, time_values)
        features['Fill_ID'] = fill_id
        features_list.append(features)

    if not features_list:
        print("No features extracted. Exiting.")
        return

    features_df = pd.DataFrame(features_list)
    
    # Handling missing values (NaN) in features
    
    print("Handling missing values in extracted features (using current data's median as fallback)...")
    for col in features_df.columns:
        if features_df[col].isnull().any():
            if features_df[col].dtype in ['float64', 'int64']: 
                median_val = features_df[col].median()
                features_df[col].fillna(median_val, inplace=True)
            else:
                features_df[col].fillna('unknown', inplace=True)

    # 3. Load trained model
    model_filename = os.path.join(MODELS_PATH, 'random_forest_model.joblib')
    try:
        model = joblib.load(model_filename)
        print(f"Model loaded from: {model_filename}")
    except FileNotFoundError:
        print(f"Error: Trained model '{model_filename}' not found. Please run train.py first.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Make predictions
    feature_cols_for_model = [col for col in features_df.columns if col not in ['Fill_ID']]
    X_for_prediction = features_df[feature_cols_for_model]

    print("Making predictions...")
    y_pred = model.predict(X_for_prediction)

    # 5. Create and saves dataframe for the analysis
    analysis_data_df = features_df.copy()
    analysis_data_df['Predicted_Is_Valid'] = y_pred
    analysis_data_df['Is_Valid'] = np.nan 

    output_analysis_path = os.path.join(PROCESSED_DATA_PATH, output_analysis_filename)
    try:
        analysis_data_df.to_pickle(output_analysis_path)
        print(f"Prediction results saved to: {output_analysis_path}")
        print(f"Shape of saved data: {analysis_data_df.shape}")
    except Exception as e:
        print(f"Error saving prediction results: {e}")

    print("\n--- Prediction process completed ---")
    print(f"Next, run: python src/analyze_predictions.py {output_analysis_filename} to analyze distributions.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/predict_new_data.py <input_interim_filename.pkl> <output_analysis_filename.pkl>")
        print("Example: python src/predict_new_data.py new_fills_2025_cleaned.pkl new_fill_predictions_2025_for_analysis.pkl")
        sys.exit(1)

    input_interim_filename = sys.argv[1]
    output_analysis_filename = sys.argv[2]

    predict_and_prepare_for_analysis(input_interim_filename, output_analysis_filename)