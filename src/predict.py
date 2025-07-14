# src/predict.py 
import os
import pickle
import joblib
import pandas as pd
# Import same functions from utils used to extract the features
from utils import extract_features, load_and_extract_raw_time_series # Assicurati di includere anche questa se ti serve il pkl originale

MODELS_PATH = 'models/'
INTERIM_DATA_PATH = 'data/interim/'

def predict_fill_validity(fill_id_to_predict):
    print(f"Model loading...")
    model_filename = os.path.join(MODELS_PATH, 'random_forest_model.joblib')
    try:
        model = joblib.load(model_filename)
        print("Model loaded succesfully.")
    except FileNotFoundError:
        print(f"Warning: Model '{model_filename}' not found.")
        return

    print(f"Load fill {fill_id_to_predict} data...")
    interim_file_path = os.path.join(INTERIM_DATA_PATH, 'all_fills_time_series.pkl')
    try:
        with open(interim_file_path, 'rb') as f:
            cleaned_fills_time_series = pickle.load(f)
    except FileNotFoundError:
        print(f"Warning: file '{interim_file_path}' not found.")
        return
    except Exception as e:
        print(f"Error while loading time series file: {e}")
        return

    if fill_id_to_predict in cleaned_fills_time_series:
        fill_data = cleaned_fills_time_series[fill_id_to_predict]
        luminosity_values = fill_data['Luminosity']
        time_values = fill_data['Relative_Time']

        print(f"Extraction fill {fill_id_to_predict} feature...")
        features = extract_features(luminosity_values, time_values)

        print(f"\nFeatures extracted for Fill_ID {fill_id_to_predict}:")
        for feature_name, value in features.items():
            if feature_name != 'Fill_ID': # Fill_ID excluded as it's not an imput feature
                print(f"  {feature_name}: {value:.4f}") 

    
        # Remove Fill_ID if in features,as model don't need it
        if 'Fill_ID' in features:
            del features['Fill_ID']

        
        features_series = pd.Series(features)
        
        for key in features_series.index:
            if pd.isna(features_series[key]) and key not in ['has_decay_start', 'has_flat_start']: 
                features_series[key] = 0 

        # Generate a DataFrame with a single row for the predictions
        X_predict = pd.DataFrame([features_series])


        prediction = model.predict(X_predict)
        prediction_proba = model.predict_proba(X_predict)

        print(f"Prediction for {fill_id_to_predict}: {'Valid' if prediction[0] == 1 else 'Not Valid'}")
        print(f"Probability: Valid={prediction_proba[0][1]:.4f}, Not Valid={prediction_proba[0][0]:.4f}")
    else:
        print(f"Fill_ID {fill_id_to_predict} not found.")

if __name__ == "__main__":
    # Prediction for a specific fill out of user's choice
    predict_fill_validity(8120) 
    predict_fill_validity(8148)