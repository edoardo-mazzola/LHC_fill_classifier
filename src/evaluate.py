# src/evaluate.py

import os
import pickle
import pandas as pd
import numpy as np
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import functions from utils.py
from utils import extract_features, load_manual_labels

# --- Path to data and models ---
INTERIM_DATA_PATH = 'data/interim/'
RAW_DATA_PATH = 'data/raw/'
MODELS_PATH = 'models/'
REPORTS_PATH = 'reports/'


# Check directories exist
os.makedirs(REPORTS_PATH, exist_ok=True)


# --- Configuration of the model (same used in train.py) ---
RANDOM_STATE = 42
TEST_SIZE = 0.2

def evaluate_model():
    print("--- Model evaluation started ---")

    # 1. Load cleaned data time series
    interim_file_path = os.path.join(INTERIM_DATA_PATH, 'all_fills_time_series.pkl')
    try:
        with open(interim_file_path, 'rb') as f:
            cleaned_fills_time_series = pickle.load(f)
        print(f"Data loaded succesfully from: {interim_file_path}")
    except FileNotFoundError:
        print(f"Warning: file '{interim_file_path}' not found. Check it.")
        return
    except Exception as e:
        print(f"Error occurred while loading file: {e}")
        return

    # 2. Load manual labels
    labels_df = load_manual_labels(raw_data_path=RAW_DATA_PATH)
    if labels_df.empty:
        print("Labels not found. Quitting.")
        return

    # 3. Feature extraction for each labelled fill
    print("Start feature extraction...")
    features_list = []
    for fill_id in labels_df['Fill_ID'].unique():
        if fill_id in cleaned_fills_time_series:
            fill_data = cleaned_fills_time_series[fill_id]
            
            luminosity_values = fill_data['Luminosity']
            time_values = fill_data['Relative_Time']

            features = extract_features(luminosity_values, time_values)
            features['Fill_ID'] = fill_id
            features_list.append(features)
        else:
            print(f"Warning: Fill_ID {fill_id} found in labels but not in data. Skipped.")

    if not features_list:
        print("No feature extracted. Check data and extract_features function.")
        return

    features_df = pd.DataFrame(features_list)
    final_df = pd.merge(features_df, labels_df, on='Fill_ID', how='inner')

    # Missing value control (NaN) 
    feature_cols = [col for col in final_df.columns if col not in ['Fill_ID', 'Is_Valid', 'has_decay_start', 'has_flat_start']]
    for col in feature_cols:
        if final_df[col].isnull().any():
            median_val = final_df[col].median() # Depends on the test set
            final_df[col].fillna(median_val, inplace=True)

    if final_df.empty:
        print("DataFrame empty. Quitting.")
        return

    # 4. Data preparation 
    X = final_df.drop(columns=['Fill_ID', 'Is_Valid'])
    y = final_df['Is_Valid']

    # Splitting in training and test set 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    # 5. Load trained model
    model_filename = os.path.join(MODELS_PATH, 'random_forest_model.joblib')
    try:
        model = joblib.load(model_filename)
        print(f"Model loaded from: {model_filename}")
    except FileNotFoundError:
        print(f"Warning: Model '{model_filename}' not found. Check train.py to have been run before.")
        return
    except Exception as e:
        print(f"Error while loading: {e}")
        return

    # 6. Evaluate on the test set
    print("\n--- Evaluation on the test set ---")
    y_pred = model.predict(X_test)

    # metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy on test set: {accuracy:.4f}")
    print(f"F1-Score on test set: {f1:.4f}")
    print("\nClassification Report:\n", report)
    print("\nConfusion matrix:\n", conf_matrix)

    

    # 7. Display confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Valid (0)', 'Valid (1)'],
                yticklabels=['Not Valid (0)', 'Valid (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title('Confusion matrix')
    
    # save graphic
    report_image_path = os.path.join(REPORTS_PATH, 'confusion_matrix.png')
    plt.savefig(report_image_path)
    print(f"Confusion matrix saved in: {report_image_path}")
    plt.show() 

    print("\n--- Evaluation process completed ---")

if __name__ == "__main__":
    evaluate_model()