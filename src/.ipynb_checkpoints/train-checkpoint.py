# src/train.py

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib 

# Import functions from the module utils.py
from utils import extract_features, load_manual_labels 

# --- Path to data and models ---
INTERIM_DATA_PATH = 'data/interim/'
RAW_DATA_PATH = 'data/raw/'
MODELS_PATH = 'models/'
REPORTS_PATH = 'reports/' 

# Check directories exist
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(REPORTS_PATH, exist_ok=True)

# --- Model configuration ---
RANDOM_STATE = 42 
TEST_SIZE = 0.2   # 20% data for test (80% for training)

def train_model():
    print("--- Training started ---")

    # 1. Load cleaned time series 
    interim_file_path = os.path.join(INTERIM_DATA_PATH, 'all_fills_time_series.pkl')
    try:
        with open(interim_file_path, 'rb') as f:
            cleaned_fills_time_series = pickle.load(f)
        print(f"Time series succesfully loaded from: {interim_file_path}")
        print(f"Number of fills loaded: {len(cleaned_fills_time_series)}")
    except FileNotFoundError:
        print(f"Warning: file '{interim_file_path}' not found.")
        print("Check utils.py to have been run.")
        return 
    except Exception as e:
        print(f"Error while loading time series file: {e}")
        return

    # 2. Load labels
    labels_df = load_manual_labels(raw_data_path=RAW_DATA_PATH)
    if labels_df.empty:
        print("Cannot procede without labels. Quitting.")
        return

    print(f"Labelled fills: {len(labels_df)}")

    # 3. Feature extraction for the labelled fills
    print("Start feature extraction of the labelled fill...")
    features_list = []
    processed_fill_ids = []

    # Iterate over Fill_ID in label dataframe
    for fill_id in labels_df['Fill_ID'].unique():
        if fill_id in cleaned_fills_time_series:
            fill_data = cleaned_fills_time_series[fill_id]
            luminosity_values = fill_data['Luminosity']
            time_values = fill_data['Relative_Time']

            features = extract_features(luminosity_values, time_values)
            features['Fill_ID'] = fill_id 
            features_list.append(features)
            processed_fill_ids.append(fill_id)
        else:
            print(f"Warning: Fill_ID {fill_id} found in labels but not in time series. Skipped.")

    if not features_list:
        print("No feature extracted for labelled fills. Check data and labels.")
        return

    # Convert dictionary list into a dataframe
    features_df = pd.DataFrame(features_list)
    print(f"Feature extracted for {len(features_df)} fills.")

    # 4. Join features and labels
    
    final_df = pd.merge(features_df, labels_df, on='Fill_ID', how='inner')
    print(f"DataFrame finale con feature ed etichette creato. Dimensione: {final_df.shape}")

    # Manage missing values (NaN) in features:
    # 1)Remove rows with NaN: final_df.dropna(inplace=True) or 
    # 2)Impute NaN in numerical features with median
    # (exclude 'Fill_ID', 'Is_Valid', and boolean features 'has_...')
    feature_cols = [col for col in final_df.columns if col not in ['Fill_ID', 'Is_Valid', 'has_decay_start', 'has_flat_start']]
    for col in feature_cols:
        if final_df[col].isnull().any():
            median_val = final_df[col].median()
            final_df[col].fillna(median_val, inplace=True)
            

    print(f"Number of rows after imputation/management NaN: {final_df.shape[0]}")
    if final_df.empty:
        print("Empty DataFrame after NaN management. Quitting.")
        return

    # 5. Data preparation for the model
    # Features (X) and target variable (y)
    X = final_df.drop(columns=['Fill_ID', 'Is_Valid'])
    y = final_df['Is_Valid']

    # Print features used and preview 
    print("\nFeature utilizzate per il training:")
    print(X.columns.tolist())
    print("\nAnteprima del DataFrame di training (prime 5 righe):")
    print(X.head())
    print("\nDistribuzione delle classi nel dataset finale:")
    print(y.value_counts())

    # Subdivision in training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    print(f"\nDimension of the training set: {X_train.shape[0]} samples")
    print(f"Dimension of the test set: {X_test.shape[0]} samples")

    # 6. Training of the Random Forest model
    print("\nTraining of the Random Forest model started...")
    # Random Forest parameters can be set here (ig. n_estimators, max_depth, etc.)
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced')
    # 'class_weight='balanced' as classes are not balanced 50/50

    model.fit(X_train, y_train)
    print("Training complete!")

    # 7. Model evaluation
    print("\n--- Model evaluation on the test set ---")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:\n", report)

    print("\n--- Features importance ---")
    # X.columns is the DataFrame of the features
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    sorted_importances = feature_importances.sort_values(ascending=False)
    print(sorted_importances)

    # Features can be shown here 
    plt.figure(figsize=(10, 6))
    sorted_importances.plot(kind='barh')
    plt.title('Importanza delle Feature nel Random Forest')
    plt.xlabel('Importanza')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()


    # 8. Model saving 
    model_filename = os.path.join(MODELS_PATH, 'random_forest_model.joblib')
    joblib.dump(model, model_filename)
    print(f"\nTrained model saved in: {model_filename}")

    print("\n--- Training completed ---")

if __name__ == "__main__":
    train_model()