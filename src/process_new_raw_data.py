# src/process_new_raw_data.py

import os
import pickle
import sys
from utils import clean_single_raw_file 

# --- Paths ---
NEW_RAW_DATA_PATH = 'data/new_raw/'
INTERIM_DATA_PATH = 'data/interim/'

# --- Directory---
os.makedirs(NEW_RAW_DATA_PATH, exist_ok=True) 
os.makedirs(INTERIM_DATA_PATH, exist_ok=True)

def process_new_raw_data_file(input_raw_filename, output_interim_filename):
    """
    Load a new .pkl file of raw data from 'new_raw', clean it and save it as a new .pkl in the interim directory.
    

    Args:
        input_raw_filename (str): Name of the new raw .pkl file (in data/new_raw/).
        output_interim_filename (str): Name of the .pkl for the cleaned data (saved in data/interim/).
    """
    print(f"--- Starting raw data processing for {input_raw_filename} ---")

    
    raw_file_path = os.path.join(NEW_RAW_DATA_PATH, input_raw_filename) 
    interim_file_path = os.path.join(INTERIM_DATA_PATH, output_interim_filename)

    
    cleaned_data = clean_single_raw_file(raw_file_path)

    if not cleaned_data:
        print("No data processed from raw file. Quitting.")
        return

    
    try:
        with open(interim_file_path, 'wb') as f:
            pickle.dump(cleaned_data, f)
        print(f"Cleaned data saved successfully to: {interim_file_path}")
        print(f"Number of fills in cleaned data: {len(cleaned_data)}")
    except Exception as e:
        print(f"Error saving cleaned interim data: {e}")
        return

    print("--- Raw data processing completed ---")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/process_new_raw_data.py <input_raw_filename.pkl> <output_interim_filename.pkl>")
        print("Example: python src/process_new_raw_data.py Run3_2025_new_data.pkl new_fills_2025_cleaned.pkl") 
        sys.exit(1)

    input_raw_filename = sys.argv[1]
    output_interim_filename = sys.argv[2]

    process_new_raw_data_file(input_raw_filename, output_interim_filename)