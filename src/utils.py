import pickle
import numpy as np
import pandas as pd
import os
from scipy.signal import savgol_filter 






# --- Function to load, join and clean the raw time series ---
def load_and_extract_raw_time_series(raw_data_path='../data/raw/', interim_data_path='../data/interim/'):
    """
    Load file .pkl raw, extract temporal series of luminosity and time and
    save in a new file .pkl cleaned in the directory interim.

    Args:
        raw_data_path (str): path to the directory with the original .pkl file.
        interim_data_path (str): path to save .pkl file of the cleaned time series.

    Returns:
        dict: dictionary with Fill_ID as key and {'Relative_Time': array, 'Luminosity': array} as value.
              Returns None if interim file has already been generated and load correctly.
    """
    # Construct the path of the interim file and verify if it already exists
    interim_file_path = os.path.join(interim_data_path, 'all_fills_time_series.pkl')
    if os.path.exists(interim_file_path):
        print(f"Interim file found '{interim_file_path}'. Loading...")
        with open(interim_file_path, 'rb') as f:
            return pickle.load(f)

    print("Interim file not found or incomplete. Loading from raw, processing...")
    os.makedirs(interim_data_path, exist_ok=True) 

    # Load the original .pkl files
    fills_data_by_year = {}
    for year in [2022, 2023, 2024]:
        file_path = os.path.join(raw_data_path, f'Run3_{year}_all_data.pkl')
        with open(file_path, 'rb') as f:
            fills_data_by_year[year] = pickle.load(f)[year]
    print("File .pkl loaded.")

    # Join all te fills' dictionaries 
    all_fills_raw_data = {}
    for year_data in fills_data_by_year.values():
        all_fills_raw_data.update(year_data)
    print(f"Totale fill univoci caricati: {len(all_fills_raw_data)}")

    # Extract time series of luminosity and time
    cleaned_fills_time_series = {}
    fills_with_missing_data = []

    print("Extraction for each fill started...")
    for fill_number, fill_content in all_fills_raw_data.items():
        try:
            luminosity_data = fill_content['Cleaned_Stable_Beams_Data']['cleaned_lumi_cut']
            time_data = fill_content['Cleaned_Stable_Beams_Data']['cleaned_relative_time']

            cleaned_fills_time_series[fill_number] = {
                'Relative_Time': np.array(time_data),
                'Luminosity': np.array(luminosity_data)
            }
        except KeyError as e:
            fills_with_missing_data.append(fill_number)
        except Exception as e:
            fills_with_missing_data.append(fill_number)

    if fills_with_missing_data:
        print(f"Warning: {len(fills_with_missing_data)} lost data or errors, fill skipped.")
    print(f"Extraction complete. Number of fills extracted: {len(cleaned_fills_time_series)}")

    # Save the new file .pkl of the clean time series
    output_file_path = interim_file_path
    with open(output_file_path, 'wb') as f:
        pickle.dump(cleaned_fills_time_series, f)
    print(f"Cleaned data saved in: {output_file_path}")

    return cleaned_fills_time_series







def extract_features(luminosity_values, time_values):
    """
    The functions extracts relevant numerical features from the temporal series luminosity data.

    The Function identify and quantify key features as the stable luminosity plateau and the fill decay,
    based on threshold values previously calibrated and smoothing Savitzky-Golay filter. Features concerning
    time duration, slopes, luminosity statistics and other relevant parameters are extrapolated.

    Args:
        luminosity_values (np.ndarray): Fill luminosity values.
        time_values (np.ndarray): Corresponding time values.

    Returns:
        dict: Dictionary of extracted features. Values are np.nan if not computable.
    """
    
    
    # Initialize all the feature at NaN, crucial for the next steps and the parameters initialization
    t0 = np.nan
    t_flat = np.nan
    durata_plateau = np.nan
    durata_decadimento = np.nan
    mean_dLdt_plateau = np.nan
    std_dLdt_plateau = np.nan
    mean_luminosity_plateau = np.nan
    total_fill_duration = np.nan # Nuova feature

    # Data length analysis
    if len(luminosity_values) < 20 or len(time_values) < 20: 
        
        return {
            't0_decay': t0,
            't_flat': t_flat,
            'durata_plateau': durata_plateau,
            'durata_decadimento': durata_decadimento,
            'mean_abs_dLdt_plateau': mean_dLdt_plateau,
            'std_dLdt_plateau': std_dLdt_plateau,
            'mean_luminosity_plateau': mean_luminosity_plateau,
            'has_decay_start': 0,
            'has_flat_start': 0,
            'max_luminosity': np.max(luminosity_values) if len(luminosity_values) > 0 else np.nan,
            'min_luminosity': np.min(luminosity_values) if len(luminosity_values) > 0 else np.nan,
            'time_series_length': len(time_values),
            'total_fill_duration': total_fill_duration
        }

    # The parameters used have been manually calibrated and their validity in the data recognition has been carefully analysed 
    
    smoothing_window_length = min(11, len(luminosity_values) - 1 if len(luminosity_values) % 2 == 0 else len(luminosity_values))
    if smoothing_window_length < 3: smoothing_window_length = 3 if len(luminosity_values) >= 3 else len(luminosity_values) - (len(luminosity_values) % 2 == 0);
    if smoothing_window_length % 2 == 0: smoothing_window_length -= 1 # odd
    if smoothing_window_length < 1: # short series
        # no savgol_filter, base features
        return {
            't0_decay': t0_decay, 't_flat': t_flat, 'durata_plateau': durata_plateau, 
            'durata_decadimento': durata_decadimento, 'mean_abs_dLdt_plateau': mean_dLdt_plateau,
            'std_dLdt_plateau': std_dLdt_plateau, 'mean_luminosity_plateau': mean_luminosity_plateau,
            'has_decay_start': 0, 'has_flat_start': 0,
            'max_luminosity': np.max(luminosity_values) if len(luminosity_values) > 0 else np.nan,
            'min_luminosity': np.min(luminosity_values) if len(luminosity_values) > 0 else np.nan,
            'time_series_length': len(time_values),
            'total_fill_duration': total_fill_duration
        }

    L_smooth = savgol_filter(luminosity_values, window_length=smoothing_window_length, polyorder=3)
    dLdt = np.gradient(L_smooth, time_values)

    # parameters to detect the decay part 
    soglia_decadimento = 1e3 
    window_decadimento = 200 

    # t0 time for decay
    for i in range(len(dLdt) - window_decadimento):
        
        if np.all(dLdt[i:i+window_decadimento] < soglia_decadimento):
            t0 = time_values[i]
            break

    # parameters to detect the plateau  
    flat_mean_threshold = 1e4  
    flat_std_threshold = 1e4   
    flat_window = 100          

    # t_flat
    for i in range(len(dLdt) - flat_window):
        window = dLdt[i:i+flat_window]
        mean_abs = np.mean(np.abs(window))
        std_dev = np.std(window)

        if mean_abs < flat_mean_threshold and std_dev < flat_std_threshold:
            t_flat = time_values[i]
            break

    # feature based on t_flat and t0
    if not np.isnan(t_flat) and not np.isnan(t0) and t_flat < t0:
        idx_flat_start_candidates = np.where(time_values >= t_flat)[0]
        idx_decay_start_candidates = np.where(time_values >= t0)[0]

        if len(idx_flat_start_candidates) > 0 and len(idx_decay_start_candidates) > 0:
            idx_flat_start = idx_flat_start_candidates[0]
            idx_decay_start = idx_decay_start_candidates[0]

            if idx_flat_start < idx_decay_start:
                # --- plateau features ---
                plateau_luminosity_segment = L_smooth[idx_flat_start:idx_decay_start]
                plateau_time_segment = time_values[idx_flat_start:idx_decay_start]

                if len(plateau_time_segment) > 1:
                    # initial plateau slope 
                    pendenza_iniziale = (plateau_luminosity_segment[-1] - plateau_luminosity_segment[0]) / \
                                        (plateau_time_segment[-1] - plateau_time_segment[0])
                    # time length of the plateau
                    durata_plateau = plateau_time_segment[-1] - plateau_time_segment[0]
                elif len(plateau_time_segment) == 1: 
                    pendenza_iniziale = 0.0
                    durata_plateau = 0.0

                # Statistics of dL/dt on the plateau
                plateau_dLdt_segment = dLdt[idx_flat_start:idx_decay_start]
                if len(plateau_dLdt_segment) > 0:
                    mean_dLdt_plateau = np.mean(np.abs(plateau_dLdt_segment)) # mean value dL/dt
                    std_dLdt_plateau = np.std(plateau_dLdt_segment) # std dev dL/dt

                # mean value of the luminosity in the levelled part
                if len(plateau_luminosity_segment) > 0:
                    mean_luminosity_plateau = np.mean(plateau_luminosity_segment)

                # --- decay features ---
                # decay until the time interval data stop
                decay_segment_luminosity = L_smooth[idx_decay_start:]
                decay_segment_time = time_values[idx_decay_start:]

                if len(decay_segment_time) > 1:
                    # deacy slope 
                    pendenza_decadimento = (decay_segment_luminosity[-1] - decay_segment_luminosity[0]) / \
                                           (decay_segment_time[-1] - decay_segment_time[0])
                    # decay length 
                    durata_decadimento = decay_segment_time[-1] - decay_segment_time[0]
                elif len(decay_segment_time) == 1: # Decadimento di un solo punto
                    pendenza_decadimento = 0.0
                    durata_decadimento = 0.0

    # --- tot fill time ---
    if len(time_values) > 1:
        total_fill_duration = time_values[-1] - time_values[0] # Differenza tra ultimo e primo tempo
    elif len(time_values) == 1:
        total_fill_duration = 0.0 # Se c'è un solo punto, la durata è 0

    # feature dictionary
    features = {
        't0_decay': t0,
        't_flat': t_flat,
        'durata_plateau': durata_plateau,
        'durata_decadimento': durata_decadimento,
        'mean_abs_dLdt_plateau': mean_dLdt_plateau,
        'std_dLdt_plateau': std_dLdt_plateau,
        'mean_luminosity_plateau': mean_luminosity_plateau,
        'has_decay_start': 0 if np.isnan(t0) else 1, 
        'has_flat_start': 0 if np.isnan(t_flat) else 1,   
        'max_luminosity': np.max(luminosity_values) if len(luminosity_values) > 0 else np.nan,
        'min_luminosity': np.min(luminosity_values) if len(luminosity_values) > 0 else np.nan,
        'time_series_length': len(time_values),
        'total_fill_duration': total_fill_duration 
    }

    return features








def load_manual_labels(raw_data_path='data/raw/', filename='manual_fill_labels.csv'):
    """
    Load manual labels of the fills from a CSV file.

    Args:
        raw_data_path (str): Path to the directory with the label files.
        filename (str): name of the CSV file of the labels.

    Returns:
        pd.DataFrame: dataframe with columns 'Fill_ID' and 'Is_Valid'.
                      Returns a void DataFrame if the file is not found.
    """
    labels_file_path = os.path.join(raw_data_path, filename)
    try:
        labels_df = pd.read_csv(labels_file_path)
        print(f"Labels loaded from: {labels_file_path}")
        #  'Fill_ID' integer for the corrispondance with the dictionary keys
        labels_df['Fill_ID'] = labels_df['Fill_ID'].astype(int)
        return labels_df
    except FileNotFoundError:
        print(f"Error: Label file '{labels_file_path}' not found.")
        print("Check file 'manual_fill_labels.csv' in 'data/raw/'.")
        return pd.DataFrame() 
    except Exception as e:
        print(f"Error in label loading: {e}")
        return pd.DataFrame()