# src/analyze_predictions.py

# src/analyze_predictions.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import sys

# --- Path to reports ---
REPORTS_PATH = 'reports/'
# --- Path to processed data ---
PROCESSED_DATA_PATH = 'data/processed/'

# Check if reports directory exists
os.makedirs(REPORTS_PATH, exist_ok=True)

# --- Support functions (gauss, normalized_double_gauss, analyze_and_plot_distribution, perform_double_gaussian_fit) ---

def gauss(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(- (x - mu)**2 / (2 * sigma**2))

def normalized_double_gauss(x, w2, mu1, sigma1, mu2, sigma2):
    w1 = 1 - w2
    return w1 * gauss(x, mu1, sigma1) + w2 * gauss(x, mu2, sigma2)

def analyze_and_plot_distribution(data_series, title_suffix, xlabel, filename_prefix, year=None, data_type="", fit_choice=None):
    if data_series.empty:
        print(f"Skipping distribution plot for {title_suffix} - No data available.")
        return

    plt.figure(figsize=(10, 6))
    # No KDE by default for cleaner fit visualization
    sns.histplot(data_series, bins='auto', kde=False, color='skyblue', edgecolor='skyblue', stat='density', label='Data Histogram') 
    
    mean_val = data_series.mean()
    std_val = data_series.std()

    # Always show mean and std dev of the raw data
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean data: {mean_val:.2f}')
    plt.axvline(mean_val + std_val, color='green', linestyle=':', label=f'Std Dev data: {std_val:.2f}')
    plt.axvline(mean_val - std_val, color='green', linestyle=':')

    # Perform fit based on user choice
    fit_params = None
    fit_type_str = ""
    
    if fit_choice == '1': # Simple Gaussian Fit
        fit_params = perform_simple_gaussian_fit(data_series, title_suffix, xlabel, filename_prefix, year, data_type, plot_on_current_ax=True)
        fit_type_str = "_simple_gauss_fit"
    elif fit_choice == '2': # Double Gaussian Fit
        fit_params = perform_double_gaussian_fit(data_series, title_suffix, xlabel, filename_prefix, year, data_type, plot_on_current_ax=True)
        fit_type_str = "_double_gauss_fit"

    # Print fit parameters if successful
    if fit_params:
        print(f"  Fit parameters for {title_suffix} ({'All Years' if year is None else year}):")
        if fit_choice == '1':
            print(f"    Mu: {fit_params['mu']:.2f}, Sigma: {fit_params['sigma']:.2f}")
        elif fit_choice == '2':
            print(f"    Component 1: w1={fit_params['w1']:.2f}, mu1={fit_params['mu1']:.2f}, sigma1={fit_params['sigma1']:.2f}")
            print(f"    Component 2: w2={fit_params['w2']:.2f}, mu2={fit_params['mu2']:.2f}, sigma2={fit_params['sigma2']:.2f}")

    plt.xlabel(xlabel)
    plt.ylabel('Density')
    
    full_title = f'Distribution {title_suffix}'
    if year:
        full_title += f' ({year})'
    plt.title(full_title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    output_filename = os.path.join(REPORTS_PATH, f'{filename_prefix}{fit_type_str}_{data_type}_{year if year else "overall"}.png')
    plt.savefig(output_filename)
    print(f"Saved distribution plot to: {output_filename}")
    plt.close()


def perform_simple_gaussian_fit(data_series, title_suffix, xlabel, filename_prefix, year=None, data_type="", plot_on_current_ax=False):
    if data_series.empty or len(data_series) < 2:
        # print(f"Skipping simple Gaussian fit for {title_suffix} - Insufficient data.") # Avoid excessive messages
        return None

    counts, bins = np.histogram(data_series, bins='auto', density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    initial_guesses = [data_series.mean(), data_series.std()]
    bounds = ([-np.inf, 1e-6], [np.inf, np.inf]) # Sigma must be positive

    try:
        popt, pcov = curve_fit(gauss, bin_centers, counts, p0=initial_guesses, bounds=bounds, maxfev=5000)
        mu_fit, sigma_fit = popt
        
        x_fit = np.linspace(min(bin_centers), max(bin_centers), 500)
        plt.plot(x_fit, gauss(x_fit, *popt), color='purple', label=f'Simple Gaussian Fit (μ={mu_fit:.2f}, σ={sigma_fit:.2f})', linestyle='-', linewidth=2)
            
        return {'mu': mu_fit, 'sigma': sigma_fit}

    except RuntimeError:
        print(f"  Warning: Could not fit simple Gaussian for {title_suffix} ({year}). Check initial guesses or data.")
        return None
    except Exception as e:
        print(f"  An error occurred during simple Gaussian fit for {title_suffix} ({year}): {e}")
        return None


def perform_double_gaussian_fit(data_series, title_suffix, xlabel, filename_prefix, year=None, data_type="", plot_on_current_ax=False):
    if data_series.empty or len(data_series) < 2:
        
        return None

    counts, bins = np.histogram(data_series, bins='auto', density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

   
    
    # Calculate quantiles
    q1, q3 = np.percentile(data_series, [25, 75])
    
    # Initial guesses: one peak towards lower end, one towards higher end
    # Or simply spread around mean if data is not clearly bimodal
    mu1_guess = data_series.mean() - data_series.std() / 2 
    mu2_guess = data_series.mean() + data_series.std() / 2

    # Ensure initial sigma guesses are positive and reasonable
    sigma_guess = data_series.std() / 4 if data_series.std() > 0 else 0.1 # A smaller std dev for individual components

    # Make sure mu1_guess is less than mu2_guess for consistent interpretation if possible
    if mu1_guess > mu2_guess:
        mu1_guess, mu2_guess = mu2_guess, mu1_guess
    
    initial_guesses = [
        0.5,           # w2 (weight of second gaussian)
        mu1_guess,     # mu1
        sigma_guess,   # sigma1
        mu2_guess,     # mu2
        sigma_guess    # sigma2
    ]
    # Bounds: w2 [0,1], sigma1/sigma2 > 0
    bounds = ([0, -np.inf, 1e-6, -np.inf, 1e-6], [1, np.inf, np.inf, np.inf, np.inf])

    try:
        popt, pcov = curve_fit(normalized_double_gauss, bin_centers, counts, p0=initial_guesses, bounds=bounds, maxfev=5000)
        w2_fit, mu1_fit, sigma1_fit, mu2_fit, sigma2_fit = popt
        w1_fit = 1 - w2_fit

        # Ensure w1 and w2 are within [0,1] due to floating point inaccuracies
        w1_fit = np.clip(w1_fit, 0, 1)
        w2_fit = np.clip(w2_fit, 0, 1)

        # Plotting logic is now solely within analyze_and_plot_distribution
        x_fit = np.linspace(min(bin_centers), max(bin_centers), 500)
        
        gauss1_fitted = w1_fit * gauss(x_fit, mu1_fit, sigma1_fit)
        gauss2_fitted = w2_fit * gauss(x_fit, mu2_fit, sigma2_fit)
        sum_fit = normalized_double_gauss(x_fit, *popt)

        plt.plot(x_fit, sum_fit, color='red', label='Double Gaussian Fit', linestyle='-', linewidth=2)
        plt.plot(x_fit, gauss1_fitted, color='blue', linestyle='--', label=f'Gauss 1 (w={w1_fit:.2f}, μ={mu1_fit:.2f}, σ={sigma1_fit:.2f})')
        plt.plot(x_fit, gauss2_fitted, color='green', linestyle='--', label=f'Gauss 2 (w={w2_fit:.2f}, μ={mu2_fit:.2f}, σ={sigma2_fit:.2f})')

        return {'w1': w1_fit, 'mu1': mu1_fit, 'sigma1': sigma1_fit, 'w2': w2_fit, 'mu2': mu2_fit, 'sigma2': sigma2_fit}

    except RuntimeError:
        print(f"  Warning: Could not fit double Gaussian for {title_suffix} ({year}). Check initial guesses or data.")
        return None
    except Exception as e:
        print(f"  An error occurred during double Gaussian fit for {title_suffix} ({year}): {e}")
        return None


def analyze_predictions(input_analysis_filename):
    """
    Load data, generate distributions, and perform analysis based on user choice for each feature.

    Args:
        input_analysis_filename (str): Name of .pkl file with features and predictions
                                        (in data/processed/).
    """
    print(f"--- Analysis of fill parameters started for {input_analysis_filename} ---")

    analysis_data_path = os.path.join(PROCESSED_DATA_PATH, input_analysis_filename)
    try:
        df_risultati = pd.read_pickle(analysis_data_path)
        print(f"Analysis data loaded successfully from: {analysis_data_path}")
        print(f"DataFrame columns: {df_risultati.columns.tolist()}")
    except FileNotFoundError:
        print(f"Error: Analysis data file '{analysis_data_path}' not found. Ensure it was generated by evaluate.py or predict_new_data.py.")
        return
    except Exception as e:
        print(f"Error loading analysis data: {e}")
        return

    # Add 'Year' column if not present
    if 'Year' not in df_risultati.columns:
        df_risultati['Year'] = df_risultati['Fill_ID'].astype(str).apply(
            lambda x: int(x.split('_')[1]) if '_' in x and len(x.split('_')) > 1 and x.split('_')[1].isdigit() else None
        )
    
    # Years available in the dataset
    years_in_data = sorted(df_risultati['Year'].dropna().unique().tolist())
    if not years_in_data:
        print("No valid years found in data. Analyzing overall dataset.")
        years_to_process = [None]  # Process overall if no years
    else:
        years_to_process = years_in_data
    
    # Store user's fit choice for each parameter to avoid re-asking
    # This dictionary will map 'feature_name' -> 'fit_choice' ('0', '1', '2')
    param_fit_choices = {} 
            
    # Parameters to analyze (make sure these keys match your extract_features output)
    parameters_to_analyze = {
        'mean_luminosity_plateau': {'xlabel': 'Luminosità Media Plateau [fb-1]', 'filename': 'mean_lumi_plateau_dist'},
        'durata_plateau': {'xlabel': 'Durata Plateau [s]', 'filename': 'durata_plateau_dist'},
        'durata_decadimento': {'xlabel': 'Durata Decadimento [s]', 'filename': 'durata_decadimento_dist'},
        't0_decay': {'xlabel': 'Tempo Inizio Decadimento [s]', 'filename': 't0_decay_dist'},
        't_flat': {'xlabel': 'Tempo Inizio Plateau [s]', 'filename': 't_flat_dist'},
        'mean_abs_dLdt_plateau': {'xlabel': 'Media |dL/dt| Plateau', 'filename': 'mean_dLdt_plateau_dist'},
        'std_dLdt_plateau': {'xlabel': 'Std Dev dL/dt Plateau', 'filename': 'std_dLdt_plateau_dist'},
        'total_fill_duration': {'xlabel': 'Durata Totale Fill [s]', 'filename': 'total_fill_duration_dist'},
        'max_luminosity': {'xlabel': 'Luminosità Massima', 'filename': 'max_lumi_dist'},
        'min_luminosity': {'xlabel': 'Luminosità Minima', 'filename': 'min_lumi_dist'},
        'time_series_length': {'xlabel': 'Lunghezza Serie Temporale', 'filename': 'ts_length_dist'}
        # 'has_decay_start' and 'has_flat_start' are binary and not typically suited for distribution fitting
    }
    
    for param, info in parameters_to_analyze.items():
        if param not in df_risultati.columns:
            print(f"Warning: Parameter '{param}' not found in the DataFrame. Skipping analysis for this parameter.")
            continue
        
        # Ask user for fit choice for this specific parameter only once
        if param not in param_fit_choices:
            fit_choice_for_param = None
            while fit_choice_for_param not in ['0', '1', '2']:
                print(f"\n--- Choose fit type for parameter: '{param.replace('_', ' ').upper()}' ---")
                print("0: No fit (only histogram and data mean/std dev)")
                print("1: Simple Gaussian Fit")
                print("2: Double Gaussian Fit")
                fit_choice_for_param = input("Enter your choice (0, 1, or 2): ").strip()
                if fit_choice_for_param not in ['0', '1', '2']:
                    print("Invalid choice. Please enter 0, 1, or 2.")
            param_fit_choices[param] = fit_choice_for_param
        else:
            fit_choice_for_param = param_fit_choices[param] # Use stored choice

        for year in years_to_process:
            
            if year is not None:
                data_filtered_by_year = df_risultati[df_risultati['Year'] == year]
            else:
                data_filtered_by_year = df_risultati 

            # 1. Analysis for fills with real labels (if 'Is_Valid' column exists)
            # This part is relevant when analyzing outputs from 'evaluate.py'
            if 'Is_Valid' in data_filtered_by_year.columns and data_filtered_by_year['Is_Valid'].dropna().any():
                print(f"\nAnalyzing REAL VALID data for '{param}' ({'All Years' if year is None else year})...")
                real_valid_data = data_filtered_by_year[data_filtered_by_year['Is_Valid'] == 1][param].dropna()
                
                if not real_valid_data.empty:
                    analyze_and_plot_distribution(
                        real_valid_data,
                        f"real valid fills for {param.replace('_', ' ')}",
                        info['xlabel'],
                        info['filename'],
                        year,
                        "real_valid",
                        fit_choice_for_param # Pass the specific choice for this parameter
                    )
                else:
                    print(f"No real valid data for '{param}' in {'All Years' if year is None else year}.")

            # 2. Analysis for predicted valid fills (relevant for both 'evaluate.py' and 'predict_new_data.py' outputs)
            print(f"\nAnalyzing PREDICTED VALID data for '{param}' ({'All Years' if year is None else year})...")
            predicted_valid_data = data_filtered_by_year[data_filtered_by_year['Predicted_Is_Valid'] == 1][param].dropna()

            if not predicted_valid_data.empty:
                analyze_and_plot_distribution(
                    predicted_valid_data,
                    f"predicted valid fills for {param.replace('_', ' ')}",
                    info['xlabel'],
                    info['filename'],
                    year,
                    "predicted_valid",
                    fit_choice_for_param # Pass the specific choice for this parameter
                )
            else:
                print(f"No predicted valid data for '{param}' in {'All Years' if year is None else year}.")

    print("\n--- Analysis completed. Check the 'reports/' folder for plots. ---")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/analyze_predictions.py <input_analysis_filename.pkl>")
        print("Example: python src/analyze_predictions.py new_fill_predictions_2025_for_analysis.pkl")
        print("Or: python src/analyze_predictions.py fill_features_and_predictions_for_analysis.pkl (for existing data)")
        sys.exit(1)

    input_filename = sys.argv[1]
    analyze_predictions(input_filename)