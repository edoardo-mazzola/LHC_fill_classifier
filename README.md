# Automatic classification of valid fills for the LHC Run 3 and HL-LHC 

## Introduction
The project implemented aims to facilitate the procedure of validation of LHC fills in presence of luminosity levelling. The Machine Learning algorithm implemented is able to recognise, after being trained on LHC RUN 3 data, whenever a fill is in compliance with the model required or not: the fill must comprehend a levelled luminosity phase (in which luminosity is kept constant) and a subsequent luminosity decay part. 
The analysis is based on luminosity time-series, extracted from the RUN 3 dataset. The algorithm, as a matter of fact, allows to automate a process that, instead, may be carried out manually. 290 fills have been analysed and compose the dataset used for the training of the algorithm. 

## Structure

project/
├── data/
│   ├── raw/           # Original raw data (ig. .pkl files of the tie series, manually generated label)
│   └──interim/        # Intermediate data, cleaned and ready to be processed (ig. all_fills_time_series.pkl)
│   
├── src/
│   ├── utils.py       # Functions for pre-processing, feature extraction and label loading
│   ├── train.py       # Main script for the training and initial evaluation of the model 
│   ├── evaluate.py    # Script for in depth evaluation of the model on the test set 
│   └── predict.py     # Script for single fill predictions (inference)
|
├── models/            # Machine Learning algorithms trained and saved
├── reports/           # Evaluation reports (ig. plots, metrics)
└── README.md          # This file


## Libraries 
To execute the project, the following Python libraries are needed

* **numpy**
* **pandas**
* **scikit-learn**
* **scipy**
* **matplotlib**
* **seaborn**
* **joblib**


## Workflow and execution instruction

To run the project, follow the steps here reported
* **Make sure to execute all commands from the project's root directory**

### Initial data preparation:



1.1) **Raw Data (.pkl):** Place the `.pkl` file containing the raw fill time series in the `data/raw/` folder. (e.g.,`path/to/project/data/raw/original_data.pkl`). These data won't be loaded on the GitHub repository, as they are too large. Only files comprehending the necessary data would be included
1.2) **Manual Labels:** Create the `manual_fill_labels.csv` file in the `data/raw/` folder with the manual fill labels. The file must have `Fill_ID` and `Is_Valid` columns (0 for not valid, 1 for valid) and use a comma (`,`) as a separator.

```csv
Fill_ID,Is_Valid
8120,0
8121,0
8124,1
...
```

### Pre-processing and Cleaned Time Series Creation

Run the script (or the function from a Jupyter Notebook) responsible for cleaning the raw data and generating the `all_fills_time_series.pkl` file in the `data/interim/` folder (already loaded in the repository). This step is typically handled by a function in `src/utils.py`. This step aims to reduce the quantity of information inside the original .pkl files, so that the algorithm will handle only the luminosity values and the asoociated time values. 

```bash
# Example if there is a run_preprocessing function in utils.py
python -c "from src.utils import load_and_extract_raw_time_series; load_and_extract_raw_time_series('data/raw/your_original_file.pkl')"

```

### Model training

The Random Forest model was chosen for its robust performance and interpretability in classifying fill validity. It handles high-dimensional feature spaces and non-linear relationships, while providing insights into feature importance. This balance of accuracy and explainability makes it ideal for characterizing LHC fills.

This script loads cleaned data, performs the features extraction, divides them in training set and test set, trains the Random Forest model and saves it. An initial evaluation of the performances and of the features' relative importance is shown. 

```bash
python src/train.py
```

The trained model would be saved as `random_forest_model.joblib` in the directory `models/`.

### Evaluation of the model

This script loads the trained model and performs a deep evaluation on the test set, generating a complete classification report and a confusion matrix saved in the directory `reports/`.

```bash
python src/evaluate.py
```

### Prediction of new fills

In order to use te trained model to perform predictions on single fills, execute the script `predict.py`. The ID of the fill can be modified in order to make predictions in the script itself. 

```bash
python src/predict.py
```


## Results and performance of the model implemented


The Random Forest Model trained has shown great performances on the test set. The confusion matrix extracted turns out to be 
[[30  0]
[ 0 28]]

showing that:
* All the 30 fills **Not valid** have been correctly classified  as True Negative.
* All the 28 filla **Valid** have been correctly classified as True Positive.

The model reached accuracy, precision and recall of **100%** on the test set, benchmarking the efficiency of the extracted feature in the discrimination process between the two classes of fills. 

## Final considerations


Despite the good performance of the model implemented, the project may be furtherly improved and validated.

* **Generation of simulated data by MonteCarlo Method:** To increase the robustness and validity, a mechanism of synthetic data generation may be implemented. This would generate a virtually infinite test dataset, exploring also scenarios not widely represented in the real data. The workflow would include the analysis of the existing feature distribution, the statistical modelling and the extraction of new values according to the desired classes.
* **Hyperparameters optimisation:** Using Grid Search methods or Random Search to explore different selections of hyperparameters of the Random Forest (or other models) in order to maximise the performances and avoid the overfitting on wider datasets.
* **Validaion with new real data:** Test the model with a datset of fills completely new to evaluate the generalizability in the real world. 
* **Integration with a robust Pipeline:** Incorporate pre-elaboration process, feature engineering and modelisation in a Scikit-learn pipeline for a cleaner and more coherent data flow.

