# AhmadAlAuzaie-Capstone

## Project Description
This project implements a machine learning pipeline to predict the risk of developing chronic diseases based on a patient's medical history and lifestyle factors. This would enable early intervention and improve preventative healthcare. The pipeline includes data cleaning, preprocessing, model training and evaluation. Multiple regression models are trained and compared to find the best performing one based on various metrics.

## Prerequisities and Installation Instructions

### Requirements
- Anaconda or Miniconda
- Python 3.13.3
- Required packages listed in 'requirements.txt'

### Installation
1. Clone the repository:
```
git clone https://github.com/username:personal-access-token@<your-assigned-github-repository>
cd hdb-resale-price-prediction
```

2. Start working on this project and push your code into the main branch:
```
git add .
git commit -m “<your-commit-message>”
git push -u origin
```

3. Create and activate a conda environment:
```
conda create -n hdbenv pythoncon
conda activate hdbenv
```

4. Install and Upgrade the required packages:
```
cd root
conda install --yes --file requirements.txt
conda update --all
python -m pip install --upgrade pip
python -m ensurepip --upgrade
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install -c conda-forge imbalanced-learn
```
or
```
cd root
pip install -U -r requirements.txt
(pip install -r requirements.txt --upgrade)
python -m pip install --upgrade pip
python -m ensurepip --upgrade
pip install -U imbalanced-learn
```

## Pipeline Execution
To run the complete pipeline, execute:
```
python main_resale.py
```

This will:
1. Load and clean the data
2. Preprocess features
3. Split data into training, validation, and test sets
4. Train baseline models
5. Perform hyperparameter tuning
6. Evaluate models and select the best one

## Logical Flow of the Pipeline

### 1. Configuration (`config.yaml`)
The pipeline starts with loading configuration parameters from `src/config.yaml`, which includes:
- Data file path
- Target column name
- Feature categorization (numerical, nominal, ordinal)
- Train/Validation/Test split ratios
- Hyperparameter grid for model tuning

### 2. Data Preparation (`DataPreparation` class)
The `Datapreparation` class handles:
- Removing duplicates
- Standardizing flat type names
- Converting storey ranges to numerical values
- Filling missing town and flat model names
- Extracing year and month from date
- Converting remaining lense information to months
- Creating a preprocessor for feature transformaton

### 3. Model Training (`ModelTrainig` class)
The `ModelTraining` class manages:
- Splitting data into training, validation, and test sets
- Training baseline models (Linear Regression, Ridge, Lasso)
- Hyperparameter tuning for Ridge and Lasso models
- Evaluating models using multiple metrics (MAE, MSE, RMSE, R²)
- Selecting the best model based on R² score

### 4. Main Execution (`main_resale.py`)
The main script orchestrates and data
- Loading configuration and data
- Initializing data preparation and model training
- Running the training and evaluation process
- Identifying and evaluating the best model on the test set

## Key Findings and Feature Handling

### Exploratory Data Analysis

Before diving into the model training process, an exploratory data analysis (EDA) was conducted to understand the data better and identify any potential issues. Here are some key findings for the EDA:

**EDA Findings and Explanations**: ...

### Data Cleaning
- Storey ranges are converted to their average values because ...
- Remaining lease information is extracted and converted to total months because ...
- Year and Month are extracted from the transaction date because ...
- Missing town and flat model names are filled using ID-to-name mappings because ...

### Feature Processing
- **Numerical Features**: `floor_area_sqm`, `remaining_lease_months`, `lease_commence_date`, `year` - Standized using `StandardScaler` because ...
- **Nominal Features**: `month`, `town_name`, `flatm_name` - Encoded using `OrdinalEncoder` with predefined categories because ...
- **Passthrough Features**: `storey_range` - Used as-is after converting to numerical values because ...

## Model Choices and Evaluation

### Models Implemented and Justifications
1. **Linear Regression**: Basic model without regularization because ...
2. **Ridge Regression**: Linear regression with L2 regularization because ...
3. **Lasso Regression**: Linear regression with L1 regularization because ...

### Hyperparameter Tuning
- Grid Search is performed for Ridge and Lasso models because ...
- Parameters tuned include `alpha` and `fit_intercept` because ...
- 5-fold cross-validation is used during tuning because ...

### Evaluation Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual prices
- **MSE (Mean Squared Error)**: Average squared difference between predicted and actual prices
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in the same unit as the target
- **R² (Co-efficient of Determination)**: Proportion of variance explained by the model

## Deployment Considerations

### Model Persistence
To make this project production-ready, implement model persistence:

1. Add model saving functionality to `ModelTraining` class:
    ```python
    import joblib

    def save_model(self, model, filename):
        """Save the trained model to a file."""
        joblib.dump(model, filename)

    def load_model(self, model, filename):
        """Save the trained model to a file."""
        return joblib.load(model, filename)
    ```

2. Create a separate inference script (`inference.py`)
for making prediction with saved models:
    ```python
    import pandas as pd
    import joblib
    from src.data_preparation import DataPreparation
    ...
    ```
