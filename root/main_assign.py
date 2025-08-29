# Standard library imports
import logging

# Third-party imports
import pandas as pd
import yaml
from sklearn.utils._testing import ignore_warnings

# Local application/library specific imports
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

logging.basicConfig(level=logging.INFO)

@ignore_warnings(category=FutureWarning)
def main():

    # Configuration file path
    config_path = "./root/src/config_assign.yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Load CSV file into a DataFrame
    df = pd.read_csv(config["file_pathAssign"])

    # Initialize and run DataPreparation
    data_prep = DataPreparation(config["data_preparationAssign"])
    df_cleaned = data_prep.clean_data(df)

    # Initialize ModelTraining with the created preprocessor
    model_training = ModelTraining(config, data_prep.preprocessor)

    # Split the data into training, validation, and test sets
    x_train, x_val, x_test, y_train, y_val, y_test = model_training.split_data(df_cleaned)

    # Train and evaluate baseline models with default hyperparameters
    baseline_models, baseline_metrics = (
        model_training.train_and_evaluate_baseline_models(
            x_train, y_train, x_val, y_val
        )
    )

    # Train and evaluate tuned models with hyperparameter tuning
    tuned_models, tuned_metrics = model_training.train_and_evaluate_tuned_models(
        x_train, y_train, x_val, y_val
    )

    # Combine all models and their metrics into dictionaries
    all_models = {
        "baseline": baseline_models,
        "tuned": tuned_models,
    }
    all_metrics = {
        "baseline": baseline_metrics,
        "tuned": tuned_metrics,
    }

    # Find the best model based on R-Squared score
    best_model_name = max(all_metrics, key=lambda k: all_metrics[k]["r2_score"])
    best_model = all_models[best_model_name]
    logging.info(f"Best Model Found: {best_model_name}")

    # Evaluate the best model on the test set
    final_metrics = model_training.evaluate_final_model(
        best_model, x_test, y_test, best_model_name
    )

if __name__ == "__main__":
    main()