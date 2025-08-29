# Standard library imports
import logging

# Third-party imports
import pandas as pd
import yaml
from sklearn.utils._testing import ignore_warnings

# Local application/library specific imports
from src.data_preparationCapstone import DataPreparation
from src.model_trainingCapstone import ModelTraining

logging.basicConfig(level=logging.INFO)

def main():

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Configuration file path
    config_path = "./root/src/config_capstone.yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Load CSV file into a DataFrame
    df = pd.read_csv(config["file_pathCapstone"])

    # Initialize and run DataPreparation
    data_prep = DataPreparation(config["data_preparationCapstone"])
    df_cleaned = data_prep.clean_data(df)

    # Initialize ModelTraining with the created preprocessor
    model_training = ModelTraining(config, data_prep.preprocessor)

    # Split the data into training, validation, and test sets
    x_train, x_val, x_test, y_train, y_val, y_test = model_training.split_data(df_cleaned)

    # Train and evaluate models with default hyperparameters
    models = {}
    metrics = {}

    # Train and evaluate tuned models with hyperparameters tuning
    tuned_models = {}
    tuned_metrics = {}

    # Train and evaluate final models with final hyperparameters
    final_models = {}
    final_metrics= {}

    # Combine all models and their metrics into dictionaries
    all_models = {
        "models": models,
        "tuned models": tuned_models,
        "final models": final_models
    }
    all_metrics = {
        "metrics": metrics,
        "tuned metrics": tuned_metrics,
        "final metrics": final_metrics
    }

    # Find the best model based on R-Squared score
    best_model_name = max(all_metrics, key=lambda k: all_metrics[k].get("r2_score", float("-inf")))
    best_model = all_models[best_model_name]
    logging.info(f"Best Model Found: {best_model_name}")

    # Evaluate the best model on the test set
    evaluate_simple = getattr(model_training, "evaluate", None)
    evaluate_final = getattr(model_training, "evaluate_final_model", None)

    if callable(evaluate_final):
        final_metrics = evaluate_final(best_model, x_test, y_test, best_model_name)
    elif callable(evaluate_simple):
        final_metrics = evaluate_simple(best_model, x_test, y_test)
    else:
        final_metrics = {}
        logging.warning("No evaluation method found on ModelTraining; final metrics set to empty dict.")

if __name__ == "__main__":

    main()
