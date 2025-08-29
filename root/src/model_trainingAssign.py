# Standart library imports
import logging
from typing import Any, Dict, Tuple

# Related third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve, auc


class ModelTraining:
    """
    A class used to train and evaluate machine learning models on HDB resale prices data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for model training and evaluation.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal and ordinal features.
    """

    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer):
        """
        Initialises the ModelTraining class with a configuration dictionary and preprocessor.
        
        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for model training and evaluation.
        preprocessor (sklearn.compose.ColumnTransformer):
            A preprocessor pipeline for transforming numerical, nominal and ordinal features.
        """
        self.config = config
        self.preprocessor = preprocessor

    def split_data(
            self, df: pd.DataFrame,
        ) -> Tuple[
            pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
        ]:
        """
        Splits the DataFrame into features and target variable.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the data.
        target (str): The name of the target variable column.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: 
        A tuple containing the training, validation, and test features and target variables.
        """
        logging.info("Starting data splitting.")
        x = df.drop(columns=self.config["target_column1"])
        y = df[self.config["target_column1"]]
        x_train, x_temp, y_train, y_temp = train_test_split(
            x, y, test_size=self.config["val_test_size"], 
            random_state=45,
            stratify=y if self.config.get("stratify", False) else None
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp, y_temp, test_size=self.config["val_size"], 
            random_state=45,
            stratify=y_temp if self.config.get("stratify", False) else None
        )
        # Combine training features and target for resampling
        train_df = pd.concat([x_train, y_train], axis=1)
        # Resample the training data if specified in the config
        if self.config.get("resample", False):
            train_df = train_df.sample(frac=self.config["resample_fraction"], random_state=45)
            x_train = train_df.drop(columns=self.config["target_column1"])
            y_train = train_df[self.config["target_column1"]]
        
        # Separate majority and minority classes in the training set only
        train_majority = train_df[train_df[self.config["target_column1"]] == 'Below Median']
        train_minority = train_df[train_df[self.config["target_column1"]] == 'Above Median']

        # Upsample the minority class to match the majority class size
        train_minority_upsampled = resample(
            train_minority,
            replace=True,  # Sample with replacement
            n_samples=len(train_majority_downsampled),  # Match downsampled majority class size
            random_state=45  # Reproducibility
        )
        # Downsample the majority class to match the minority class size
        train_majority_downsampled = resample(
            train_majority,
            replace=False,  # Sample without replacement
            n_samples=len(train_minority_upsampled),  # Match upsampled minority class size
            random_state=45  # Reproducibility
        )
        # Combine upsampled minority class with downsampled majority class
        train_combined = pd.concat([train_majority_downsampled, train_minority_upsampled])

        # Re-separate the features and target in the combined training set 
        x_train_combined = train_combined.drop(columns=self.config["target_column1"])
        y_train_combined = train_combined[self.config["target_column1"]]

        # Apply SMOTE to the training set
        smote = SMOTE(random_state=45)
        x_train_resampled, y_train_resampled = smote.fit_resample(x_train_combined, y_train_combined)

        # Combine resampled features and target into a new DataFrame
        df_train_resampled = pd.concat([pd.DataFrame(x_train_resampled), 
                                        pd.DataFrame(y_train_resampled, 
                                        columns =['target_column1'])], axis=1)

        return (x_train, x_val, x_test, y_train, y_val, y_test, 
                x_train_combined, y_train_combined, 
                x_train_resampled, y_train_resampled, df_train_resampled)
    
    def train_and_evaluate_model(
            self, 
            x_train: pd.DataFrame,
            y_train: pd.Series,
            x_val: pd.DataFrame,
            y_val: pd.Series
        ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Creates, trains and evaluate baseline models.

        Args:
        -----
        x_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        x_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target variable.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
            A tuple containing the trained pipelines and their evaluation metrics.
        """
        logging.info("Training and evaluating baseline models.")
        models = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "ridge_regression": Ridge(),
            "decision_tree_classifier": DecisionTreeClassifier(),
            "random_forest_regression": RandomForestClassifier(),
            "svc": SVC(probability=True)
        }
        pipelines = {}
        metrics = {}
        for model_name, model in models.items():
            pipeline = Pipeline(steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", model)
            ])
            pipeline.fit(x_train, y_train)
            pipelines[model_name] = pipeline
            # Predict on the validation set in the decision tree pipeline
            y_pred = pipeline.predict(x_val)
            # Get the predicted probabilities for the positive class
            y_val_pred_proba = pipeline.predict_proba(x_val)[:, 1]
            
            y_pred = pipeline.predict(x_val)
            metrics[model_name] = {
                "accuracy": accuracy_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred),
                "recall": recall_score(y_val, y_pred),
                "f1": f1_score(y_val, y_pred), 
                "cm": confusion_matrix(y_val, y_pred)
            }
            disp = ConfusionMatrixDisplay(confusion_matrix=metrics[model_name]["cm"])
            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(cmap=plt.cm.Oranges, ax=ax, colorbar=False, values_format='d')
            plt.grid(False)
            logging.info(f"{model_name} - Accuracy Score: {metrics[model_name]['ac score']}, "
                         f"Precision Score: {metrics[model_name]['pr score']}, "
                         f"Recall Score: {metrics[model_name]['r score']}, "
                         f"F1 Score: {metrics[model_name]['f1 score']}",
                         f"Confusion Matrix: {metrics[model_name]['cm']}")

            # Calculate precision, recall and thresholds
            precision, recall, thresholds = precision_recall_curve(y_val, y_val_pred_proba)

            # Calculate the area under the precision-recall curve
            pr_auc = auc(recall, precision)

            # Plot the precision-recall curve
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', marker='.', 
                     label=f'{model_name} (PR AUC = {pr_auc:.2f})')
            
            plt.plot([0, 1], [np.mean(y_val), np.mean(y_val)], color='red', 
                     linestyle='--', label='Random Classifier')
            
            # Set plot limits and title
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve for {model_name}')
            plt.legend(loc='lower left')
            plt.grid(True)
            plt.savefig(f'precision_recall_curve_{model_name}.png')
            plt.show()

            # Fit the pipeline to the training data
            dt_pipeline.fix(x_train, y_train)
            # Calculate metrics for the training set in the decision tree pipeline
            y_pred = dt_pipeline.predict(x_train)
            y_train_pred_proba = dt_pipeline.predict_proba(x_train)[:, 1]
            # Create a decision tree pipeline
            dt_pipeline = Pipeline(steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", DecisionTreeClassifier(criterion='gini', min_samples_split=10, 
                                                      max_depth=5, random_state=45))
            ])
            metrics[model_name] = {
                "accuracy": accuracy_score(y_train, y_pred),
                "precision": precision_score(y_train, y_pred),
                "recall": recall_score(y_train, y_pred),
                "f1": f1_score(y_train, y_pred),
                "cm": confusion_matrix(y_train, y_pred)
            }
            
        return pipelines, metrics
    
    def train_and_evaluate_tuned_models(
            self, 
            x_train: pd.DataFrame,
            y_train: pd.Series,
            x_val: pd.DataFrame,
            y_val: pd.Series
        ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Performs hyperparameter tuning for Lasso and Ridge models and evaluates them.

        Args:
        -----
        x_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        x_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target variable.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
            A tuple containing the tuned pipelines and their evaluation metrics.
        """
        logging.info("Starting hyperparameter tuning.")
        tuned_models = {}
        tuned_metrics = {}
        param_grid = self.config["param_grid"]
        cv = self.config["cv"]
        scoring = self.config["scoring"]

        models = {"ridge_tuned": Ridge(), "logistic_tuned": LogisticRegression()}

        for model_name, model in models.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", self.preprocessor),
                    ("classifier", model)
                ]
            )
            grid_search = GridSearchCV(
                pipeline, 
                param_grid, 
                cv=cv, 
                scoring=scoring, 
                n_jobs=-1
            )
            grid_search.fit(x_train, y_train)
            tuned_models[model_name] = grid_search.best_estimator_
            tuned_metrics[model_name] = self.evaluate_model(
                tuned_models[model_name], x_val, y_val,
                x_val, y_val, model_name + " (tuned)"
            )

        # Get the predicted probabilities for the tuned positive class
        y_val_pred_proba = tuned_models[model_name].predict_proba(x_val)[:, 1]

        # Calculate precision, recall and thresholds
        tuned_precision, tuned_recall, tuned_thresholds = precision_recall_curve(y_val, y_val_pred_proba)
        
        # Calculate the area under the precision-recall curve
        tuned_pr_auc = auc(tuned_recall, tuned_precision)

        # Plot the precision-recall curve for the tuned model
        plt.figure(figsize=(8, 6))
        plt.plot(tuned_recall, tuned_precision, color='blue', marker='.', 
                 label=f'{model_name} (Tuned PR AUC = {tuned_pr_auc:.2f})')
        plt.plot([0, 1], [np.mean(y_val), np.mean(y_val)], color='red', 
                 linestyle='--', label='Random Classifier')
        
        # Set plot limits and title
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {model_name} (Tuned)')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig(f'precision_recall_curve_{model_name}_tuned.png')
        plt.show()
        return tuned_models, tuned_metrics
    
    def _evaluate_model(
            self, 
            model: Pipeline, 
            x_val: pd.DataFrame, 
            y_val: pd.Series,
            x_train: pd.DataFrame, 
            y_train: pd.Series,
            model_name: str
    ) -> Dict[str, float]:
        """
        Evaluates the model on the validation set and logs the metrics.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        x_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target variable.
        model_name (str): Name of the model for logging.

        Returns:
        --------
        Dict[str, float]: A dictionary containing evaluation metrics.
        """
        y_pred = model.predict(x_val)
        metrics = {
            "val accuracy": accuracy_score(y_val, y_pred),
            "val precision": precision_score(y_val, y_pred),
            "val recall": recall_score(y_val, y_pred),
            "val f1": f1_score(y_val, y_pred), 
            "cm": confusion_matrix(y_val, y_pred),
            
            "tain accuracy": accuracy_score(y_train, y_pred),
            "train precision": precision_score(y_train, y_pred),
            "train recall": recall_score(y_train, y_pred),
            "train f1": f1_score(y_train, y_pred)
        }
        logging.info(f"{model_name} Validation Metrics:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name.upper()}: {metric_value:.4f}")