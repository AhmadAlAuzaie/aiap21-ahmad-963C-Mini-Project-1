# Standart library imports
import logging
from typing import Any, Dict, Tuple, cast

# Related third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
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
        x = df.drop(columns=self.config["target_columnAssign"])
        y = df[self.config["target_columnAssign"]]
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
        df_train = pd.concat([x_train, y_train], axis=1)
        # Resample the training data if specified in the config
        if self.config.get("resample", False):
            df_train = df_train.sample(frac=self.config["resample_fraction"], random_state=45)
            x_train = df_train.drop(columns=self.config["target_columnAssign"])
            y_train = df_train[self.config["target_columnAssign"]]
        
        # Separate majority and minority classes in the training set only
        train_majority = df_train[df_train[self.config["target_columnAssign"]] == 'Below Median']
        train_minority = df_train[df_train[self.config["target_columnAssign"]] == 'Above Median']

        # Upsample the minority class to match the majority class size
        train_minority_upsampled = resample(
            train_minority,
            replace=True,  # Sample with replacement
            n_samples=len(train_majority),  # Match majority class size
            random_state=45  # Reproducibility
        )

        # Downsample the minority class to match the majority class size
        train_majority_downsampled = resample(
            train_majority,
            replace=False,  # Sample without replacement
            n_samples=len(train_minority),  # Match minority class size
            random_state=45  # Reproducibility
        )

        # Combine downsample majority class with upsampled minority class (ensure result is a DataFrame)
        # Ensure any earlier resample results are proper DataFrame/Series objects for pandas.concat
        train_majority_downsampled = pd.DataFrame(train_majority_downsampled)
        train_minority_upsampled = pd.DataFrame(train_minority_upsampled)

        train_combined = pd.concat(
            [train_majority_downsampled, train_minority_upsampled], 
            ignore_index=True
        )

        # Re-separate the features and target in the combined training set 
        x_train_combined = train_combined.drop(columns=self.config["target_columnAssign"])
        y_train_combined = train_combined[self.config["target_columnAssign"]]

        # Apply SMOTE to the training set using numpy arrays to satisfy type stubs
        smote = SMOTE(random_state=45)
        x_train_resampled, y_train_resampled = cast(
            Tuple[np.ndarray, np.ndarray],
            smote.fit_resample(
                x_train_combined.to_numpy(), y_train_combined.to_numpy()
            )
        )

        # Combine resampled features and target into a new DataFrame (preserve column names)
        df_x_resampled = pd.DataFrame(x_train_resampled, columns=x_train_combined.columns)
        df_y_resampled = pd.Series(y_train_resampled, name=self.config["target_columnAssign"])
        df_train_resampled = pd.concat([df_x_resampled, df_y_resampled], axis=1)

        # Return only the canonical six train/val/test splits as declared in the function signature
        return (x_train, x_val, x_test, y_train, y_val, y_test)
    
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
            "ridge_classification": Ridge(),
            "decision_tree_classifier": DecisionTreeClassifier(),
            "random_forest_classifer": RandomForestClassifier(),
            "KNeighbors Classifier": KNeighborsClassifier(),
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
            disp.plot(cmap='Oranges', ax=ax, colorbar=False, values_format='d')
            plt.grid(False)
            logging.info(
                f"{model_name} - Accuracy Score: {metrics[model_name]['accuracy']}, "
                f"Precision Score: {metrics[model_name]['precision']}, "
                f"Recall Score: {metrics[model_name]['recall']}, "
                f"F1 Score: {metrics[model_name]['f1']}; "
                f"Confusion Matrix: {metrics[model_name]['cm']}"
            )

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

            # Create and fit the decision tree pipeline to the training data
            dt_pipeline = Pipeline(steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", DecisionTreeClassifier(criterion='gini', min_samples_split=10, 
                                                      max_depth=5, random_state=45))
            ])
            dt_pipeline.fit(x_train, y_train)

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
            x_train_tuned: pd.DataFrame,
            y_train_tuned: pd.Series,
            x_val_tuned: pd.DataFrame,
            y_val_tuned: pd.Series
        ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Creates, trains and evaluate tuned models using GridSearchCV.

        Args:
        -----
        x_train_tuned (pd.DataFrame): Training features.
        y_train_tuned (pd.Series): Training target variable.
        x_val_tuned (pd.DataFrame): Validation features.
        y_val_tuned (pd.Series): Validation target variable.
        
        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
            A tuple containing the trained pipelines and their evaluation metrics.
        """
        logging.info("Training and evaluating tuned models.")
        tuned_models = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "ridge_classification": Ridge(),
            "decision_tree_classifier": DecisionTreeClassifier(),
            "random_forest_classifer": RandomForestClassifier(),
            "KNeighbors Classifier": KNeighborsClassifier(),
            "svc": SVC(probability=True)
        }
        tuned_metrics = {}
        tuned_pipelines = {}
        for tuned_model_name, model in tuned_models.items():
            pipeline = Pipeline(steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", model)
            ])
            param_grid = self.config.get("param_grid", {}).get(tuned_model_name, {})
            if param_grid:
                grid_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    scoring='f1',
                    cv=5,
                    n_jobs=-1
                )
                grid_search.fit(x_train_tuned, y_train_tuned)
                best_pipeline = grid_search.best_estimator_
                tuned_pipelines[tuned_model_name] = best_pipeline
                # Predict on the validation set in the decision tree pipeline
                y_pred = best_pipeline.predict(x_val_tuned)
                # Get the predicted probabilities for the positive class
                y_val_pred_proba = best_pipeline.predict_proba(x_val_tuned)[:, 1]
                
                tuned_metrics[tuned_model_name] = {
                    "accuracy": accuracy_score(y_val_tuned, y_pred),
                    "precision": precision_score(y_val_tuned, y_pred),
                    "recall": recall_score(y_val_tuned, y_pred),
                    "f1": f1_score(y_val_tuned, y_pred), 
                    "cm": confusion_matrix(y_val_tuned, y_pred)
                }
                disp = ConfusionMatrixDisplay(confusion_matrix=tuned_metrics[tuned_model_name]["cm"])
                fig, ax = plt.subplots(figsize=(8, 6))
                disp.plot(cmap='Oranges', ax=ax, colorbar=False, values_format='d')
                plt.grid(False)
                logging.info(
                    f"{tuned_model_name} - Accuracy Score: {tuned_metrics[tuned_model_name]['accuracy']}, "
                    f"Precision Score: {tuned_metrics[tuned_model_name]['precision']}, "
                    f"Recall Score: {tuned_metrics[tuned_model_name]['recall']}, "
                    f"F1 Score: {tuned_metrics[tuned_model_name]['f1']}; "
                    f"Confusion Matrix: {tuned_metrics[tuned_model_name]['cm']}"
                )

                # Calculate precision, recall and thresholds
                precision, recall, thresholds = precision_recall_curve(y_val_tuned, y_val_pred_proba)

                # Calculate the area under the precision-recall curve
                pr_auc = auc(recall, precision)

                # Plot the precision-recall curve
                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, color='blue', marker='.', 
                         label=f'{tuned_model_name} (PR AUC = {pr_auc:.2f})')
                plt.plot([0, 1], [np.mean(y_val_tuned), np.mean(y_val_tuned)], color='red', 
                         linestyle='--', label='Random Classifier')
                # Set plot limits and title
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve for {tuned_model_name}')
                plt.legend(loc='lower left')
                plt.grid(True)
                plt.savefig(f'precision_recall_curve_{tuned_model_name}_tuned.png')
                plt.show()
            else:
                logging.warning(f"No parameter grid found for {tuned_model_name}. Skipping GridSearchCV.")
                tuned_pipelines[tuned_model_name] = pipeline
                pipeline.fit(x_train_tuned, y_train_tuned)
        return tuned_pipelines, tuned_metrics
    
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
        val_y_pred = model.predict(x_val)
        train_y_pred = model.predict(x_train)
        metrics = {
            "val accuracy": accuracy_score(y_val, val_y_pred),
            "val precision": precision_score(y_val, val_y_pred),
            "val recall": recall_score(y_val, val_y_pred),
            "val f1": f1_score(y_val, val_y_pred), 
            "cm": confusion_matrix(y_val, val_y_pred),
            
            "tain accuracy": accuracy_score(y_train, train_y_pred),
            "train precision": precision_score(y_train, train_y_pred),
            "train recall": recall_score(y_train, train_y_pred),
            "train f1": f1_score(y_train, train_y_pred)
        }
        logging.info(f"{model_name} Validation Metrics:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name.upper()}: {metric_value:.4f}")
        return metrics