# Standard library imports
import logging
import re
from typing import Any, Dict

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

class DataPreparation:
    """
    A class used to clean and preprocess HDB predict risk data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical, nominal and ordinal features.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialises the DataPreparation class with a configuration dictionary.
        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for data cleaning and preprocessing.
        """
        self.config = config
        self.preprocessor = self._create_preprocessor()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by performing several preprocessing steps.

        Args:
        -----
        df(pd.DataFrame): The input DataFrame containing the raw data.

        Returns:
        --------
        pd.DataFrame: The cleaned DataFrame.
        """
        logging.info("Starting data cleaning.")
        df.drop_duplicates(inplace=True)
        df["Risk_of_Heart_Disease"].replace("HIGH_RISK", "High Risk", inplace=True)
        return df

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Create a preprocessor pipeline for transforming numerical, nominal and ordinal features.
        
        Returns:
        """
        numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        nominal_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )
        ordinal_transformer = Pipeline(
            steps=[("ordinal", OrdinalEncoder())]
        )
        # Remove the incorrect passthrough transformer that used StandardScaler with invalid params.
        # passthrough features will be passed through unchanged by using "passthrough" below.
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.config["numerical_features"]),
                ("nom", nominal_transformer, self.config["nominal_features"]),
                ("ord", ordinal_transformer, self.config["ordinal_features"]),
                ("pass", "passthrough", self.config["passthrough_features"])
            ],
            remainder="passthrough",
            n_jobs=-1
        )
        return preprocessor

    @staticmethod
    def _convert_Risk_of_Heart_Disease(Risk_of_Heart_Disease: str) -> str:
        """
        Converts a Risk_of_Heart_Disease string into its average numerical value.

        Args:
        -----
        Risk_of_Heart_Disease (str): The Risk_of_Heart_Disease string to convert.

        Returns:
        --------
        str: The converted Risk_of_Heart_Disease value.
        """
        if Risk_of_Heart_Disease == "High Risk":
            return "High Risk"
        elif Risk_of_Heart_Disease == "Low Risk":
            return "Low Risk"
        else:
            return "Unknown Risk"

    @staticmethod
    def _convert_BMI(BMI: Any) -> float:
        """
        Converts a BMI value into a float.

        Args:
        -----
        BMI (int | str | float | None): The BMI value to convert to float positive and negative values.

        Returns:
        --------
        float: The converted BMI value.
        """
        if isinstance(BMI, str):
            BMI = re.sub(r"[^\d.]", "", BMI)
        try:
            return float(BMI) if BMI else 0.0
        except (ValueError, TypeError):
            logging.warning(f"Could not convert BMI value: {BMI}. Returning NaN.")
            return float("nan")

    @staticmethod
    def _convert_Cholesterol_Levels(Cholesterol_Levels: Any) -> float:
        """
        Converts a Cholesterol_Levels value into a float.

        Args:
        -----
        Cholesterol_Levels (int | str | float | None): The Cholesterol_Levels value to convert to float values.

        Returns:
        --------
        float: The converted Cholesterol_Levels value.
        """
        if isinstance(Cholesterol_Levels, str):
            Cholesterol_Levels = re.sub(r"[^\d.]", "", Cholesterol_Levels)
        try:
            return float(Cholesterol_Levels) if Cholesterol_Levels else 0.0
        except (ValueError, TypeError):
            logging.warning(f"Could not convert Cholesterol_Levels value: {Cholesterol_Levels}. Returning NaN.")
            return float("nan")

    @staticmethod
    def _fill_Family_History(
        df: pd.DataFrame, family_history_column: str
    ) -> pd.DataFrame:
        """
        Fills missing values in the 'family_history_column' with 'unknown'.
        
        Args:
        -----
        df (pd.DataFrame): The DataFrame containing the 'family_history_column'.
        family_history_column (str): The name of the column to fill missing values in.
        
        Returns:
        --------
        pd.DataFrame: The DataFrame with missing values in 'family_history_column' filled.
        """
        df[family_history_column].fillna("unknown", inplace=True)
        return df