import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataPreprocessStrategy, DataDivideStrategy
from typing_extensions import Annotated
from typing import Tuple



@step
def clean_df(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
    ]:
    """
    cleans the data and divides it into train and test data.
    Args:
        data (pd.DataFrame): Data to be cleaned and divided.
    Returns:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
    """
    try:
        process_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data, process_strategy)
        preprocessed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        return X_train, X_test, y_train, y_test
        logging.info("Data divided successfully.")

    except Exception as e:
        logging.error(f"Error in data cleaning step: {e}")
        raise e
    