import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
import pandas as pd
from typing import Union
# from zenml import step

class Model(ABC):
    """
    Abstract Class defining model
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model on the provided data.
        Args:
            data (pd.DataFrame): The data to train the model on.
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression model
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Train the Linear Regression model on the provided data.
        Args:
            X_train (pd.DataFrame): The training features.
            y_train (pd.Series): The training labels.
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Linear Regression model trained successfully.")
            return reg
        except Exception as e:
            logging.error(f"Error training Linear Regression model: {e}")
            raise e
       