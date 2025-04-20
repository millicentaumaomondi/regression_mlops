import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score #root_mean_squared_error

class Evaluation(ABC):
    """
    Abstract Class defining evaluation
    """

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Evaluate the model on the provided data.
        Args:
            data (pd.DataFrame): The data to evaluate the model on.
        """
        pass
class MSE(Evaluation):
    """
    Mean Squared Error evaluation
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error (MSE) between true and predicted values.
        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
        Returns:
            float: MSE score.
        """
        try:
            logging
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"Mean Squared Error: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error calculating MSE: {e}")
            raise e
    
class R2(Evaluation):
    """
    R-squared evaluation
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the R-squared score between true and predicted values.
        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
        Returns:
            float: R-squared score.
        """
        try:
            # Calculate R-squared score
            r2 = r2_score(y_true, y_pred)  # Corrected the function call for R-squared score
            logging.info(f"R-squared score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error calculating R-squared score: {e}")
            raise 
