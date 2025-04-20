import logging
import os
import joblib
import pandas as pd
from zenml import step
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin
from src.config import ModelNameConfig

# ✅ MLflow import (optional)
import mlflow
import mlflow.sklearn

@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig
) -> RegressorMixin:
    """Train a machine learning model and save it locally.

    Args:
        X_train (pd.DataFrame): The training features.
        X_test (pd.DataFrame): The testing features.
        y_train (pd.Series): The training labels.
        y_test (pd.Series): The testing labels.
        config (ModelNameConfig): Configuration object specifying the model.
        
    Returns:
        model: The trained model.
    """
    try:
        model = None

        if config.model_name == "LinearRegression":
            model = LinearRegression()
            model.fit(X_train, y_train)
        else:
            raise ValueError(f"Model {config.model_name} not supported.")

        logging.info(f"Model {config.model_name} trained successfully.")

        # ✅ Save model locally
        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/linear_model.pkl")
        logging.info("Model saved to model/linear_model.pkl")

        # ✅ Log model to MLflow (optional)
        with mlflow.start_run(nested=True):
            mlflow.sklearn.log_model(model, "linear_model")
            logging.info("Model logged to MLflow.")

        return model

    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e
