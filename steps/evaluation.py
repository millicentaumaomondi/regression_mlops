from zenml import step
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
import pandas as pd
import logging

from src.evaluation import MSE, R2

# ✅ NEW
import mlflow

@step
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> Tuple[Annotated[float, "MSE"], Annotated[float, "R2"]]:

    try:
        predictions = model.predict(X_test)

        mse = MSE().calculate_scores(y_test, predictions)
        r2 = R2().calculate_scores(y_test, predictions)

        logging.info(f"MSE: {mse}")
        logging.info(f"R2: {r2}")

        # ✅ NEW: Log metrics to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("R2", r2)

        return mse, r2

    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise
