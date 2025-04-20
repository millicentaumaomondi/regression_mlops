# from zenml.config import BaseSettings
from pydantic import BaseModel

class ModelNameConfig(BaseModel):
# class ModelNameConfig(BaseSettings):
    """Configuration for the model name."""
    model_name: str = "linear_regression"  # Default model name
    model_path: str = "models/linear_regression_model.pkl"  # Default model path
    model_type: str = "regression"  # Default model type
    test_size: float = 0.2  # Default test size for train-test split
    random_state: int = 42  # Default random state for reproducibility