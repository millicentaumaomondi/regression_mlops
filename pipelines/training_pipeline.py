from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from src.config import ModelNameConfig  # Make sure ModelNameConfig is imported

@pipeline(enable_cache=True)
def train_pipeline(data_path: str, model_name: str):
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(data=df)

    # pass through the CLI/model_name argument, not a hard‑coded string
    config = ModelNameConfig(model_name=model_name)

    model = train_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        config=config,
    )

    # Return the step outputs – ZenML will capture these as artifacts
    mse, r2 = evaluate_model(model=model, X_test=X_test, y_test=y_test)
    return mse, r2

