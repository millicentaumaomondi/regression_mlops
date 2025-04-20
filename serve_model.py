from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# ✅ Load the model
model = joblib.load("model/linear_model.pkl")

# ✅ Initialize FastAPI app
app = FastAPI(title="Linear Regression API")

# ✅ Define request schema
class Features(BaseModel):
    features: list[float]

# ✅ Define prediction endpoint
@app.post("/predict")
def predict(data: Features):
    # Reshape to match model input: (1, n_features)
    input_data = np.array(data.features).reshape(1, -1)
    prediction = model.predict(input_data).tolist()
    return {"prediction": prediction}
