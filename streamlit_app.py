import streamlit as st
import requests

st.set_page_config(page_title="ML Model Predictor", layout="centered")

st.title("📊 Linear Regression Predictor")
st.markdown("Enter feature values to get a prediction from your trained model.")

# Number of features — update this to match your model!
num_features = 12  # change this to match your model's feature count

inputs = []
for i in range(num_features):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(val)

if st.button("Predict"):
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"features": inputs}
    )
    if response.status_code == 200:
        prediction = response.json()["prediction"][0]  # Get the first value
        st.success(f"🔮 Prediction: {prediction}")
    else:
        st.error("❌ Error from prediction API.")
