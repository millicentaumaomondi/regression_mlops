
# 🤖 Regression ML Model Deployment with Streamlit + Hugging Face

This repository contains the full pipeline for training, saving, and deploying a `LinearRegression` model using **scikit-learn**, **ZenML**, **Streamlit**, and **Hugging Face Spaces**.

---

## 📦 Project Structure

```bash
├── data/                   # Raw data (CSV, etc.)
├── model/                  # Trained model (packaged as .zip)
├── pipelines/             # ZenML training pipeline
├── src/                   # Configs, utilities, cleaning logic
├── streamlit_app.py       # UI app for predictions
├── requirements.txt       # Dependency list
└── README.md              # You're here!
'''


 #Demo
👉 [Try the deployed app on Hugging Face:]
(https://huggingface.co/spaces/momondi/regression_mlops)

 ML Pipeline (ZenML)
Ingestion: Load dataset

Cleaning: Handle missing values, feature selection

Training: Train LinearRegression model

Evaluation: Calculate MSE and R² score

Deployment: Package and upload model for use in app

# Model Info
Algorithm: LinearRegression (sklearn)

Metrics:

✅ Mean Squared Error (MSE)

✅ R² Score

# How to Run Locally
# Step 1: Clone repo
git clone https://github.com/millicentaumaomondi/regression_mlops.git
cd regression_mlops

# Step 2: Create environment and install dependencies
pip install -r requirements.txt

# Step 3: Launch the app
streamlit run streamlit_app.py

# Acknowledgements
Built with:

Streamlit

Hugging Face Spaces

ZenML

scikit-learn

