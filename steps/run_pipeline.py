# from pipelines.training_pipeline import train_pipeline
# from zenml.client import Client

# # 1. Run the pipeline (runs immediately — no need for .run())
# run = train_pipeline(data_path="/Users/millicentomondi/Documents/ammi_project_past_papers/practice_app/recognition/data/olist_customers_dataset.csv", model_name="LinearRegression")  # run is already PipelineRunResponse

# # 2. Get outputs safely — list all available outputs first
# print("Available steps:", run.steps.keys())
# print("Evaluate step outputs:", run.steps["evaluate_model"].outputs.keys())

# # 3. Try loading with fallback names
# mse = run.steps["evaluate_model"].outputs["MSE"][0]
# r2 = run.steps["evaluate_model"].outputs["R2"][0]

# if mse and r2:
#     print(f"✅ Mean Squared Error: {mse}")
#     print(f"✅ R² Score: {r2}")

# else:
#     print("❌ Could not find output artifacts. Try disabling cache next.")
