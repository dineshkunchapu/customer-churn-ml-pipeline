# import streamlit as st
# import pandas as pd
# import joblib
# import mlflow
# import mlflow.sklearn
# import os
# from PIL import Image

# # === PAGE SETUP ===
# st.set_page_config(
#     page_title="Telecom Customer Churn Prediction",
#     page_icon="📊",
#     layout="wide"
# )

# st.title("📞 Telecom Customer Churn Prediction Dashboard")
# st.markdown("### Powered by DVC + MLflow + Streamlit")

# # === LOAD LATEST MODEL FROM MLFLOW ===
# mlflow.set_tracking_uri("file:./mlruns")
# experiment_name = "telecom-churn"
# client = mlflow.tracking.MlflowClient()
# experiment = client.get_experiment_by_name(experiment_name)

# # Get the latest run
# runs = client.search_runs(
#     experiment.experiment_id, order_by=["attributes.start_time DESC"], max_results=1
# )
# if len(runs) == 0:
#     st.warning("⚠️ No MLflow runs found. Please train a model first using DVC.")
#     st.stop()

# latest_run_id = runs[0].info.run_id
# model_uri = f"runs:/{latest_run_id}/model"
# model = mlflow.sklearn.load_model(model_uri)

# # Load preprocessing artifacts
# ohe = joblib.load("artifacts/ohe.pkl")
# scaler = joblib.load("artifacts/scaler.pkl")

# st.success(f"✅ Loaded latest trained model (Run ID: {latest_run_id})")

# # === SIDEBAR INPUT FORM ===
# st.sidebar.header("🧩 Input Customer Details")

# # Input fields for user
# gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
# SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
# Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
# Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
# tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
# PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
# MultipleLines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
# InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
# OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
# OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
# DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
# TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
# StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
# StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
# Contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
# PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
# PaymentMethod = st.sidebar.selectbox(
#     "Payment Method",
#     ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
# )
# MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
# TotalCharges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=3000.0)

# # === BUILD INPUT DATAFRAME ===
# input_dict = {
#     "gender": gender,
#     "SeniorCitizen": SeniorCitizen,
#     "Partner": Partner,
#     "Dependents": Dependents,
#     "tenure": tenure,
#     "PhoneService": PhoneService,
#     "MultipleLines": MultipleLines,
#     "InternetService": InternetService,
#     "OnlineSecurity": OnlineSecurity,
#     "OnlineBackup": OnlineBackup,
#     "DeviceProtection": DeviceProtection,
#     "TechSupport": TechSupport,
#     "StreamingTV": StreamingTV,
#     "StreamingMovies": StreamingMovies,
#     "Contract": Contract,
#     "PaperlessBilling": PaperlessBilling,
#     "PaymentMethod": PaymentMethod,
#     "MonthlyCharges": MonthlyCharges,
#     "TotalCharges": TotalCharges,
# }

# input_df = pd.DataFrame([input_dict])

# # === APPLY SAME FEATURE PROCESSING ===
# categorical_cols = [
#     "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
#     "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
#     "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
#     "PaperlessBilling", "PaymentMethod"
# ]
# numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

# # OneHotEncode + Scale
# encoded = ohe.transform(input_df[categorical_cols])
# encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(categorical_cols))
# scaled = scaler.transform(input_df[numerical_cols])
# scaled_df = pd.DataFrame(scaled, columns=numerical_cols)

# final_input = pd.concat([scaled_df, encoded_df], axis=1)

# # === PREDICT ===
# if st.sidebar.button("🔮 Predict Churn"):
#     pred_proba = model.predict_proba(final_input)[0, 1]
#     pred_label = "Yes" if pred_proba > 0.5 else "No"

#     st.subheader("🧠 Prediction Result")
#     st.metric("Churn Prediction", pred_label)
#     st.progress(int(pred_proba * 100))
#     st.metric("Churn Probability (%)", f"{pred_proba * 100:.2f}")

# # === SHOW MODEL METRICS AND VISUALS ===
# st.markdown("---")
# st.subheader("📈 Model Evaluation Metrics")

# metrics_path = "artifacts/evaluation_report.json"
# if os.path.exists(metrics_path):
#     import json
#     with open(metrics_path, "r") as f:
#         metrics = json.load(f)
#     st.json(metrics)
# else:
#     st.warning("Evaluation metrics not found. Run the pipeline with evaluation stage.")

# # Display confusion matrix and ROC curve if available
# col1, col2 = st.columns(2)

# if os.path.exists("artifacts/confusion_matrix.png"):
#     with col1:
#         st.image(Image.open("artifacts/confusion_matrix.png"), caption="Confusion Matrix")

# if os.path.exists("artifacts/roc_curve.png"):
#     with col2:
#         st.image(Image.open("artifacts/roc_curve.png"), caption="ROC Curve")

# st.markdown("---")
# st.info("ℹ️ Latest MLflow model, metrics, and artifacts are auto-loaded for each new pipeline run.")



import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image

# === PAGE CONFIGURATION ===
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📞 Telecom Customer Churn Prediction Dashboard")
st.markdown("### Powered by DVC + Streamlit")

# === LOAD LOCAL ARTIFACTS ===
if not os.path.exists("artifacts/model.pkl"):
    st.error("❌ Model not found. Please train it locally and push artifacts to your repo.")
    st.stop()

# Load trained model and preprocessing tools
model = joblib.load("artifacts/model.pkl")
ohe = joblib.load("artifacts/ohe.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

st.success("✅ Model and preprocessing artifacts loaded successfully!")

# === SIDEBAR INPUT FORM ===
st.sidebar.header("🧩 Input Customer Details")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
TotalCharges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=3000.0)

# === PREPARE INPUT FOR MODEL ===
input_data = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
}

input_df = pd.DataFrame([input_data])

categorical_cols = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]
numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

# OneHotEncode + Scale
encoded = ohe.transform(input_df[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(categorical_cols))
scaled = scaler.transform(input_df[numerical_cols])
scaled_df = pd.DataFrame(scaled, columns=numerical_cols)

final_input = pd.concat([scaled_df, encoded_df], axis=1)

# === PREDICTION ===
if st.sidebar.button("🔮 Predict Churn"):
    pred_proba = model.predict_proba(final_input)[0, 1]
    pred_label = "Yes" if pred_proba > 0.5 else "No"

    st.subheader("🧠 Prediction Result")
    st.metric("Churn Prediction", pred_label)
    st.progress(int(pred_proba * 100))
    st.metric("Churn Probability (%)", f"{pred_proba * 100:.2f}")

# === DISPLAY METRICS & VISUALS ===
st.markdown("---")
st.subheader("📈 Model Evaluation")

col1, col2 = st.columns(2)

if os.path.exists("artifacts/confusion_matrix.png"):
    with col1:
        st.image(Image.open("artifacts/confusion_matrix.png"), caption="Confusion Matrix")

if os.path.exists("artifacts/roc_curve.png"):
    with col2:
        st.image(Image.open("artifacts/roc_curve.png"), caption="ROC Curve")

st.markdown("---")
st.info("ℹ️ Model and metrics are loaded from the local artifacts folder for Streamlit Cloud compatibility.")
