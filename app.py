import streamlit as st
import torch
import json
import numpy as np
import pandas as pd
from client_torch import MLP

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Heart Risk Predictor", layout="wide")
MODEL_PATH = "global_model_bundle.pt"
SCALER_PATH = "scaler.pkl"
CLIP_VALUES = {
    "age": (20, 80),
    "height": (140, 200),
    "weight": (40, 120),
    "ap_hi": (90, 180),
    "ap_lo": (60, 120),
}

# -------------------------------------------------
# FIELD TRANSLATORS
# -------------------------------------------------
FEATURE_TRANSLATIONS = {
    "age": "Age (years)",
    "height": "Height (cm)",
    "weight": "Weight (kg)",
    "ap_hi": "Systolic Blood Pressure",
    "ap_lo": "Diastolic Blood Pressure",
    "pulse_pressure": "Pulse Pressure",
    "bmi": "Body Mass Index",
    "age_bmi": "Age × BMI",
    "pp_chol": "Pulse Pressure × Cholesterol",
    "bmi2": "BMI²",
    "cholesterol_2": "Cholesterol: Above Normal",
    "cholesterol_3": "Cholesterol: Well Above Normal",
    "gluc_2": "Glucose: Above Normal",
    "gluc_3": "Glucose: Well Above Normal",
    "smoke_1": "Smoker",
    "alco_1": "Alcohol Intake"
}

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_bundle():
    bundle = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model_state = bundle["model_state_dict"]
    feature_names = bundle["feature_names"]
    global_shap = bundle.get("global_shap_json", None)

    first_layer_weight = model_state["net.0.weight"]
    in_features = first_layer_weight.shape[1]

    model = MLP(in_features=in_features)
    model.load_state_dict(model_state)
    model.eval()

    return model, feature_names, global_shap

model, feature_names, global_shap = load_bundle()

# -------------------------------------------------
# PREPROCESSING
# -------------------------------------------------
def preprocess_input(data_dict, feature_names, scaler_path=SCALER_PATH):
    import joblib
    import os
    import pandas as pd
    import numpy as np
    import torch
    
    #st.subheader("🕵️ Debugging Feature Alignment")

    # 1. Create DataFrame
    df = pd.DataFrame([data_dict])

    # 2. Engineering
    # We use a copy to avoid SettingWithCopy warnings
    df = df.copy()
    df["pulse_pressure"] = float(df["ap_hi"].iloc[0] - df["ap_lo"].iloc[0])
    h_m = float(df["height"].iloc[0] / 100.0)
    df["bmi"] = float(df["weight"].iloc[0] / (h_m ** 2))
    df["age_bmi"] = float(df["age"].iloc[0] * df["bmi"].iloc[0])
    df["bmi2"] = float(df["bmi"].iloc[0] ** 2)
    
    orig_chol = 1
    if data_dict.get("cholesterol_2") == 1: orig_chol = 2
    elif data_dict.get("cholesterol_3") == 1: orig_chol = 3
    df["pp_chol"] = float(df["pulse_pressure"].iloc[0] * orig_chol)

    # --- LOGGER 1: Before Reindexing ---
    #st.write(f"Columns generated in App: {list(df.columns)}")
    #st.write(f"Total Count (Pre-reindex): {len(df.columns)}")

    # 3. Align with Model Features (The list from the server)
    # --- LOGGER 2: The Target List ---
    #st.write(f"Target Features (from Bundle): {feature_names}")
    #st.write(f"Target Count: {len(feature_names)}")

    # Perform the reindex
    df_final = df.reindex(columns=feature_names, fill_value=0.0)

    # --- LOGGER 3: Final Verification ---
    # st.write(f"Final Column Count: {len(df_final.columns)}")
    if len(df_final.columns) != 17:
        #st.error(f"⚠️ Mismatch! Reindex resulted in {len(df_final.columns)} columns, but Scaler needs 17.")
        pass

    # 4. Scaling
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            # Use the underlying numpy values
            input_array = df_final.values.astype(np.float32)
            
            # --- LOGGER 4: The Array Shape ---
            #st.write(f"Shape sent to Scaler: {input_array.shape}")
            
            scaled_values = scaler.transform(input_array)
            return torch.tensor(scaled_values, dtype=torch.float32)
            
        except Exception as e:
            st.error(f"Transformation Failed: {e}")
            # This is where your error is likely triggering
            return torch.tensor(df_final.values, dtype=torch.float32)
    else:
        st.error("Scaler file not found!")
        return torch.tensor(df_final.values, dtype=torch.float32)

# -------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------
def run_prediction(input_data):
    import os
    tensor = preprocess_input(input_data, feature_names)
    
    # st.write("DEBUG: Scaler exists?", os.path.exists(SCALER_PATH))  # Debug line

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).item()
        risk_percent = prob * 100

    # --- Determine Risk Category ---
    if prob < 0.3:
        risk_label = "Low Risk"
        color_func = st.success
        risk_message = "Your current metrics indicate a low probability of heart disease."
    elif prob < 0.7:
        risk_label = "Moderate Risk"
        color_func = st.warning
        risk_message = "Your metrics suggest a moderate risk. Consider lifestyle adjustments."
    else:
        risk_label = "High Risk"
        color_func = st.error
        risk_message = "Your metrics indicate a high risk of heart disease. Please consult a physician."

    # --- Display Risk Summary ---
    st.divider()
    st.subheader("Heart Disease Risk Assessment")
    color_func(f"{risk_label}: {risk_percent:.1f}%")
    st.info(risk_message)

    # --- Progress Bar ---
    st.progress(min(int(risk_percent), 100))

    # --- Detailed Feature Contributions (if SHAP exists) ---
    contributors = get_top_contributors()
    if contributors is not None:
        st.divider()
        st.subheader("Top Global Features Contributing to Risk")
        st.dataframe(contributors, width = 'stretch')
        top_features = ", ".join(contributors["Feature"].tolist())
        st.markdown(
            f"These are the top {len(contributors)} features globally contributing to heart risk predictions: **{top_features}**."
        )

# -------------------------------------------------
# INTERPRET PROBABILITY
# -------------------------------------------------
def interpret_probability(prob):
    if prob < 0.3:
        return "Low predicted risk."
    elif prob < 0.7:
        return "Moderate predicted risk."
    else:
        return "High predicted risk."

# -------------------------------------------------
# SHAP CONTRIBUTORS
# -------------------------------------------------
def get_top_contributors(top_n=5, shap_path="shap_outputs/global_shap.json"):
    # Check if file exists
    import os
    if not os.path.exists(shap_path):
        return None

    # Load the global SHAP JSON
    with open(shap_path, "r") as f:
        global_shap_json = json.load(f)

    if len(global_shap_json) == 0:
        return None

    # Take last round if keys are numeric or just pick any
    try:
        last_round = sorted(global_shap_json.keys(), key=lambda x: int(x))[-1]
    except ValueError:
        last_round = list(global_shap_json.keys())[-1]

    shap_dict = global_shap_json[last_round]

    # Convert to DataFrame
    shap_df = pd.DataFrame(list(shap_dict.items()), columns=["Feature", "SHAP Value"])
    shap_df["Feature"] = shap_df["Feature"].map(FEATURE_TRANSLATIONS).fillna(shap_df["Feature"])
    shap_df["SHAP Value"] = shap_df["SHAP Value"].astype(float)

    # Sort by absolute value and pick top N
    top_df = shap_df.reindex(shap_df["SHAP Value"].abs().sort_values(ascending=False).index).head(top_n)
    top_df = top_df.reset_index(drop=True)

    return top_df
# -------------------------------------------------
# UI
# -------------------------------------------------
st.header("Heart Risk Predictor - Federated Learning (FedReX)")
tab1, tab2, tab3 = st.tabs(["Structured Input", "JSON Input", "Experiment Setup"])

# TAB 1: Structured input
with tab1:
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 50)
        height = st.number_input("Height (cm)", 100, 250, 170)
        weight = st.number_input("Weight (kg)", 30, 200, 70)
        ap_hi = st.number_input("Systolic BP", 80, 250, 120)
        ap_lo = st.number_input("Diastolic BP", 40, 150, 80)

    with col2:
        gender = st.radio("Gender", ["Female", "Male"]) 
        cholesterol = st.selectbox(
            "Cholesterol Level",
            ["Normal", "Above Normal", "Well Above Normal"]
        )
        gluc = st.selectbox(
            "Glucose Level",
            ["Normal", "Above Normal", "Well Above Normal"]
        )
        smoke = st.checkbox("Smoker")
        alco = st.checkbox("Alcohol Intake")
        active = st.checkbox("Physically Active")

    input_data = {
        "age": age,
        "gender": 1 if gender == "Female" else 2,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol_2": 1.0 if cholesterol=="Above Normal" else 0.0,
        "cholesterol_3": 1.0 if cholesterol=="Well Above Normal" else 0.0,
        "gluc_2": 1.0 if gluc=="Above Normal" else 0.0,
        "gluc_3": 1.0 if gluc=="Well Above Normal" else 0.0,
        "smoke_1": 1.0 if smoke else 0.0,
        "alco_1": 1.0 if alco else 0.0,
        "active_1": 1.0 if active else 0.0,
    }

    if st.button("Predict"):
        run_prediction(input_data)

# TAB 2: JSON input
with tab2:
    st.subheader("Paste JSON Input")
    json_input = st.text_area("JSON Format", height=200, value=json.dumps(input_data, indent=2))
    if st.button("Predict from JSON"):
        try:
            data = json.loads(json_input)
            run_prediction(data)
        except Exception as e:
            st.error(f"Invalid JSON input: {e}")

with tab3:
    st.subheader("Federated Learning Experiment Overview")

    # Experiment details
    total_records = 70000
    num_clients = 3
    rounds = 30
    data_per_client = total_records // num_clients
    framework = "Flower (FLWR) + PyTorch"
    flow = "Client-side training → Model aggregation → Global update"

    st.markdown(f"""
    **Total Records:** {total_records}  
    **Number of Clients:** {num_clients}  
    **Data per Client:** ~{data_per_client} records  
    **Federated Rounds:** {rounds}  
    **Framework:** {framework}  
    **Training Flow:** {flow}
    """)

    st.divider()

    st.info("This tab summarizes the experiment setup for the federated learning simulation.")