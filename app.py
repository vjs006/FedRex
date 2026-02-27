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
    st.write(f"Final Column Count: {len(df_final.columns)}")
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
    st.write("DEBUG: Scaler exists?", os.path.exists(SCALER_PATH))
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).item()
        pred = 1 if prob >= 0.5 else 0
    
    st.write("Input tensor:", tensor)      # logs the input tensor values
    st.write("Raw model probability:", prob)  # logs the predicted probability

    # Display results
    risk_percent = prob * 100
    if pred == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")

    st.progress(min(int(risk_percent), 100))
    st.markdown(f"### Risk Probability: **{risk_percent:.2f}%**")

    # Probability interpretation
    st.write(interpret_probability(prob))

    # SHAP contributors
    contributors = get_top_contributors()
    if contributors is not None:
        st.divider()
        st.subheader("Top contributing features")
        st.dataframe(contributors, use_container_width=True)

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
def get_top_contributors():
    if global_shap is None:
        return None
    last_round = sorted(global_shap.keys(), key=lambda x: int(x))[-1]
    shap_dict = global_shap[last_round]
    shap_df = (
        pd.DataFrame(shap_dict.items(), columns=["Feature", "Value"])
        .sort_values("Value", ascending=False)
        .head(3)
    )
    shap_df["Feature"] = shap_df["Feature"].map(FEATURE_TRANSLATIONS)
    shap_df["Value"] = shap_df["Value"].round(4)
    return shap_df

# -------------------------------------------------
# UI
# -------------------------------------------------
tab1, tab2 = st.tabs(["Structured Input", "JSON Input"])

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