import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

MODEL_BINARY_FILE = 'predictive_maintenance_binary.pkl'
MODEL_MULTI_FILE = 'predictive_maintenance_multi.pkl'
FEATURES_FILE = 'predictive_maintenance_features.pkl'

# Search for any csv file in the current directory if default name isn't found
DATA_FILE = 'predictive_maintenance.csv'
if not os.path.exists(DATA_FILE):
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if csv_files:
        DATA_FILE = csv_files[0]

def train_and_save_models():
    if not os.path.exists(DATA_FILE):
        st.error(f"❌ No CSV dataset found in the project folder. Please make sure your .csv file is inside 'iris_project'.")
        return False

    with st.spinner("⏳ Training Machine Learning models from your dataset... Please wait a few seconds."):
        df = pd.read_csv(DATA_FILE)
        
        df_prep = df.copy()
        df_prep['Temp_Diff'] = df_prep['Process temperature [K]'] - df_prep['Air temperature [K]']
        df_prep['Power_kW'] = df_prep['Torque [Nm]'] * (df_prep['Rotational speed [rpm]'] * (2 * np.pi / 60)) / 1000
        df_prep['Strain'] = df_prep['Torque [Nm]'] * df_prep['Tool wear [min]']
        
        df_prep.drop(columns=['UDI', 'Product ID'], inplace=True, errors='ignore')
        type_map = {'L': 0, 'M': 1, 'H': 2}
        df_prep['Type'] = df_prep['Type'].map(type_map)
        
        drop_targets = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        X = df_prep.drop(columns=drop_targets, errors='ignore')
        y_binary = df_prep['Machine failure']
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
        rf_model.fit(X, y_binary)
        
        def assign_failure_type(row):
            if row.get('TWF') == 1: return 'TWF'
            elif row.get('HDF') == 1: return 'HDF'
            elif row.get('PWF') == 1: return 'PWF'
            elif row.get('OSF') == 1: return 'OSF'
            elif row.get('RNF') == 1: return 'RNF'
            else: return 'No Failure'

        df_prep['Failure_Type'] = df_prep.apply(assign_failure_type, axis=1)
        y_multi = df_prep['Failure_Type']
        
        rf_multi = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
        rf_multi.fit(X, y_multi)
        
        joblib.dump(rf_model, MODEL_BINARY_FILE)
        joblib.dump(rf_multi, MODEL_MULTI_FILE)
        joblib.dump(list(X.columns), FEATURES_FILE)
        
        st.success("✅ Models successfully trained and saved!")
        return True

@st.cache_resource
def load_models():
    if not (os.path.exists(MODEL_BINARY_FILE) and os.path.exists(MODEL_MULTI_FILE) and os.path.exists(FEATURES_FILE)):
        success = train_and_save_models()
        if not success:
            return None, None, None

    binary_model = joblib.load(MODEL_BINARY_FILE)
    multi_model = joblib.load(MODEL_MULTI_FILE)
    feature_names = joblib.load(FEATURES_FILE)
    return binary_model, multi_model, feature_names

rf_model, rf_multi, feature_columns = load_models()
models_loaded = rf_model is not None

st.title("⚙️ Predictive Maintenance & Fault Diagnosis System")
st.markdown("An end-to-end Machine Learning pipeline for real-time equipment monitoring, failure detection, and multi-class fault diagnosis.")

st.sidebar.header("📊 Input Sensor Parameters")

def user_input_features():
    product_type = st.sidebar.selectbox("Product Type (Type)", options=['L', 'M', 'H'], index=0)
    air_temp = st.sidebar.number_input("Air temperature [K]", min_value=250.0, max_value=350.0, value=298.1, step=0.1)
    process_temp = st.sidebar.number_input("Process temperature [K]", min_value=250.0, max_value=350.0, value=308.6, step=0.1)
    rot_speed = st.sidebar.number_input("Rotational speed [rpm]", min_value=0, max_value=5000, value=1551, step=10)
    torque = st.sidebar.number_input("Torque [Nm]", min_value=0.0, max_value=200.0, value=42.8, step=0.5)
    tool_wear = st.sidebar.number_input("Tool wear [min]", min_value=0, max_value=500, value=0, step=1)
    
    threshold = st.sidebar.slider("Failure Probability Threshold", min_value=0.1, max_value=0.9, value=0.35, step=0.05)

    data = {
        'Type': product_type,
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rot_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear
    }
    return data, threshold

input_data, custom_threshold = user_input_features()

def process_and_predict(input_data, threshold):
    df_in = pd.DataFrame([input_data])

    type_map = {'L': 0, 'M': 1, 'H': 2}
    df_in['Type'] = df_in['Type'].map(type_map)

    df_in['Temp_Diff'] = df_in['Process temperature [K]'] - df_in['Air temperature [K]']
    df_in['Power_kW'] = df_in['Torque [Nm]'] * (df_in['Rotational speed [rpm]'] * (2 * np.pi / 60)) / 1000
    df_in['Strain'] = df_in['Torque [Nm]'] * df_in['Tool wear [min]']

    df_in = df_in[feature_columns]

    prob_failure = rf_model.predict_proba(df_in)[0, 1]

    if prob_failure < threshold:
        return {
            "is_failure": False,
            "prob_failure": prob_failure,
            "message": "Equipment operates within normal parameters. No failure risk detected."
        }

    multi_probs = rf_multi.predict_proba(df_in)[0]
    classes = rf_multi.classes_

    fault_prob_dict = {
        cls: prob for cls, prob in zip(classes, multi_probs) if cls != 'No Failure'
    }

    top_fault = max(fault_prob_dict, key=fault_prob_dict.get)
    top_fault_prob = fault_prob_dict[top_fault]

    recommendations = {
        'HDF': "Heat Dissipation Failure detected: Inspect and service the cooling system immediately.",
        'PWF': "Power Failure detected: Check power supply stability and operational load limits.",
        'OSF': "Overstrain Failure detected: Reduce operating torque and inspect for mechanical overload.",
        'TWF': "Tool Wear Failure detected: Replace worn cutting tools before continuing operation.",
        'RNF': "Random Failure detected: Perform a general diagnostic inspection before restarting."
    }

    return {
        "is_failure": True,
        "prob_failure": prob_failure,
        "top_fault": top_fault,
        "top_fault_prob": top_fault_prob,
        "detailed_probs": fault_prob_dict,
        "action": recommendations.get(top_fault, "Perform a comprehensive machine inspection.")
    }

st.subheader("📋 Input Telemetry Summary")
col_in1, col_in2, col_in3 = st.columns(3)
col_in1.metric("Product Type", input_data['Type'])
col_in1.metric("Air Temp (K)", input_data['Air temperature [K]'])
col_in2.metric("Process Temp (K)", input_data['Process temperature [K]'])
col_in2.metric("Rotational Speed (RPM)", input_data['Rotational speed [rpm]'])
col_in3.metric("Torque (Nm)", input_data['Torque [Nm]'])
col_in3.metric("Tool Wear (min)", input_data['Tool wear [min]'])

st.divider()

if st.button("🚀 Run Predictive Diagnostic", type="primary"):
    if models_loaded:
        result = process_and_predict(input_data, custom_threshold)

        st.subheader("🔍 Stage 1: Anomaly Detection")
        fail_prob_pct = result["prob_failure"] * 100

        if not result["is_failure"]:
            st.success(f"✅ **Status: Normal Operation** | Predicted Failure Probability: **{fail_prob_pct:.2f}%**")
            st.info(result["message"])
        else:
            st.error(f"⚠️ **Status: Potential Machine Failure Detected!** | Probability: **{fail_prob_pct:.2f}%** (Exceeds decision threshold: {custom_threshold*100:.0f}%)")

            st.divider()
            st.subheader("🛠️ Stage 2: Fault Diagnosis & Root Cause Analysis")

            col_res1, col_res2 = st.columns([1, 1])

            with col_res1:
                st.warning(f"**Primary Diagnosis:** `{result['top_fault']}`")
                st.write(f"**Confidence Level:** `{result['top_fault_prob']*100:.2f}%`")
                st.subheader("💡 Action Plan:")
                st.info(result["action"])

            with col_res2:
                st.write("**Fault Mode Probability Distribution:**")
                prob_df = pd.DataFrame(
                    list(result["detailed_probs"].items()),
                    columns=['Fault Type', 'Probability']
                )
                prob_df['Probability (%)'] = prob_df['Probability'] * 100
                st.bar_chart(prob_df.set_index('Fault Type')['Probability (%)'])