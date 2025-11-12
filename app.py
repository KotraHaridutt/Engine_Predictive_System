# ----------------------------------------------------------------------------
# ✈️ TURBOFAN ENGINE PREDICTIVE MAINTENANCE DASHBOARD (V3.2)
# ----------------------------------------------------------------------------
# V3.2 changes (User Feedback):
# - "Individual Engine Lookup" is now at the top of the sidebar.
# - "Red Alert List" is now in a collapsible expander at the
#   bottom of the sidebar.
# - App now defaults to Engine 1 on load, not the first red alert engine.
#
# To run this app:
# 1. Open your terminal in GitHub Codespaces.
# 2. Run the command: streamlit run app.py
# ----------------------------------------------------------------------------

# --- 1. Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3
import plotly.graph_objects as go

# --- 2. Load Assets ---
MODEL_PATH = 'models/rul_model.pkl'
FEATURES_PATH = 'models/model_features.pkl'
DATA_PATH = 'data/turbofan.db'
TEST_DATA_PATH = 'data/test_FD001.txt'
RUL_DATA_PATH = 'data/RUL_FD001.txt'

@st.cache_data
def load_model(path):
    """Loads the pre-trained model."""
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_features(path):
    """Loads the list of features the model was trained on."""
    with open(path, 'rb') as f:
        features = pickle.load(f)
    return features

@st.cache_data
def load_test_data(test_path, rul_path, col_names):
    """Loads the test data and true RUL values."""
    df_test = pd.read_csv(test_path, sep='\s+', header=None, names=col_names)
    df_test = df_test.dropna(axis='columns', how='all')
    
    df_rul = pd.read_csv(rul_path, sep='\s+', header=None, names=['RUL'])
    df_rul = df_rul.dropna(axis='columns', how='all')
    
    df_test_grouped = df_test.groupby('unit_number')
    max_cycles = df_test_grouped['time_cycle'].max().reset_index()
    max_cycles = max_cycles.merge(df_rul, left_index=True, right_index=True)
    
    df_test = df_test.merge(max_cycles[['unit_number', 'RUL']], on='unit_number', how='left')
    
    return df_test

# --- 3. Helper Functions ---
def create_rolling_features(df, sensors, window_size=5):
    """Engineers rolling features for a given DataFrame."""
    df_out = df.copy()
    df_grouped = df_out.groupby('unit_number')
    
    for sensor in sensors:
        df_out[f'{sensor}_avg'] = df_grouped[sensor].rolling(window=window_size).mean().reset_index(level=0, drop=True)
        df_out[f'{sensor}_std'] = df_grouped[sensor].rolling(window=window_size).std().reset_index(level=0, drop=True)
    
    df_out = df_out.fillna(0)
    return df_out

def get_recommendation(rul, rmse=49):
    """Returns a status, recommendation, and color based on the predicted RUL."""
    red_threshold = 10 + rmse # ~59 cycles
    yellow_threshold = 100
    
    if rul <= red_threshold:
        status = "🔴 RED ALERT"
        recommendation = ("**SERVICE IMMEDIATELY.** Predicted RUL is within the "
                          f"model's error margin ({rmse} cycles). The ~$2M risk of "
                          "imminent failure outweighs the $250k cost of service.")
        color = "red"
    elif red_threshold < rul <= yellow_threshold:
        status = "🟡 YELLOW ALERT"
        recommendation = ("**SCHEDULE SERVICE.** Engine is approaching the "
                          "risk window. Plan for scheduled maintenance at the "
                          "next convenient opportunity.")
        color = "orange"
    else:
        status = "🟢 GREEN"
        recommendation = ("**NORMAL OPERATION.** Engine is operating safely. "
                          "Continue routine monitoring.")
        color = "green"
        
    return status, recommendation, color

@st.cache_data
def calculate_fleet_summary(_model, _df_test, _model_features, sensors_to_engineer):
    """Pre-calculates RUL, True RUL, and Status for the entire test fleet."""
    df_test_featured = create_rolling_features(_df_test, sensors_to_engineer)
    df_last_points = df_test_featured.groupby('unit_number').last().reset_index()
    X_fleet = df_last_points[_model_features]
    fleet_predictions = _model.predict(X_fleet)
    
    df_summary = pd.DataFrame({
        'unit_number': df_last_points['unit_number'],
        'predicted_rul': fleet_predictions.astype(int),
        'true_rul': df_last_points['RUL'].astype(int)
    })
    
    df_summary['status'] = df_summary['predicted_rul'].apply(lambda x: get_recommendation(x)[0])
    
    return df_summary.sort_values(by='predicted_rul')


# --- 4. Load All Data & Models ---
try:
    model = load_model(MODEL_PATH)
    model_features = load_features(FEATURES_PATH)
    
    column_names = ['unit_number', 'time_cycle', 'setting_1', 'setting_2', 'setting_3'] + \
                   [f'sensor_{i}' for i in range(1, 22)]
                   
    df_test = load_test_data(TEST_DATA_PATH, RUL_DATA_PATH, column_names)
    
    sensors_to_engineer = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 
                           'sensor_11', 'sensor_12', 'sensor_13', 'sensor_15', 'sensor_17', 
                           'sensor_20', 'sensor_21']
                           
    df_fleet_summary = calculate_fleet_summary(model, df_test, model_features, sensors_to_engineer)

except Exception as e:
    st.error(f"FATAL ERROR: Could not load assets or calculate fleet summary. {e}")
    st.stop()

# --- 5. Streamlit UI ---
st.set_page_config(
    page_title="Turbofan RUL Dashboard",
    page_icon="✈️",
    layout="wide"
)

# --- Sidebar ---
st.sidebar.title("✈️ Fleet Monitor")

# --- *** FIX 1 & 2: Individual Lookup is now first *** ---
st.sidebar.subheader("Individual Engine Lookup")
engine_list = df_test['unit_number'].unique().tolist()

# --- *** FIX 3: Default index is now 0 (Engine 1) *** ---
default_index = 0 
engine_id = st.sidebar.selectbox("Select Engine ID:", engine_list, index=default_index)

st.sidebar.divider()

# --- *** FIX 4: Red Alert List is now in a collapsible expander *** ---
red_alert_engines = df_fleet_summary[df_fleet_summary['status'] == "🔴 RED ALERT"]
with st.sidebar.expander(f"🔴 View Red Alert List ({len(red_alert_engines)} engines)"):
    if red_alert_engines.empty:
        st.success("No engines in red alert. ✅")
    else:
        st.dataframe(
            red_alert_engines[['unit_number', 'predicted_rul', 'true_rul']],
            column_config={
                "unit_number": "Engine ID",
                "predicted_rul": "Predicted RUL",
                "true_rul": "True RUL"
            }
        )

# --- 6. Main Dashboard Page ---
st.title(f"Engine {engine_id}: Predictive Maintenance Dashboard")

engine_data = df_test[df_test['unit_number'] == engine_id].copy()
engine_summary = df_fleet_summary[df_fleet_summary['unit_number'] == engine_id].iloc[0]

if not engine_data.empty:
    predicted_rul = engine_summary['predicted_rul']
    true_rul = engine_summary['true_rul']
    status, recommendation, color = get_recommendation(predicted_rul)

    # --- Display KPIs ---
    st.header(f"Status: {status}")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Predicted RUL (Cycles)", value=predicted_rul, delta=f"{predicted_rul - true_rul} vs. True RUL", delta_color="inverse")
    col2.metric(label="True RUL (Cycles)", value=true_rul)
    col3.metric(label="Last Recorded Cycle", value=engine_data['time_cycle'].max())

    # --- Display Recommendation ---
    st.subheader("Maintenance Recommendation")
    st.markdown(f"**:{color}[{recommendation}]**")

    # --- Sensor Graph ---
    st.header("Sensor Degradation Analysis")
    
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': model_features,
        'importance': feature_importances
    }).sort_values(by='importance', ascending=False)
    
    top_features_names = []
    for f in importance_df['feature']:
        if 'sensor_' in f:
            base_sensor = f.split('_')[0] + '_' + f.split('_')[1]
            if base_sensor not in top_features_names and base_sensor in engine_data.columns:
                top_features_names.append(base_sensor)
        if len(top_features_names) == 4:
            break
    
    if not top_features_names:
        top_features_names = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7']

    fig_sensors = go.Figure()
    for sensor in top_features_names:
        fig_sensors.add_trace(go.Scatter(
            x=engine_data['time_cycle'], 
            y=engine_data[sensor], 
            mode='lines', 
            name=sensor
        ))
    
    fig_sensors.update_layout(
        title=f"Top Sensor Readings for Engine {engine_id}",
        xaxis_title="Time (Cycles)",
        yaxis_title="Sensor Value",
        legend_title="Sensors"
    )
    st.plotly_chart(fig_sensors, use_container_width=True)

    # --- Feature Importance ---
    st.header("Model's Top 5 Predictive Features (Fleet-Wide)")
    st.markdown("This chart shows the top 5 features the *model* found most critical for making predictions during training. This is **model-level (global)** info, so it will not change for each engine.")

    try:
        top_5_features = importance_df.head(5)
        
        fig_imp = go.Figure(go.Bar(
            x=top_5_features['importance'],
            y=top_5_features['feature'],
            orientation='h',
            marker_color='#007bff' 
        ))
        fig_imp.update_layout(
            title="Top 5 Most Important Features (Global)",
            xaxis_title="Feature Importance Score",
            yaxis_title="Feature Name",
            yaxis_autorange="reversed",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    except Exception as e:
        st.error(f"Could not generate feature importance chart: {e}")

else:
    st.error(f"No data found for Engine ID {engine_id}.")