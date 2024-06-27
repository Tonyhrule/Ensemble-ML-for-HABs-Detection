import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from dotenv import load_dotenv
import os
from autogen.agentchat import AssistantAgent

# Load environment variables
load_dotenv()

# Check if the OpenAI API key is loaded correctly
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Load the scaler and models
models_dir = 'models'
output_dir = 'output'
scaler = joblib.load(os.path.join(output_dir, 'scaler.pkl'))
rf = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
gb = joblib.load(os.path.join(models_dir, 'gb_model.pkl'))
nn = joblib.load(os.path.join(models_dir, 'nn_model.pkl'))

# Load the processed data to fit the polynomial features
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = joblib.load(os.path.join(output_dir, 'processed_data.pkl'))

# Recreate the polynomial features transformer with the same degree as used in training
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
poly.fit(X_train_scaled)

def predict_chlorophyll(temperature, salinity, uvb):
    # Create a DataFrame for the input features to include feature names
    features = pd.DataFrame([[float(temperature), float(salinity), float(uvb)]], columns=['Temperature', 'Salinity', 'UVB'])
    
    # Scale the input features
    features_scaled = scaler.transform(features)
    
    # Apply polynomial transformation
    features_poly = poly.transform(features_scaled)
    
    # Make predictions with each model
    rf_pred = rf.predict(features_poly)
    gb_pred = gb.predict(features_poly)
    nn_pred = nn.predict(features_poly)
    
    # Ensemble prediction (simple averaging)
    ensemble_pred = (rf_pred + gb_pred + nn_pred) / 3
    
    return ensemble_pred[0]

def categorize_chlorophyll_a(chlorophyll_a_value):
    if chlorophyll_a_value < 2:
        return "Low Risk (0-5% Probability)"
    elif 2 <= chlorophyll_a_value < 7:
        return "Moderate Risk (5-25% Probability)"
    elif 7 <= chlorophyll_a_value < 12:
        return "High Risk (25-50% Probability)"
    elif chlorophyll_a_value >= 12:
        return "Very High Risk (50-100% Probability)"
    else:
        return "Invalid Value"

# Streamlit App
st.set_page_config(page_title="Harmful Algal Blooms Prediction", page_icon=":algae:", layout="wide")

st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
    """, unsafe_allow_html=True)

st.title('Predicting Harmful Algal Blooms')

st.write("""
Harmful algal blooms (HABs) are overgrowths of algae in water that can have severe impacts on human health, aquatic ecosystems, and the economy. 
This app predicts the risk of HABs by estimating corrected chlorophyll-a levels based on the input values of temperature, salinity, and UVB radiation.
""")

# Input fields
temperature = st.slider('Temperature of Water (°C)', min_value=0, max_value=40, value=15, step=1)
salinity = st.slider('Salinity (PSU)', min_value=0, max_value=40, value=20, step=1)
uvb = st.slider('UVB Radiation (mW/m²)', min_value=0, max_value=100, value=30, step=1)

# Initialize session state variables
if "chlorophyll_a_corrected" not in st.session_state:
    st.session_state["chlorophyll_a_corrected"] = None
if "risk_category" not in st.session_state:
    st.session_state["risk_category"] = None
if "assistant_response" not in st.session_state:
    st.session_state["assistant_response"] = None
if "tailored_response" not in st.session_state:
    st.session_state["tailored_response"] = None
if "show_assistant_response" not in st.session_state:
    st.session_state["show_assistant_response"] = False
if "show_tailored_form" not in st.session_state:
    st.session_state["show_tailored_form"] = False

# Predict button
if st.button('Predict'):
    chlorophyll_a_corrected = predict_chlorophyll(temperature, salinity, uvb)
    risk_category = categorize_chlorophyll_a(chlorophyll_a_corrected)
    st.session_state["chlorophyll_a_corrected"] = chlorophyll_a_corrected
    st.session_state["risk_category"] = risk_category
    st.session_state["assistant_response"] = None  # Reset the assistant response

# Show prediction result if available
if st.session_state["chlorophyll_a_corrected"] is not None:
    st.write(f"**Predicted Chlorophyll a Corrected:** {st.session_state['chlorophyll_a_corrected']:.2f} µg/L - **{st.session_state['risk_category']}**")
    st.write("""
    ### Model Details
    This model is trained to predict the corrected Chlorophyll-a level using the input values of temperature, salinity, and UVB radiation.
    """)

# Assistant integration
config_list = [{"model": "gpt-4", "api_key": openai_api_key}]
messages = [
    {"role": "system", "content": "You are a helpful assistant that finds the probability of Harmful Algal Blooms existing in a lake given a Chlorophyll A Corrected value and provides recommendations."},
    {
        "role": "user",
        "content": f"## Chlorophyll A corrected value: {st.session_state['chlorophyll_a_corrected']} µg/L\n## Risk Category: {st.session_state['risk_category']}\n\nDo I have to be concerned about this, and do I need to take any action to prevent it developing into HABs?",
    },
]

assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant that gives information about the probability of Harmful Algal Blooms in lakes and provides recommendations.",
    llm_config={
        "timeout": 600,
        "seed": 42,
        "config_list": config_list,
    },
)

if st.session_state["chlorophyll_a_corrected"] is not None and st.session_state["assistant_response"] is None:
    if st.button('Learn More About Your Result'):
        with st.spinner('Please wait while we generate the response...'):
            response = assistant.generate_reply(messages)
            st.session_state["assistant_response"] = response
            st.session_state["show_assistant_response"] = True

# Show assistant response if available
if st.session_state["show_assistant_response"]:
    st.write(st.session_state["assistant_response"])

    if st.button('Get a More Tailored Response'):
        st.session_state["show_tailored_form"] = True

    if st.session_state["show_tailored_form"]:
        with st.form(key='tailored_form'):
            weather = st.text_input('Current Weather:', value="", placeholder="Enter current weather")
            season = st.radio('Current Season:', ['Spring', 'Summer', 'Autumn', 'Winter'])
            water_flow = st.text_input('Describe the Water Flow (e.g., still, slow, fast):', value="", placeholder="Describe the water flow")
            submit_button = st.form_submit_button(label='Confirm')

            if submit_button:
                tailored_message = {
                    "role": "user",
                    "content": (f"Considering the following additional details:\n"
                                f"Weather: {weather}\n"
                                f"Season: {season}\n"
                                f"Water Flow: {water_flow}\n"
                                f"The calculated Chlorophyll A corrected value is {st.session_state['chlorophyll_a_corrected']} µg/L which indicates a '{st.session_state['risk_category']}' risk. "
                                f"Should I be concerned and what actions can I take to prevent it developing into HABs?")
                }

                with st.spinner('Please wait while we generate the tailored recommendations...'):
                    tailored_response = assistant.generate_reply(messages + [tailored_message])
                    st.session_state["tailored_response"] = tailored_response

# Show tailored response if available
if st.session_state["tailored_response"] is not None:
    st.write(st.session_state["tailored_response"])
