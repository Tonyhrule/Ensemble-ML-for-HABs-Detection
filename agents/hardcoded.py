import os
import joblib
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from dotenv import load_dotenv

load_dotenv()

# Check if the OpenAI API key is loaded correctly
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Load the scaler and models
models_dir = 'models'
output_dir = 'output'
scaler = joblib.load(os.path.join(output_dir, 'scaler.pkl'))
rf = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
gb = joblib.load(os.path.join(models_dir, 'gb_model.pkl'))
nn = joblib.load(os.path.join(models_dir, 'nn_model.pkl'))

# Recreate the polynomial features transformer with the same degree as used in training
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

def predict_chlorophyll(temperature, salinity, uvb):
    # Create a DataFrame for the input features to include feature names
    features = pd.DataFrame([[float(temperature), float(salinity), float(uvb)]], columns=['Temperature', 'Salinity', 'UVB'])
    
    # Scale the input features
    features_scaled = scaler.transform(features)
    
    # Apply polynomial transformation
    features_poly = poly.fit_transform(features_scaled)
    
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

# User input
temperature = float(input('What is the temperature? '))
salinity = float(input('What is the salinity level? '))
uvb = float(input('What is the uvb level? '))

# Prediction and categorization
chlorophyll_a_corrected = predict_chlorophyll(temperature, salinity, uvb)
risk_category = categorize_chlorophyll_a(chlorophyll_a_corrected)

# Print the prediction results
print(f"Predicted Chlorophyll a Corrected: {chlorophyll_a_corrected} µg/L - {risk_category}")

# Ask user if they want to learn more about the result
user_prompt = "Would you like to learn more about this result? (yes/no): "
user_input = input(user_prompt).strip().lower()

if user_input in ["yes", "y"]:
    # Provide recommendation based on risk category
    if risk_category == "Low Risk (0-5% Probability)":
        response = "The risk of harmful algal blooms (HABs) is low. No immediate action is required."
    elif risk_category == "Moderate Risk (5-25% Probability)":
        response = ("There is a moderate risk of harmful algal blooms (HABs). Consider monitoring the water quality "
                    "and reducing nutrient runoff from nearby agricultural or residential areas.")
    elif risk_category == "High Risk (25-50% Probability)":
        response = ("There is a high risk of harmful algal blooms (HABs). It is advisable to take preventive measures "
                    "such as reducing nutrient runoff, improving water circulation, and monitoring the water quality regularly.")
    elif risk_category == "Very High Risk (50-100% Probability)":
        response = ("The risk of harmful algal blooms (HABs) is very high. Immediate action is required to prevent HABs. "
                    "Implement measures to reduce nutrient runoff, increase water circulation, and closely monitor the water quality.")
    else:
        response = "The calculated value is invalid. Please check the input values and try again."

    print(response)
else:
    print("End of conversation.")