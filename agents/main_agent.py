import os
import joblib
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from dotenv import load_dotenv
from autogen.agentchat import AssistantAgent

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

# User inputs
temperature = float(input('What is the temperature? '))
salinity = float(input('What is the salinity level? '))
uvb = float(input('What is the uvb level? '))
chlorophyll_a_corrected = predict_chlorophyll(temperature, salinity, uvb)
risk_category = categorize_chlorophyll_a(chlorophyll_a_corrected)

print(f"Predicted Chlorophyll a Corrected: {chlorophyll_a_corrected} µg/L - {risk_category}")

config_list = [{"model": "gpt-4", "api_key": openai_api_key}]

messages = [
    {"role": "system", "content": "You are a helpful assistant that finds the probability of Harmful Algal Blooms existing in a lake given a Chlorophyll A Corrected value and provides recommendations."},
    {
        "role": "user",
        "content": f"## Chlorophyll A corrected value: {chlorophyll_a_corrected} µg/L\n## Risk Category: {risk_category}\n\nDo I have to be concerned about this, and do I need to take any action to prevent it developing into HABs?",
    },
]

# Create an AssistantAgent instance named "assistant"
assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant that gives information about the probability of Harmful Algal Blooms in lakes and provides recommendations.",
    llm_config={
        "timeout": 600,
        "seed": 42,
        "config_list": config_list,
    },
)

# Ask user if they want to learn more about the result
user_prompt = "Would you like to learn more about this result? (yes/no): "
user_input = input(user_prompt).strip().lower()

if user_input in ["yes", "y"]:
    # Send the query to the assistant agent and get the response
    additional_message = {
        "role": "user",
        "content": f"The calculated Chlorophyll A corrected value is {chlorophyll_a_corrected} µg/L which indicates a '{risk_category}' risk. Should I be concerned and what actions can I take to prevent it developing into HABs?"
    }
    
    # Include the initial messages and the new message
    response = assistant.generate_reply(messages + [additional_message])
    
    # Print the response
    print(response)
    
    # Ask user if they want a more tailored response
    tailored_prompt = "Would you like a more tailored response with additional details? (yes/no): "
    tailored_input = input(tailored_prompt).strip().lower()
    
    if tailored_input in ["yes", "y"]:
        # Gather additional details from the user
        weather = input('What is the current weather? ')
        season = input('What is the current season? ')
        water_flow = input('Describe the water flow (e.g., still, slow, fast): ')
        
        # Send the tailored query to the assistant agent and get the response
        tailored_message = {
            "role": "user",
            "content": (f"Considering the following additional details:\n"
                        f"Weather: {weather}\n"
                        f"Season: {season}\n"
                        f"Water Flow: {water_flow}\n"
                        f"The calculated Chlorophyll A corrected value is {chlorophyll_a_corrected} µg/L which indicates a '{risk_category}' risk. "
                        f"Should I be concerned and what actions can I take to prevent it developing into HABs?")
        }
        
        tailored_response = assistant.generate_reply(messages + [tailored_message])
        
        # Print the tailored response
        print(tailored_response)
else:
    print("End of conversation.")
