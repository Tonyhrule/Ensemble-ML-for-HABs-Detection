import os
import autogen
from autogen.agentchat import UserProxyAgent
import joblib
import pandas as pd

output_dir = 'output'

# Load the scaler and models
models_dir = 'models'
output_dir = 'output'
scaler = joblib.load(os.path.join(output_dir, 'scaler.pkl'))
rf = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
gb = joblib.load(os.path.join(models_dir, 'gb_model.pkl'))
nn = joblib.load(os.path.join(models_dir, 'nn_model.pkl'))

def predict_chlorophyll(temperature, salinity, uvb):
    # Create a DataFrame for the input features to include feature names
    features = pd.DataFrame([[temperature, salinity, uvb]], columns=['Temperature', 'Salinity', 'UVB'])
    
    # Scale the input features
    features_scaled = scaler.transform(features)
    
    # Make predictions with each model
    rf_pred = rf.predict(features)
    gb_pred = gb.predict(features_scaled)
    nn_pred = nn.predict(features_scaled)
    
    # Ensemble prediction (simple averaging)
    ensemble_pred = (rf_pred + gb_pred + nn_pred) / 3
    
    return ensemble_pred[0]

temperature = input('What is the temperature? ')
salinity = input('What is the salinity level? ')
uvb = input('What is the uvb level? ')
chlorophyll_a_fluorescence = predict_chlorophyll(temperature, salinity, uvb)

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": {
            "gpt-4-1106-preview",
            "gpt-3.5-turbo",
            "gpt-35-turbo",
        }
    },
)

#1. Create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    system_message="You are a helpful assisstant that gives information about the probability of Harmful Algal Blooms in lakes.",
    llm_config={
        "timeout": 600,
        "seed": 42,
        "config_list": config_list,
    },
)

#2. Create the MathUserProxyAgent instance named "mathproxyagent"
userproxyagent = UserProxyAgent(
    name="userproxyagent",
    human_input_mode="ALWAYS",
    code_execution_config={"use_docker": False},
)

#Given a math problem, we use the mathproxyagent to generate a prompt to be sent to the assistant as the initial message.
problem = "What is the probabilty that Harmful Algal Blooms exist in a lake given a Chlorophyll a Corrected value of {chlorophyll_a_corrected} ?"
#We call initiate_chat to start the conversation.
#When setting message=mathproxyagent.message_generator, you need to pass in the problem through the problem parameter.
userproxyagent.initiate_chat(assistant, message=userproxyagent.message_generator, problem=problem)