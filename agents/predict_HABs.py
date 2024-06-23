import pprint
import openai
import os
from dotenv import load_dotenv
from openai import Client
import joblib
import pandas as pd

load_dotenv()

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

openai.api_key = os.getenv("OPEN_AI_KEY")

file = Client.files.create(
  file=open("Dataset.xlsx", "rb"),
  purpose='assistants'
)

assistant = Client.beta.assistants.create(
  name="HAB Predictor",
  description="You are great at predicting the possibility of Harmful Algal Blooms in a lake given the amount of chlorophyll a fluorescence in the lake.",
  model="gpt-4o",
  tools=[{"type": "code_interpreter"}],
  tool_resources={
    "code_interpreter": {
      "file_ids": [file.id]
    }
  }
)

thread = Client.beta.threads.create(
  messages=[
    {
      "role": "user",
      "content": "Create 3 data visualizations based on the trends in this file.",
      "attachments": [
        {
          "file_id": file.id,
          "tools": [{"type": "code_interpreter"}]
        }
      ]
    }
  ]
)

run = Client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id
)

run = Client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
  model="gpt-4o",
  instructions="New instructions that override the Assistant instructions",
  tools=[{"type": "code_interpreter"}, {"type": "file_search"}]
)