import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import openai
from dotenv import load_dotenv
from io import StringIO

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def chatgpt_response(messages, model):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )
    return response['choices'][0]['message']['content']

print("Type '/exit' to end the conversation at any time.")

# Load the dataset
print("Loading dataset...")
data_path = 'Dataset.xlsx'
data = pd.read_excel(data_path)
print("Dataset loaded.")

# Drop rows with missing values in features or target
print("Dropping rows with missing values...")
data = data.dropna(subset=['Temperature', 'Salinity', 'UVB', 'ChlorophyllaFlor'])
print("Rows with missing values dropped.")

# Sample a smaller subset of data to send to the model
sample_data = data.sample(n=100)  # Adjust the sample size as needed

messages = [
    {"role": "system", "content": "You are a generator that generates additional data that is proportional but not the same to the given dataset. Do not change the values to make it deviate from the original data set too much. The temperature should be between 7 and 10, the salinity should be between 32 and 35, and the UVB should not be greater than 100. Generate 100 rows."},
    {"role": "user", "content": f"{sample_data.to_csv(index=False)}"}
]

model = "gpt-4"

new_data_list = []
print("Generating additional data...")
for i in range(1):  # Adjust the number of batches as needed
    print(f"Generating batch {i+1}/1...")
    string_data = StringIO(chatgpt_response(messages=messages, model=model))
    more_data = pd.read_csv(string_data, sep=',', header=None)
    more_data.columns = ['Temperature', 'Salinity', 'UVB', 'ChlorophyllaFlor']
    new_data_list.append(more_data)
print("Additional data generation complete.")

more_data = pd.concat(new_data_list, axis=0, ignore_index=True)

# Debugging: Print the columns of more_data
print("Columns of more_data:", more_data.columns)

columns = ['Temperature', 'Salinity', 'UVB', 'ChlorophyllaFlor']

# Ensure all required columns are present in more_data
missing_columns = [col for col in columns if col not in more_data.columns]
if missing_columns:
    raise KeyError(f"Missing columns in generated data: {missing_columns}")

data = data[columns]
more_data = more_data[columns]

new_data = pd.concat([data, more_data], axis=0, ignore_index=True)

# Select features and target
print("Selecting features and target...")
X = new_data[['Temperature', 'Salinity', 'UVB']]
y = new_data['ChlorophyllaFlor']

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split complete.")

# Scale the features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling complete.")

print(X_train_scaled)

# Save the processed data and the scaler
print("Saving processed data and scaler...")
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
joblib.dump((X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test), os.path.join(output_dir, 'processed_data.pkl'))
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
print("Preprocessing complete. Data saved to the output folder.")
