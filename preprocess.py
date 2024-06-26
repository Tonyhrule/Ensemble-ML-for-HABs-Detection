import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import openai
from dotenv import load_dotenv
from io import StringIO
import time

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to get GPT-4 response with retry logic
def chatgpt_response(messages, model, retries=3, timeout=300):
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                timeout=timeout
            )
            return response.choices[0].message['content']
        except openai.error.Timeout as e:
            print(f"Request timed out. Attempt {attempt + 1} of {retries}. Retrying...")
            time.sleep(5)  # Wait for a bit before retrying
        except Exception as e:
            print(f"An error occurred: {e}")
            break
    return None

# Load the dataset
print("Loading dataset...")
data_path = 'Dataset.xlsx'
data = pd.read_excel(data_path)
print("Dataset loaded.")

# Drop rows with missing values in features or target
print("Dropping rows with missing values...")
data = data.dropna(subset=['Temperature', 'Salinity', 'UVB', 'ChlorophyllaFlor'])
print("Rows with missing values dropped.")

# Sample a subset of data prioritizing the highest ChlorophyllaFlor values
print("Sampling data with highest ChlorophyllaFlor values...")  
data = data.sort_values(by='ChlorophyllaFlor', ascending=False)
sample_data = data.head(200)  # Reduce the sample size

# Define messages for GPT-4
messages = [
    {"role": "system", "content": "You are a data generator that generates additional data rows based on the given dataset. Each row should include 'Temperature', 'Salinity', 'UVB', and 'ChlorophyllaFlor' columns. The temperature should be between 8.25 and 10, the salinity should be between 33 and 35, the UVB should not be greater than 100, and the ChlorophyllaFlor should be between 1.5 and 2.5. Generate 100 rows."},
    {"role": "user", "content": f"Please generate additional rows similar to the following dataset:\n\n{sample_data.to_csv(index=False)}"}
]

# Generate additional data using GPT-4
model = "gpt-4"
new_data_list = []
print("Generating additional data...")
for i in range(3):  # Adjust the number of batches as needed
    print(f"Generating batch {i+1}/3...")
    response_content = chatgpt_response(messages=messages, model=model)
    if response_content:
        string_data = StringIO(response_content)
        more_data = pd.read_csv(string_data, sep=',')
        
        # Debugging: Print the content of more_data to check its structure
        print(f"Content of generated data batch {i+1}:")
        print(more_data.head())
        
        new_data_list.append(more_data)
    else:
        print("Failed to generate data for this batch. Skipping...")
print("Additional data generation complete.")

if new_data_list:
    more_data = pd.concat(new_data_list, axis=0, ignore_index=True)
else:
    raise ValueError("No data was generated. Please check the generation process.")

# Ensure all required columns are present in more_data
columns = ['Temperature', 'Salinity', 'UVB', 'ChlorophyllaFlor']
missing_columns = [col for col in columns if col not in more_data.columns]
if missing_columns:
    raise KeyError(f"Missing columns in generated data: {missing_columns}")

data = data[columns]
more_data = more_data[columns]

# Convert all columns to numeric, coercing errors to NaN
more_data = more_data.apply(pd.to_numeric, errors='coerce')

# Drop any rows with NaN values that resulted from conversion
more_data = more_data.dropna()

# Concatenate original and generated data
new_data = pd.concat([data, more_data], axis=0, ignore_index=True)

# Select features and target
print("Selecting features and target...")
X = new_data[['Temperature', 'Salinity', 'UVB']]
y = new_data['ChlorophyllaFlor']

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split complete.")

# Ensure there are no NaN values before scaling
X_train = X_train.dropna()
X_test = X_test.dropna()

# Scale the features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling complete.")

# Save the processed data and the scaler
print("Saving processed data and scaler...")
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
joblib.dump((X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test), os.path.join(output_dir, 'processed_data.pkl'))
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
print("Preprocessing complete. Data saved to the output folder.")
