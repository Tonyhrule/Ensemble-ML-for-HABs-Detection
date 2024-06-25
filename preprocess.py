import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import openai
import os
from dotenv import load_dotenv
from io import StringIO

load_dotenv()

openai.api_key = 'sk-proj-2K97w5DZf4F5TRizKhRuT3BlbkFJXYbLwgQDueJPUgmUT24j'

def chatgpt_response(messages, model):
    completion = openai.chat.completions.create(model=model, messages=messages)
    return completion.choices[0].message.content


print("Type '/exit' to end the conversation at any time.")


# Load the dataset
data_path = 'Dataset.xlsx'
data = pd.read_excel(data_path)

# Drop rows with missing values in features or target
data = data.dropna(subset=['Temperature', 'Salinity', 'UVB', 'ChlorophyllaFlor'])

messages = [
    {"role": "system", "content": "You are generator that generates additional data that is proportional but not the same to the given dataset. Do not change the values to make it deviate from the original data set too much. The temperature should be between 7 and 10, the salinity should be between 32 and 35, and the UVB should not be greater than 100. Generate 10000 rows."},
    {"role": "user", "content": f"{data}"}
]

model = "gpt-4"

string_data = StringIO(chatgpt_response(messages = messages, model = model))
more_data = pd.read_csv(string_data, sep=';')

columns = ['Temperature', 'Salinity', 'UVB', 'ChlorophyllaFlor']
data = data[columns]
more_data = data[columns]

new_data = pd.concat([data, more_data], axis=0, ignore_index=True)

new_data = pd.concat([data, more_data])

for i in new_data['Temperature']:
    try: 
        row = new_data.iloc[[i]].values.tolist()[0]
    except TypeError:
        new_data = new_data.drop(labels=i, axis=0)

# Select features and target
X = new_data[['Temperature', 'Salinity', 'UVB']]
y = new_data['ChlorophyllaFlor']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train)

# Save the processed data and the scaler
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
joblib.dump((X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test), os.path.join(output_dir, 'processed_data.pkl'))
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

print("Preprocessing complete. Data saved to the output folder.")

