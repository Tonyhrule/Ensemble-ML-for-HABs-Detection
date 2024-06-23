import pprint
import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPEN_AI_KEY")

def add_message(messages, role, content):
    messages.append({"role": role, "content": content})
    return messages

def chatgpt_response(messages, model):
    completion = openai.chat.completions.create(model=model, messages=messages)
    return completion.choices[0].message.content

total_tokens = 0
cost = 0

messages = [
    {"role": "system", "content": "You are a calculator that find the probability of Harmful Algal Blooms given the Chlorophyll A Corrected Value. "},
    {"role": "user", "content": "What is the Chlorophyll A Corrected Value"}
]

# Model selection
while True:
    model = input('Would you like to use GPT 3.5 or GPT 4? Type 3.5 for GPT 3.5 Turbo or 4 for GPT 4.0: ')
    if model == "3.5":
        model = "gpt-3.5-turbo"
        break
    elif model == "4":
        model = "gpt-4"
        break
    else:
        print("You need to type either 3.5 or 4.")

print("Type '/exit' to end the conversation at any time.")

# Conversation loop
while True:
    pprint.pprint(messages[-1]['content'])
    user_message = input("You: ")
    if user_message.lower() == "/exit":
        break
    messages = add_message(messages, "user", user_message)
    response = chatgpt_response(messages, model)
    messages = add_message(messages, "assistant", response)

# Final response for closing
response = openai.chat.completions.create(model=model, messages=messages)