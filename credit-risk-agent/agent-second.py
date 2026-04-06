import requests
import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Replace with your actual AWS Invoke URL
API_BASE_URL = "https://ww2vgg54gf.execute-api.us-east-1.amazonaws.com/prod"

def list_pets(pet_type=None):
    # Fetch all pets (since the mock API ignores query params)
    response = requests.get(f"{API_BASE_URL}/pets")
    response.raise_for_status() # Good practice for error handling
    all_pets = response.json()
    
    # If the LLM asked for a specific type, filter the list in Python
    if pet_type:
        filtered_pets = [pet for pet in all_pets if pet.get("type", "").lower() == pet_type.lower()]
        return filtered_pets
        
    # Otherwise, return everything
    return all_pets

def get_pet_by_id(pet_id):
    response = requests.get(f"{API_BASE_URL}/pets/{pet_id}")
    return response.json()

def add_pet(pet_type, price):
    payload = {"type": pet_type, "price": price}
    response = requests.post(f"{API_BASE_URL}/pets", json=payload)
    return response.json()

tools = [
    {
        "type": "function",
        "function": {
            "name": "list_pets",
            "description": "Get a list of pets available in the store.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pet_type": {"type": "string", "description": "Filter by type, e.g. 'dog'"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pet_by_id",
            "description": "Get details for a specific pet using its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pet_id": {"type": "integer", "description": "The unique ID of the pet"}
                },
                "required": ["pet_id"]
            }
        }
    }
]

available_functions = {
    "list_pets": list_pets,
    "get_pet_by_id": get_pet_by_id,
}

model = "llama-3.3-70b-versatile"

def run_agent(user_prompt):
    messages = [
        {"role": "system", "content": "You are a helpful Pet Store assistant."},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        temperature=0,                 
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        messages.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"--- Action: Calling {function_name}({function_args}) ---")
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_response),
            })

        final_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,               
            messages=messages
        )
        return final_response.choices[0].message.content
    
    return response_message.content

print(run_agent("Can you list all the cats in the store?"))