import requests
import json
import os

API_BASE_URL = "https://ww2vgg54gf.execute-api.us-east-1.amazonaws.com/prod"

def list_pets(pet_type=None):
    """
    Get a list of pets available in the store. Filtering is done locally.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/pets")
        response.raise_for_status() 
        all_pets = response.json()
        
        if pet_type:
            filtered_pets = [pet for pet in all_pets if pet.get("type", "").lower() == pet_type.lower()]
            return filtered_pets
        return all_pets
    except Exception as e:
        return {"error": str(e)}

def get_pet_by_id(pet_id):
    """
    Get details for a specific pet using its ID.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/pets/{pet_id}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


petstore_tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_pets",
            "description": "Get a list of pets available in the store.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pet_type": {"type": "string", "description": "Filter by type, e.g. 'dog' or 'cat'"}
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

petstore_functions = {
    "list_pets": list_pets,
    "get_pet_by_id": get_pet_by_id,
}