#!/usr/bin/env python3
"""Test NVIDIA NIM API connection"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Get the NIM API key
nvidia_nim_api_key = os.getenv('NVIDIA_NIM_API_KEY', '')

if not nvidia_nim_api_key:
    print("Error: NVIDIA_NIM_API_KEY not found in .env file")
    exit(1)

print(f"API Key found: {nvidia_nim_api_key[:10]}...")

# Create OpenAI client with NIM endpoint
client = OpenAI(
    api_key=nvidia_nim_api_key,
    base_url="https://integrate.api.nvidia.com/v1"
)

# Test with a simple prompt
try:
    print("\nTesting NIM API connection...")
    
    # Try with a generic model first
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-70b-instruct",
        messages=[
            {"role": "user", "content": "What is 2 + 2?"}
        ],
        max_tokens=100,
        temperature=0.0
    )
    
    print(f"Success! Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying to list available models...")
    
    # Try to list models
    try:
        models = client.models.list()
        print("Available models:")
        for model in models.data:
            print(f"  - {model.id}")
    except Exception as e2:
        print(f"Could not list models: {e2}")