#!/usr/bin/env python3
"""Test which models are available on NVIDIA NIM"""

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

# List of models to test
models_to_test = [
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-405b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/llama-3.1-nemotron-51b-instruct",
    "deepseek/deepseek-v3",
    "deepseek/deepseek-r1",
    "qwen/qwen2.5-72b-instruct",
    "meta/llama3-8b-instruct",
    "meta/llama3-70b-instruct",
]

print("\nTesting models...")
print("-" * 50)

for model in models_to_test:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Say 'yes' if you work"}
            ],
            max_tokens=10,
            temperature=0.0
        )
        print(f"✅ {model}: WORKS")
        print(f"   Response: {response.choices[0].message.content[:50]}")
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            print(f"❌ {model}: NOT FOUND")
        elif "401" in error_msg or "403" in error_msg:
            print(f"⚠️  {model}: ACCESS DENIED")
        else:
            print(f"❌ {model}: ERROR - {error_msg[:100]}")

print("\nTrying to list models endpoint...")
try:
    models = client.models.list()
    print("\nAvailable models:")
    for model in models.data[:10]:  # Show first 10
        print(f"  - {model.id}")
except Exception as e:
    print(f"Could not list models: {e}")