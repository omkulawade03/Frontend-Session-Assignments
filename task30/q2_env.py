import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
mistral_key = os.getenv("MISTRAL_API_KEY")

if groq_key and mistral_key:
    print("Success: API keys loaded successfully!")
else:
    print("Error: Missing one or more API keys. Check your .env file.")