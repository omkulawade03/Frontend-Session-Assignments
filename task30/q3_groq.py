from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

# Initialize the chat model using the specified model name
llm = ChatGroq(model="llama-3.1-8b-instant")

# Send the prompt
response = llm.invoke("Introduce yourself in 3 sentences.")

# Print the response content clearly
print("Groq Response:\n", response.content)


