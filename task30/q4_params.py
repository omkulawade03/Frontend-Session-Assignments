from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

# Initialize the Groq model with modified parameters
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=50
)

# Send the same prompt as Q3
response = llm.invoke("Introduce yourself in 3 sentences.")

# Print the constrained response content
print("Constrained Groq Response:\n", response.content)


