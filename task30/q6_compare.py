from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI

# Load environment variables from .env
load_dotenv()

# The same question for both models
question = "What are the advantages of using LangChain?"

# Initialize both models
groq_llm = ChatGroq(model="llama-3.1-8b-instant")
mistral_llm = ChatMistralAI(model="mistral-small-2603")

print("Fetching responses from both models, please wait...\n")

# Get responses
groq_response = groq_llm.invoke(question)
mistral_response = mistral_llm.invoke(question)

# Print clearly labeled responses
print("=== Groq Response ===")
print(groq_response.content)

print("\n" + "="*40 + "\n")

print("=== Mistral Response ===")
print(mistral_response.content)


