from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

# Load environment variables from .env
load_dotenv()

# Initialize the Mistral model using LangChain
llm = ChatMistralAI(model="mistral-small-2603")

# Send the prompt
response = llm.invoke("Explain what is Artificial Intelligence in simple words.")

# Print the complete response
print("Mistral Response:\n", response.content)

