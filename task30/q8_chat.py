from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize the chat model
llm = ChatGroq(model="llama-3.1-8b-instant")

print("Chatbot initialized! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    
    # Check for exit condition
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    
    # Get and print response
    response = llm.invoke(user_input)
    print(f"Bot: {response.content}\n")


    