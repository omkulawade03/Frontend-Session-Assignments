from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()

# Initialize the model
llm = ChatGroq(model="llama-3.1-8b-instant")

# Store session histories in a dictionary
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap the model with message history handling
chat_with_history = RunnableWithMessageHistory(
    llm,
    get_session_history
)

# Configuration for the session
config = {"configurable": {"session_id": "user_session_1"}}

print("Memory Chatbot initialized! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    
    # Invoke the model with history config
    response = chat_with_history.invoke(user_input, config=config)
    print(f"Bot: {response.content}\n")


    