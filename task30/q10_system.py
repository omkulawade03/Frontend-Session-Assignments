from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()

# Initialize the model
llm = ChatGroq(model="llama-3.1-8b-instant")

# Create a prompt template with a System Persona
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sarcastic and witty AI assistant. Always add a witty remark before answering any question."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Combine prompt and model into a chain
chain = prompt | llm

# Store session histories
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap the chain with message history
chat_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Configuration for the session
config = {"configurable": {"session_id": "user_session_1"}}

print("Persona Chatbot initialized! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    
    # Invoke the chain with history config
    response = chat_with_history.invoke({"input": user_input}, config=config)
    print(f"Bot: {response.content}\n")


    