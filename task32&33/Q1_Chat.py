from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# Initialize the LLM
model = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq"
)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{question}")
])

# Create chain
chain = prompt | model

print("===================================")
print("       AI Chat Application")
print("===================================")
print("Type 'exit' to stop the program.\n")

while True:
    user_input = input("You: ")

    if user_input.strip().lower() == "exit":
        print("AI: Goodbye!")
        break

    response = chain.invoke({
        "question": user_input
    })

    print("AI:", response.content)


    