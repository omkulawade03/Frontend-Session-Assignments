import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables from .env file
load_dotenv()

api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN missing in your .env file!")

# 1. Create ChatPromptTemplate with System & Human messages
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a very polite and helpful AI assistant."),
    ("user", "{user_input}")
])

# 2. Initialize the Hugging Face Endpoint
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.7,
    huggingfacehub_api_token=api_token,
)

# 3. Convert to Chat Model wrapper
chat_model = ChatHuggingFace(llm=llm)

# 4. Chain prompt and model together
chain = prompt_template | chat_model

# 5. Take input from user, format prompt, and send to model
user_query = input("Enter your message: ")

response = chain.invoke({"user_input": user_query})

# 6. Print response
print("\n--- Assistant Response ---")
print(response.content)

