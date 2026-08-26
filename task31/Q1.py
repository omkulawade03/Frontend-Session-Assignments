import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# Verify token is loaded properly
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file!")

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    max_new_tokens=256,
    huggingfacehub_api_token=api_token,  # Pass the loaded token variable
)

chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke([HumanMessage(content="Please introduce yourself.")])
print(response.content)


