import os
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline

load_dotenv()

api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is missing in your .env file!")

question = [HumanMessage(content="What is the difference between AI and Machine Learning?")]

# 1. API Model Setup (DeepSeek-R1)
print("Initializing API Model...")
api_llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.7,
    huggingfacehub_api_token=api_token,
)
api_chat = ChatHuggingFace(llm=api_llm)

# 2. Local Model Setup (TinyLlama)
print("Loading Local Model into memory...")
local_llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 150,
        "do_sample": True,
        "temperature": 0.7,
    },
)
local_chat = ChatHuggingFace(llm=local_llm)

# Execute API Model
print("\n=== Querying API Model ===")
start_time = time.time()
api_response = api_chat.invoke(question)
api_duration = time.time() - start_time

# Execute Local Model
print("=== Querying Local Model ===")
start_time = time.time()
local_response = local_chat.invoke(question)
local_duration = time.time() - start_time

# Display Results
print("\n" + "="*40)
print(f"API Model Response (Time taken: {api_duration:.2f}s)")
print("="*40)
print(api_response.content)

print("\n" + "="*40)
print(f"Local Model Response (Time taken: {local_duration:.2f}s)")
print("="*40)
print(local_response.content)

# Observation
print("\n" + "="*40)
print("Observation Summary")
print("="*40)
print(
    "Observation:\n"
    "1. Quality: The API Model (DeepSeek-R1) yields a much more structured, comprehensive, "
    "and nuanced answer due to its significantly larger parameter count. The Local Model "
    "(TinyLlama-1.1B) generates a much simpler, basic definition.\n"
    "2. Speed: The API model handles generation on cloud servers without consuming local resources. "
    "The local model avoids network calls after loading, but generation speed depends entirely on your local machine's hardware."
)