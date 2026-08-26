import torch
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

print("Loading local model into memory... (This may take a moment)")

# 1. Load local model using HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 150,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
    },
)

# 2. Convert to Chat Model wrapper
local_chat_model = ChatHuggingFace(llm=llm)

# 3. Create question prompt
messages = [
    HumanMessage(content="Introduce Yourself in 100 words")
]

# 4. Invoke model and print response
print("\n--- Generating Response ---")
response = local_chat_model.invoke(messages)

print("\n--- Local Model Response ---")
print(response.content)


