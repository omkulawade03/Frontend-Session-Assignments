from langchain_core.prompts import ChatPromptTemplate

# RAG Prompt Template
prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Answer the question using ONLY the information provided in the context below.

If the answer is not found in the context, say:
"I could not find the answer."

Context:
{context}

Question:
{question}

Answer:
""")

# Example retrieved context
context = """
TechNova Solutions is headquartered in Pune, Maharashtra, India.
The company provides artificial intelligence, cloud computing,
and software development services.
"""

# User question
question = "Where is TechNova Solutions headquartered?"

# Pass context and question into the prompt
formatted_prompt = prompt.invoke({
    "context": context,
    "question": question
})

print("Formatted Prompt:\n")
print(formatted_prompt.to_string())

