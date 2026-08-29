from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


# ============================================================
# 1. Load Existing Chroma Database
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embeddings
)

print("Chroma database loaded successfully!")


# ============================================================
# 2. Create Retriever
# ============================================================

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)


# ============================================================
# 3. RAG Prompt
# ============================================================

prompt = ChatPromptTemplate.from_template("""
You are TechNova Solutions' knowledge-base assistant.

Answer the user's question ONLY using the information
provided in the context.

Do not use outside knowledge.

If the answer cannot be found in the context, say exactly:

"I could not find the answer."

Context:
{context}

Question:
{question}

Answer:
""")


# ============================================================
# 4. Initialize Groq LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ============================================================
# 5. RAG Function
# ============================================================

def ask_question(question):

    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    formatted_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(formatted_prompt)

    return response.content


# ============================================================
# 6. Continuous Chat Interface
# ============================================================

print("\n========================================")
print("   TechNova Solutions RAG Chatbot")
print("========================================")
print("Type 'exit' to quit.\n")


while True:

    question = input("You: ")

    if question.lower().strip() == "exit":
        print("Chatbot: Goodbye!")
        break

    if not question.strip():
        continue

    answer = ask_question(question)

    print("\nChatbot:", answer)
    print()

    