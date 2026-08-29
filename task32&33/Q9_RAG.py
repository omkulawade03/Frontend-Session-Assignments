from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


# ============================================================
# 1. Load existing Chroma Vector Store
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
# 2. Create Retriever using MMR
# ============================================================

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)


# ============================================================
# 3. Create RAG Prompt
# ============================================================

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Answer the question using ONLY the information provided
in the context below.

Do not use outside knowledge.

If the answer is not found in the context, say exactly:
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

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)

    # Combine retrieved documents into context
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # Create formatted prompt
    formatted_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    # Generate answer
    response = llm.invoke(formatted_prompt)

    return response.content


# ============================================================
# 6. Test Questions
# ============================================================

questions = [
    "Where is TechNova Solutions headquartered?",
    "What is the main product of TechNova Solutions?",
    "What is the capital of France?"
]


# ============================================================
# 7. Display Results
# ============================================================

for question in questions:

    print("\n" + "=" * 60)
    print("Question:", question)

    answer = ask_question(question)

    print("Answer:", answer)


    