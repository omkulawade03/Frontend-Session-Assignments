from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load existing Chroma database
vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embeddings
)

print("Chroma database loaded successfully!")

# 3. Create retriever using MMR
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)

# 4. Ask a question
question = "Where is TechNova Solutions headquartered?"

# 5. Retrieve relevant documents
results = retriever.invoke(question)

print("\nQuestion:")
print(question)

print("\nRetrieved Documents:")

for i, doc in enumerate(results, start=1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)



    