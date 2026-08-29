import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_vector_db():
    data_path = "data/company_info.txt"
    persist_dir = "chroma-db"

    # Create dummy data if directory/file doesn't exist
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(data_path):
        with open(data_path, "w") as f:
            f.write(
                "Acme Corp Policy Manual\n"
                "1. Work Hours: Standard work hours are 9:00 AM to 5:00 PM EST.\n"
                "2. Remote Work: Employees are allowed to work remotely 2 days per week.\n"
                "3. Leave Policy: Employees receive 20 days of paid leave annually."
            )

    # 1. Load Document
    loader = TextLoader(data_path)
    documents = loader.load()

    # 2. Split Document into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # 3. Create Embeddings Model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Save to Chroma DB
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"Vector DB created successfully in folder '{persist_dir}'.")

if __name__ == "__main__":
    build_vector_db()