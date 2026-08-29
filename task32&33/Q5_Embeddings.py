from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load document
loader = TextLoader("data/company_info.txt", encoding="utf-8")
documents = loader.load()

print("Documents loaded:", len(documents))

# 2. Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))

# Check that chunks were created
if not chunks:
    print("Error: No chunks were created.")
    exit()

# Display first chunk
print("\nFirst chunk:")
print(chunks[0].page_content)

# 3. Initialize HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Convert one chunk into embedding
vector = embeddings.embed_query(chunks[0].page_content)

# 5. Print embedding dimension
print("\nEmbedding dimension:", len(vector))