from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the PDF
loader = PyPDFLoader("data/Python_Full_Notes.pdf")
documents = loader.load()

print("Number of PDF pages loaded:", len(documents))

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split documents into chunks
chunks = text_splitter.split_documents(documents)

# Print total number of chunks
print("Total number of chunks created:", len(chunks))

# Display first 3 chunks
print("\nFirst 3 chunks:\n")

for i, chunk in enumerate(chunks[:3], start=1):
    print(f"--- Chunk {i} ---")
    print(chunk.page_content)
    print()


    