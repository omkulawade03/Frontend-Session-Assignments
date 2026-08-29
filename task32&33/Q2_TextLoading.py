from langchain_community.document_loaders import TextLoader

# Load the text file
loader = TextLoader("data/company_info.txt")

# Load the document
documents = loader.load()

# Print number of documents
print("Number of documents loaded:", len(documents))

# Print content of the first document
print("\nContent of the first document:")
print(documents[0].page_content)


