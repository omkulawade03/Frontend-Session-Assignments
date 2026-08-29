from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader

# Load PDF
pdf_loader = PyPDFLoader("data/Python_Full_Notes.pdf")
pdf_documents = pdf_loader.load()

print("Number of PDF pages loaded:", len(pdf_documents))

# Load Wikipedia page
web_loader = WebBaseLoader(
    "https://en.wikipedia.org/wiki/Artificial_intelligence"
)

web_documents = web_loader.load()

print("Number of Wikipedia documents loaded:", len(web_documents))