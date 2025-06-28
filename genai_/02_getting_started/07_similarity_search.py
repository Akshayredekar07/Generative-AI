
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Data ingestion from website
loader = WebBaseLoader("https://docs.smith.langchain.com/tutorials/Administrators/manage_spend")
print(loader)  # Print loader object for verification

# Load documents
docs = loader.load()
print(docs)  # Print loaded documents for verification

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)
print(documents)  # Print split documents for verification

# Initialize embeddings
embeddings = OllamaEmbeddings(
    model="llama3.2",
)

# Create vector store
vectorstoredb = FAISS.from_documents(documents, embeddings)
print(vectorstoredb)  # Print vector store object for verification

# Query
query = "LangSmith has two usage limits: total traces and extened"
result = vectorstoredb.similarity_search(query)
result[0].page_content


