
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
import numpy as np
import os

# Load environment variables from .env file
load_dotenv()

# Set Hugging Face cache directory
os.environ['HF_HOME'] = 'D:/huggingface_cache'

# Retrieve the Hugging Face API token from environment variable
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if token is None:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable must be set.")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

# Initialize and load the document
loader = TextLoader("speech.txt")
documents = loader.load()

# Split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
docs = text_splitter.split_documents(documents=documents)
print(f"Number of documents after splitting: {len(docs)}")

# Initialize Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS vector store from documents
db = FAISS.from_documents(docs, embeddings)

# Output the vector store info
print(db)


# Access the internal FAISS index to see stored vectors
index = db.index

# Reconstruct all vectors stored
vectors = index.reconstruct_n(0, index.ntotal)

# Print details
print("Total vectors stored:", vectors.shape)
print("First vector:", vectors[0])

# Optionally, print the first few vectors
for i in range(min(3, vectors.shape[0])):
    print(f"Vector {i}: {vectors[i]}")