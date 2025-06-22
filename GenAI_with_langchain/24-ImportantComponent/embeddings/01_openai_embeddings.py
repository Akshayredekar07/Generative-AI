# Step 1: Import required libraries
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Step 2: Load environment variables from .env file
load_dotenv()

# Step 4: Initialize OpenAI embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Step 5: Create a sample query and generate its embedding
# This demonstrates how to convert text to a vector
text = "This is a test query"
query_result = embeddings.embed_query(text)
# Optional: Print the embedding vector to verify
# print(query_result)


# Step 6: Load text from a file using TextLoader
# Assumes 'speech.txt' exists in the same directory as this script
loader = TextLoader('speech.txt')
docs = loader.load()


# Step 7: Split the loaded document into smaller chunks
# Using RecursiveCharacterTextSplitter with chunk size of 500 and 50-character overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
final_documents = text_splitter.split_documents(docs)


# Step 8: Create a Chroma vector store from the documents and embeddings
# This stores the document embeddings for similarity search
db = Chroma.from_documents(final_documents, embeddings)


# Step 9: Perform a similarity search using a query
# The query searches for documents similar to the provided text
query = "It will be all the easier for us to conduct ourselves as belligerents"
retrieved_results = db.similarity_search(query)


# Step 10: Print the retrieved results
# This will display the matching documents with their content and metadata
print(retrieved_results)