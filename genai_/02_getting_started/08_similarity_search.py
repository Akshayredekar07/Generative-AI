
import os
from dotenv import load_dotenv
load_dotenv()


# Data Ingestion: Scrape data from a website using WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader

# Initialize the loader with the target URL
loader = WebBaseLoader("https://docs.smith.langchain.com/tutorials/Administrators/manage_spend")

# Load the documents
docs = loader.load()

# Split documents into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize the text splitter with chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split the loaded documents into smaller chunks
documents = text_splitter.split_documents(docs)

# Create vector embeddings for the document chunks using Hugging Face
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize Hugging Face embeddings with a sentence-transformers model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a FAISS vector store from the documents and embeddings
from langchain_community.vectorstores import FAISS

vectorstoredb = FAISS.from_documents(documents, embeddings)

# Initialize the LLM (Gemini 2.0 Flash model via Google Generative AI)
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize the Gemini 2.0 Flash model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY"))

# Create a document chain for answering questions based on context
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
"""
Answer the following question based only on the provided context:
<context>
{context}
</context>
"""
)

# Create the document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create a retriever from the vector store
retriever = vectorstoredb.as_retriever()

# Create a retrieval chain combining the retriever and document chain
from langchain.chains import create_retrieval_chain

retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Example: Invoke the retrieval chain with a question
response = retrieval_chain.invoke({"input": "LangSmith has two usage limits: total traces and extended"})

# Print the answer
print(response['answer'])

# Optionally, print the context documents retrieved
print(response['context'])
