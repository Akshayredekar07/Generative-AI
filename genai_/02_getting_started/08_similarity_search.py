import os
from dotenv import load_dotenv
load_dotenv()

# Load data from a webpage
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://www.geeksforgeeks.org/python/fastapi-path-parameters/")
docs = loader.load()

# Split documents into manageable chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

# Convert text chunks into embeddings
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store embeddings in FAISS vector database
from langchain_community.vectorstores import FAISS
vectorstoredb = FAISS.from_documents(documents, embeddings)

# Use Gemini 2.0 Flash as the LLM
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY"))

# Create a QA chain with prompt and documents
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:
<context>
{context}
</context>
""")

document_chain = create_stuff_documents_chain(llm, prompt)

# Connect retriever with document chain
retriever = vectorstoredb.as_retriever()
from langchain.chains import create_retrieval_chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Ask a question
response = retrieval_chain.invoke({"input": "What is dynamic routing in FastAPI"})

# Print result
print("Answer:", response['answer'])
print("\nContext:\n", response['context'])
