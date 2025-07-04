
# Import necessary libraries
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()

# Set HuggingFace cache
os.environ['HF_HOME'] = 'D:/huggingface_cache'

# Define the documents
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# Initialize the language model
# llm = ChatGroq(model="Llama3-8b-8192")
llm = ChatOllama(model="llama3.2:1b")


# Initialize embeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create vector store from documents
vectorstore = Chroma.from_documents(documents, embeddings)

# Define a function to format retrieved documents into a string
def format_context(documents: List[Document]) -> str:
    return "\n".join(doc.page_content for doc in documents)

# Create the retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 1})

# Define the prompt template
message = """
Answer the question using only the provided context. Do not use any external knowledge or make assumptions beyond the context.

Question: {question}

Context: {context}
"""
prompt = ChatPromptTemplate.from_template(message)

# Create the RAG chain
rag_chain = {
    "context": retriever | format_context,
    "question": RunnablePassthrough()
} | prompt | llm

# Invoke the chain with the query
response = rag_chain.invoke("Tell me about dogs")

# Print the response
print(response.content)