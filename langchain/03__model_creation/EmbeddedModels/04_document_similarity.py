import os
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from pydantic import SecretStr

load_dotenv()

documents = [
    Document(page_content="Mumbai is the financial capital of India."),
    Document(page_content="Delhi is the capital city of India."),
    Document(page_content="Bangalore is known as the Silicon Valley of India."),
    Document(page_content="Chennai is a coastal city in South India.")
]
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=SecretStr(os.getenv("GOOGLE_API_KEY") or "")
)


vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)

query = "What is the capital of India?"
result = vectorstore.similarity_search(query=query, k=2)

print(type(result))
print(len(result))
print(result[0])
print(result[1])