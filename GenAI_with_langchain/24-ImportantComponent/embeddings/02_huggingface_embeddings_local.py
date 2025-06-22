import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

text = "This is a test query"
query_result = embeddings.embed_query(text)

loader = TextLoader('speech.txt')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
final_documents = text_splitter.split_documents(docs)

db = Chroma.from_documents(final_documents, embeddings)

query = "It will be all the easier for us to conduct ourselves as belligerents"
retrieved_results = db.similarity_search(query)

print(retrieved_results)