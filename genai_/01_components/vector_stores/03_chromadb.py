
# Chroma is the open-source AI application database.
# Embeddings, vector search, document storage, full-text search, metadata filtering, and multi-modal. 
# All in one place. Retrieval that just works. As it should be.


# building the sample vectordb
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from dotenv import load_dotenv
import os


# load_dotenv()


os.environ['HF_HOME'] = 'D:/huggingface_cache'


# token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# if token is None:
#     raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable must be set.")
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = token


loader = TextLoader("speech.txt")
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(documents=documents)


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(documents=splits, embedding=embedding)
# vectordb = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory="./chroma_db")

# send the query
query = "What does the speaker believe is the main reason the United States should enter the war?"

docs = vectordb.similarity_search(query=query)

# print(docs[0].page_content)

# saving to the disk
# vectordb = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory="./chroma_db")

# Load from disk
db2 = Chroma(persist_directory="./chroma_db",  embedding_function=embedding)

docs_with_scores = db2.similarity_search_with_score(query=query)

# Check if any documents are found
if not docs_with_scores:
    print("No documents found.")
else:
    # Iterate and print each document and its score
    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        print(f"\nDocument {i}:\n")
        print(doc.page_content)
        print(f"\nSimilarity score: {score}\n{'-'*40}")



# Retriver

retriver = db2.as_retriever()

print(f"Retriver content\n {retriver.invoke(query)[0].page_content}")



