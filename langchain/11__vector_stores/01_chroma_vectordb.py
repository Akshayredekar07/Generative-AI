
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
print("\nEmbeddings initialized successfully.\n")

# Create Document objects for IPL players
doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
    metadata={"team": "Royal Challengers Bangalore"}
)
doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
    metadata={"team": "Mumbai Indians"}
)
doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
    metadata={"team": "Chennai Super Kings"}
)
doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
    metadata={"team": "Mumbai Indians"}
)
doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
    metadata={"team": "Chennai Super Kings"}
)
docs = [doc1, doc2, doc3, doc4, doc5]
print("\nDocuments created:", [doc.page_content for doc in docs], "\n")

# Initialize vector store
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory='chroma_db',
    collection_name='sample'
)
print("\nVector store initialized.\n")

vector_store_doc_ids = vector_store.add_documents(docs)
print("\nDocument IDs after adding:", vector_store_doc_ids, "\n")

all_docs = vector_store.get(include=['embeddings', 'documents', 'metadatas'])
print("\nAll documents in vector store:", all_docs, "\n")

search_results = vector_store.similarity_search(
    query="Who among these are a bowler?",
    k=2
)
print("\nSearch results (top 2):", [(doc.page_content, doc.metadata) for doc in search_results], "\n")

search_with_score = vector_store.similarity_search_with_score(
    query="Who among these are a bowler?",
    k=2
)
print("\nSearch results with scores (top 2):", [(doc.page_content, score) for doc, score in search_with_score], "\n")

# Metadata filtering example
filtered_results = vector_store.similarity_search_with_score(
    query="",  # Empty query relies on metadata filter
    filter={"team": "Chennai Super Kings"}
)
print("\nFiltered results (Chennai Super Kings):", [(doc.page_content, score) for doc, score in filtered_results], "\n")

# Update a document (using the first document ID)
updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata={"team": "Royal Challengers Bangalore"}
)
vector_store.update_document(document_id=vector_store_doc_ids[0], document=updated_doc1)
print("\nDocument with ID {vector_store_doc_ids[0]} updated.\n")

all_docs_after_update = vector_store.get(include=['embeddings', 'documents', 'metadatas'])
print("\nAll documents after update:", all_docs_after_update, "\n")

vector_store.delete(ids=[vector_store_doc_ids[0]])
print("\nDocument with ID {vector_store_doc_ids[0]} deleted.\n")

all_docs_after_delete = vector_store.get(include=['embeddings', 'documents', 'metadatas'])
print("\nAll documents after deletion:", all_docs_after_delete, "\n")