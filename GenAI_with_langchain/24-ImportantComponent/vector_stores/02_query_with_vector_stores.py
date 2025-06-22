


from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
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

# # Output the vector store info
# print(db)

query = "how does the speaker describe the desired outcome of the war?"

result = db.similarity_search(query=query)

# print(result[0].page_content)


# Retrive the result 
retriver = db.as_retriever()

ret_docs = retriver.invoke(query)
# print(ret_docs[0].page_content)


# Similarity search and scores
dcos_and_scores = db.similarity_search_with_score(query=query)
# print(dcos_and_scores[0])


# pass direclty vector
embedding_vector=embeddings.embed_query(query)

# result = db.similarity_search_by_vector(embedding_vector)
# print(result[0])

# Saving and loading
# db.save_local("faiss_index")


# load the folder
new_db = new_df=FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)

similarity_search_result = new_db.similarity_search(query=query) 