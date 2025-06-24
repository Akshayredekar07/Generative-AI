
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


# Set HF cache
os.environ['HF_HOME'] = 'D:/huggingface_cache'


token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if token is None:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable must be set.")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = token


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "This is the llm from huggingface"

result = embeddings.embed_query(text=text)

print(len(result))
print(result[:5])