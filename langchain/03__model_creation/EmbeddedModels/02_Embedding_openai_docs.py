
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

documents = [
    "Delhi is the capital of india",
    "Mumbai is he capital of Maharashtra",
    "Nagpur is the subcapital of maharashtra"
]

result = embedding.embed_documents(documents)

print(str(result))