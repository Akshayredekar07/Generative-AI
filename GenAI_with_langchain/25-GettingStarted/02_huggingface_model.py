
import os
from dotenv import load_dotenv

load_dotenv()

# os.environ["HF_HOME"] = 'D:/huggingface_cache'

## Langsmith Tracking
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key is None:
    raise ValueError("LANGCHAIN_API_KEY environment variable must be set.")
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

os.environ["LANGCHAIN_TRACING_V2"] = "true"

langchain_project = os.getenv("LANGCHAIN_PROJECT")
if langchain_project is None:
    raise ValueError("LANGCHAIN_PROJECT environment variable must be set.")
os.environ["LANGCHAIN_PROJECT"] = langchain_project



# Import all libraries
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate


llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")
print(result.content)

