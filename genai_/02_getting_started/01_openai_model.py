
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_HOME"] = 'D:/huggingface_cache'

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if token is None:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable must be set.")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

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
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# Input and response from LLM

result = llm.invoke("What is generative AI")

print(result.content)


