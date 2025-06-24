

import os
from dotenv import load_dotenv

load_dotenv()

langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key is None:
    raise ValueError("LANGCHAIN_API_KEY environment variable must be set.")
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

os.environ["LANGCHAIN_TRACING_V2"] = "true"

langchain_project = os.getenv("LANGCHAIN_PROJECT")
if langchain_project is None:
    raise ValueError("LANGCHAIN_PROJECT environment variable must be set.")
os.environ["LANGCHAIN_PROJECT"] = langchain_project

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate

llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)

chat_model = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI engineer. Please provide an answer based on the question."),
    ("user", "{input}")
])

chain = prompt | chat_model

response = chain.invoke({"input": "Can you please tell me about agentic AI?"})

print("Response:")
print(response.content)

