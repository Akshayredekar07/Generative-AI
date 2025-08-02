from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

try:
    llm = ChatGroq(model="moonshotai/kimi-k2-instruct")
    response = llm.invoke([
        HumanMessage(content="What is the capital of Maharashtra?")
    ])
    print(response.content)

except Exception as e:
    print(f"An error occurred: {str(e)}")