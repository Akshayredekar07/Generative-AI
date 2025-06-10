

from langchain_xai import ChatXAI
from dotenv import load_dotenv


load_dotenv()


llm = ChatXAI(
    model="grok-beta",  # Use Grok model (grok-beta is available as of my last update)
    temperature=0      
)

result = llm.invoke("What is the capital of Maharashtra")


print(result.content)