
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


llm = OpenAI(model="gpt-3.5-turbo-instruct")

result = llm.invoke("What is the capital of Maharashtra")

<<<<<<< HEAD
print(result)


=======
print(result)
>>>>>>> 0cb547da614276a66d850a151d3917e178cab207
