
from langchain_ollama import ChatOllama

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


model = ChatOllama(model="gemma3:1b")



# 1st prompt - detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)
template2 = PromptTemplate(
    template="Write a 5‑line summary on the following text:\n{text}",
    input_variables=['text']
)


# Create  a parser
parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'Agentic AI'})
print(result)