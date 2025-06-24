

from langchain_core import prompts
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

model = ChatOllama(model="llama3.2")
parser = StrOutputParser()

passthrough = RunnablePassthrough()

print(passthrough.invoke(2))
print(passthrough.invoke({'name':'akshay'}))