
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


prompt = PromptTemplate(
    template='Generate 5 interesting fact about {topic}',
    input_variables=['topic']
)

model = ChatOllama(model='gemma3:1b')


parser = StrOutputParser()


chain = prompt | model | parser

result = chain.invoke({'topic':'indain cricket'})

print(result)

chain.get_graph().print_ascii()



