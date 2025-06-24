

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

model = ChatOllama(model="gemma3:1b")

# Create a parser
parser = JsonOutputParser()

template = PromptTemplate(
    template="give me five fact about {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic': 'bloack hole'})


print(result)
print(type(result))