
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


model = ChatOllama(model="gemma3:1b")

# Create a parser
parser = JsonOutputParser()

template = PromptTemplate(
    template="give me the name, age and city of a fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt = template.format()

result = model.invoke(prompt)

final_result = parser.parse(str(result.content))

# print(final_result)

print(final_result['name'])
print(type(final_result))