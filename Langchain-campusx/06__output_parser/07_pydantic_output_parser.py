
from socket import TIPC_CRITICAL_IMPORTANCE
from langchain_ollama import ChatOllama 
from langchain_core.prompts import PromptTemplate, prompt
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# model = ChatOllama(model="llama3.2")
model = ChatOllama(model="qwen3:8b")


# 2. Define the output schema (Pydantic model)
class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city person belongs to')

# 3. Create the parser
parser = PydanticOutputParser(pydantic_object=Person)

# 4. Build the PromptTemplate with clear example (important for small models!)
template = PromptTemplate(
    template='Generate the name, age and city of fictional person {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)


chain = template | model | parser

result = chain.invoke({'place':'pune'})

print(result)