
from socket import TIPC_CRITICAL_IMPORTANCE
from langchain_ollama import ChatOllama 
from langchain_core.prompts import PromptTemplate, prompt
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


model = ChatOllama(model="llama3.2")


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


# 5. Format the prompt
prompt = template.format(place='India')
print(prompt)

# 6. Invoke the model
result = model.invoke(prompt)

output = parser.parse(str(result.content))

print(output)