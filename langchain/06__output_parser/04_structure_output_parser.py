
from langchain_ollama import ChatOllama 
from langchain_core.prompts import PromptTemplate, prompt
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

model = ChatOllama(model="gemma3:1b")

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]


parser = StructuredOutputParser.from_response_schemas(schema)


template = PromptTemplate(
    template = 'give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt = template.invoke({'topic':'quant finance'})

result = model.invoke(prompt)

final_result = parser.parse(str(result.content))

print(final_result)