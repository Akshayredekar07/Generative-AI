
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

# Initialize the model
model = ChatOllama(model='gemma3:1b')

# Define the Pydantic model for feedback sentiment
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

# Initialize the Pydantic output parser
parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Create a more explicit prompt to ensure the model returns only a JSON object
prompt1 = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback text as either 'positive' or 'negative'. "
        "Return the result as a JSON object with a single key 'sentiment' and the value being either 'positive' or 'negative'. "
        "Do not include any additional text, explanations, or code snippets. Only return the JSON object.\n\n"
        "Feedback: {feedback}\n\n"
        "Format: {format_instruction}"
    ),
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

# Create the chain
classifier_chain = prompt1 | model | parser2

# Invoke the chain with the feedback and handle potential errors
try:
    result = classifier_chain.invoke({'feedback': 'This is a wonderful smartphone'})
    print(result.sentiment)
except Exception as e:
    print(f"Error: {e}")
    # For debugging: print the raw model output
    raw_output = model.invoke(prompt1.format(feedback='This is a wonderful smartphone'))
    print(f"Raw model output: {raw_output}")