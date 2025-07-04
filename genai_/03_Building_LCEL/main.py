from pydantic import BaseModel, Field
# from pydantic.v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
from dotenv import load_dotenv
import os

load_dotenv()

# Define the input schema for the chain
class TranslationInput(BaseModel):
    language: str = Field(..., description="The target language for translation")
    text: str = Field(..., description="The text to translate")

# Initialize the model
model = ChatGroq(model="gemma2-9b-it")

# Define the prompt template
system_template = "Translate the following into this {language}"
prompt = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', "{text}")
])

# Define the parser
parser = StrOutputParser()

# Create the chain
chain = prompt | model | parser

# Create the FastAPI app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="API server using langchain runnable interfaces"
)

# Add routes with explicit input schema
add_routes(
    app,
    chain,
    path="/chain",
    input_type=TranslationInput  # Explicitly specify the input schema
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)