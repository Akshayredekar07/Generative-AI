
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    # model="mistralai/Mistral-7B-Instruct-v0.3", 
    model="microsoft/Phi-3-mini-4k-instruct", 
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Wrtie a summary of the following poem\n - {poem}",
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader("cricket.txt", encoding="utf-8")
docs = loader.load()

# print(type(docs))
# print(len(docs))

# print(type(docs[0]))

# print(docs[0].page_content)
# print(docs[0].metadata)


chain = prompt | model | parser

print(chain.invoke({'poem': docs[0].page_content}))