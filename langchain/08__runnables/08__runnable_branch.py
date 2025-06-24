
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

# Define prompts
prompt1 = PromptTemplate(
    template="Write a detailed report on\n {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Summarize the following text {text}",
    input_variables=['text']
)

# Initialize model and parser
model = ChatOllama(model="mistral")
parser = StrOutputParser()

# Generate report chain
report_gen_chain = prompt1 | model | parser

# Summarization chain: transform string to dict, then process
summarization_chain = RunnableLambda(lambda x: {'text': x}) | prompt2 | model | parser

# Branching logic: summarize if > 500 words, else pass through
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, summarization_chain),  # If true, summarize
    RunnablePassthrough()                                   # Else, keep as is
)

# Final chain: report generation followed by conditional summarization
final_chain = report_gen_chain | branch_chain

# Execute the chain
result = final_chain.invoke({'topic': 'Russia vs Ukraine'})
print(result)