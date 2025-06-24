
from langchain_core import prompts
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

prompt1 = PromptTemplate(
    template="Write a tweet on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Write a linkedin post about  {topic}",
    input_variables=['topic']
)

model = ChatOllama(model="llama3.2")
parser = StrOutputParser()


parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic':'AI'})

print(result['tweet'])
print(result['linkedin'])