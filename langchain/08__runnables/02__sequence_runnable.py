
# from langchain_core import prompts
# from langchain_ollama import ChatOllama
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableSequence

# prompt1 = PromptTemplate(
#     template="Wrtie a joke about\n {topic}",
#     input_variables=['topic']
# )

# model = ChatOllama(model="llama3.2")

# parser = StrOutputParser()


# prompt2 = PromptTemplate(
#     template="Explain the following joke {text}",
#     input_variables=['text']
# )


# chain = RunnableSequence(prompt1, model, parser, prompt2,model, parser)

# print(chain.invoke({'topic':'AI'}))


from langchain_core import prompts
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

prompt1 = PromptTemplate(
    template="Write a joke about\n {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Explain the following joke {text}",
    input_variables=['text']
)

model = ChatOllama(model="llama3.2")
parser = StrOutputParser()

# Create individual sequences by chaining runnables
joke_sequence = prompt1 | model | parser
explanation_sequence = prompt2 | model | parser

# Combine the sequences into a parallel runnable
sequence = RunnableParallel(
    {
        "joke": joke_sequence,
        "explanation": explanation_sequence
    }
)

# Invoke the sequence and print both joke and explanation
result = sequence.invoke({'topic': 'AI'})
print("Generated Joke:", result['joke'])
print("Explanation:", result['explanation'])