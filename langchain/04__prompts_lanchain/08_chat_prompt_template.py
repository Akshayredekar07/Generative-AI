

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Create a prompt template with variables
chat_template = ChatPromptTemplate.from_messages([
    ('system', "You are the helpful {domain} expert."),
    ('human', "Explain in simple terms, what is {topic}?")
])



# Invoke with input values
prompt = chat_template.invoke({
    'domain': 'cricket',
    'topic': 'dusra'
})

# Print the formatted prompt
print(prompt)
