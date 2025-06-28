from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 1. Define the prompt template with a system message, history, and user query
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

# 2. Load previous messages from file
with open("chat_history.txt", "r", encoding="utf-8") as f:
    chat_history = [
        eval(line.strip(), {"HumanMessage": HumanMessage, "AIMessage": AIMessage})
        for line in f if line.strip()
    ]

# 3. Fill the template with actual data
filled_prompt = chat_template.invoke({
    "chat_history": chat_history,
    "query": "Where is my refund?"
})

# 4. Print the full prompt object (not sending to model here)
print("\nComplete Prompt:")
print(filled_prompt)
