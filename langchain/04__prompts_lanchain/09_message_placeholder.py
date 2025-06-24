


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])


chat_history = []

with open("chat_history.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            msg = eval(line, {"HumanMessage": HumanMessage, "AIMessage": AIMessage})
            chat_history.append(msg)



prompt = chat_template.invoke({
    "chat_history": chat_history,
    "query": "Where is my refund?"
})

# Step 3: Send prompt to model (if you have one hooked)
# result = model.invoke(prompt.messages)
# print("AI:", result.content)



print("\nComplete Prompt:")
print(prompt) 
