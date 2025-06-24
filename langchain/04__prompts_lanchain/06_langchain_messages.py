

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Initialize model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Prepare messages
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Tell me about langchain"),
]

# Invoke the model
result = model.invoke(messages)

# Append the model's response
messages.append(AIMessage(content=result.content))

# Print the full conversation
for msg in messages:
    print(f"{msg.__class__.__name__}: {msg.content}")
