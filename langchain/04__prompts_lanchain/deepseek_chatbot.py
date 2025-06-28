from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/deepseek-coder-6.7b-instruct",  # or another DeepSeek HF model
    model="deepseek-ai/deepseek-coder-6.7b-instruct",  # or another DeepSeek HF model
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Message-based chat history
chat_history = []

# Add initial system message
chat_history.append(SystemMessage(content="You are a helpful assistant."))

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        print("Goodbye!")
        break

    # Add the user message
    chat_history.append(HumanMessage(content=user_input))

    # Get response
    result = model.invoke(chat_history)

    # Display and save the AI response
    print("AI:", result.content)
    chat_history.append(AIMessage(content=result.content))

# Print the full conversation (optional)
print("\n\n", chat_history, "\n\n")
for msg in chat_history:
    print(f"{msg.__class__.__name__}: {msg.content}","\n")
