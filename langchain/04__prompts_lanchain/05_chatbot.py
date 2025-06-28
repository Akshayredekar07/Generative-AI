

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Chat history should be in the format of a list of dicts
chat_history = []

while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        print("Goodbye!")
        break

    chat_history.append({"role": "user", "content": user_input})

    # Invoke the model with the structured message history
    result = model.invoke(chat_history)

    print("AI:", result.content)

    chat_history.append({"role": "assistant", "content": result.content})

print(chat_history)