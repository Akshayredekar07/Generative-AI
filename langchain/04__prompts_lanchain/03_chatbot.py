
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)


while True:
    user_input = input("You: ")
    if user_input == 'exit':
        break 
    result = model.invoke(user_input)
    print("AI: ", result.content)




