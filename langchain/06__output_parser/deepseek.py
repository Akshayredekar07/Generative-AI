
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    model="deepseek-ai/DeepSeek-R1",
    provider="together"
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of India")
print(result.content)
