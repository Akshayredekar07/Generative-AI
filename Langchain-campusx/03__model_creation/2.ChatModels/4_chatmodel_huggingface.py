

# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv


# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#     )


# model = ChatHuggingFace(llm=llm)

# result = model.invoke("What is the capital of india")

# print(result.content)


# Import necessary libraries
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables (like the API key) from a .env file
load_dotenv()

# Set up the Hugging Face model using the free Inference API
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # A free model available on Hugging Face
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",    # Add the required model parameter
    task="text-generation",
    max_new_tokens=100,  # Limit the length of the response
    temperature=0,       # Make the output focused and less random
)

# Ask the question
result = llm.invoke("What is the capital of Maharashtra?")

# Print the answer
print(result)